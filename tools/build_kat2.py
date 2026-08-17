#!/usr/bin/env python
# KAT-CQ2: Q8_0 control-plane rebake of KAT-CQ1.
#
# PURPOSE: KAT-CQ1's F16 control plane trips this llama.cpp build's CUDA
# binbcast assert (nb10 % sizeof(src1_t)) on GDN tensors. Official Qwen3.6
# Q4_K_M layout (Q8_0 attn matmuls, F32 norms/routers) runs clean on the
# same build. CQ2 = official-parity dtypes, identical expert weights.
#
# KEY DECISIONS
# - Expert tensors BYTE-COPIED from KAT-CQ1.gguf (already our validated Q4_K
#   RTN, same provenance) — no re-streaming, no re-quantization.
# - Control plane from D:/merge/cp/kat BF16 dumps -> Q8_0 for matmul tensors
#   (official parity), F32 for norms/routers/ssm state.
# - Q8_0 reference: block=32, {fp16 d, int8 qs[32]}, d=amax/127
#   (llama.cpp quantize_row_q8_0_ref).
# - DeltaNet transforms reused verbatim from dnet_transform (A_log negation,
#   conv1d V-channel reorder) — validated in the CQ1 build.
#
# BUG FIXES / HISTORY
# - [2026-08-17] Initial. Fixes CQ1 CUDA assert; also halves control-plane
#   bytes vs F16 (VRAM win for spark residency).
import struct, sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0 as BG
from build_gguf import SrcGGUF
from dnet_transform import transform_for, t_norm

CQ1    = "D:/merge/out/KAT-CQ1.gguf"
OUT    = "D:/merge/out/KAT-CQ2.gguf"
CP_DIR = "D:/merge/cp/kat"
ALIGN  = 32

T_F32, T_Q8_0, T_Q4_K = 0, 8, 12

def q8b(x):
    """llama.cpp Q8_0: blocks of 32 -> {fp16 d, int8 qs[32]} (34B/block)."""
    x = np.ascontiguousarray(x, dtype=np.float32).ravel()
    assert x.size % 32 == 0, x.size
    n = x.size // 32
    xb = x.reshape(n, 32)
    amax = np.abs(xb).max(axis=1)
    # BUGFIX [2026-08-17]: subnormal BF16 weights (amax ~1e-40) made d
 # subnormal => 1/d overflowed to inf => rint(inf) cast garbage. Floor d
    # at FLT_MIN normal; subnormal-tiny values quantize to 0 (harmless).
    d = np.maximum(np.where(amax > 0, amax / 127.0, 1.0), 1.1754944e-38)
    d = np.where(np.isfinite(d), d, 1.0)   # NaN/Inf blocks -> unit scale, q=0
    q = np.rint(xb * (1.0 / d)[:, None]).astype(np.int8)
    blk = np.zeros((n, 34), dtype=np.uint8)
    blk[:, :2] = d.astype("<f2").view(np.uint8).reshape(n, 2)
    blk[:, 2:] = q.view(np.uint8).reshape(n, 32)
    return blk.tobytes()

def main():
    t0 = time.time()
    tpl = SrcGGUF(CQ1)          # header template from CQ1 (same tensor set)
    names = list(tpl.entries)
    print(f"template tensors: {len(names)}")

    def base_of(n):
        return n.split(".", 2)[2].replace(".weight", "").replace(".bias", "") if n.startswith("blk.") else n

    def is_exp(n):
        return "exps" in n and "shexp" not in n

    def out_type(n, prev_t):
        b = base_of(n)
        if is_exp(n):
            return prev_t              # byte-copy: keep CQ1 Q4_K type id
        if b in ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
                 "attn_q", "attn_k", "attn_v", "attn_output",
                 "attn_qkv", "attn_gate", "ssm_out") \
                or n in ("token_embd.weight", "output.weight"):
            return T_Q8_0
        return T_F32

    # byte sizes: experts keep CQ1's own nbytes (already on disk); cp recomputed
    def nbytes_for(n, nt, prev_t, prev_nby):
        if nt == prev_t:
            return prev_nby
        return BG.tensor_nbytes(tpl.entries[n][0], nt)

    offs, cur = {}, 0
    for n in names:
        dims, prev_t, prev_off, tp, op = tpl.entries[n]
        # prev nbytes from CQ1's own layout: next-aligned delta is unknown;
        # recompute: for experts prev type is Q4_K => tensor_nbytes(dims, Q4_K)
        nby = BG.tensor_nbytes(dims, out_type(n, prev_t))
        offs[n] = (cur, nby)
        cur = (cur + nby + ALIGN - 1) // ALIGN * ALIGN
    print(f"plan: data {cur/2**30:.2f} GiB (CQ1 20.05 GiB target ~14)", flush=True)

    P = "model.language_model."
    n_cp = n_exp = 0
    cq1 = open(CQ1, "rb")
    with open(OUT, "wb") as out:
        out.write(tpl.header[:tpl.header_len])
        for n in names:
            dims, prev_t, prev_off, type_pos, off_pos = tpl.entries[n]
            nt = out_type(n, prev_t)
            out.seek(type_pos); out.write(struct.pack("<I", nt))
            out.seek(off_pos);  out.write(struct.pack("<Q", offs[n][0]))
        for i, n in enumerate(names):
            dims, prev_t, prev_off, type_pos, off_pos = tpl.entries[n]
            nt = out_type(n, prev_t)
            out.seek(tpl.header_len + offs[n][0])
            if is_exp(n):
                sz = offs[n][1]
                cq1.seek(tpl.header_len + prev_off)
                remaining = sz
                while remaining > 0:
                    chunk = cq1.read(min(1 << 24, remaining))
                    if not chunk:
                        raise RuntimeError(f"short read {n}")
                    out.write(chunk); remaining -= len(chunk)
                n_exp += 1
            else:
                _, tf = transform_for(n)  # returns (base, fn_or_None)
                if n == "output_norm.weight":
                    tf = t_norm
                if tf is not None:
                    y = tf(n)
                else:
                    fn = os.path.join(CP_DIR, n.replace(".", "_") + ".bin")
                    u = np.frombuffer(open(fn, "rb").read(), dtype="<u2").astype(np.uint32)
                    y = (u << 16).view(np.float32)
                nel = int(np.prod(dims))
                assert len(y) == nel, (n, len(y), nel)
                if nt == T_Q8_0:
                    out.write(q8b(y))
                else:
                    out.write(y.astype("<f4").tobytes())
                n_cp += 1
            if (i + 1) % 40 == 0:
                print(f"  [{i+1}/733] {out.tell()/2**30:.1f} GiB ({time.time()-t0:.0f}s)", flush=True)
    cq1.close()
    print(f"DONE {OUT}: cp={n_cp} experts={n_exp} in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
