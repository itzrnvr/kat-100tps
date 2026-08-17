#!/usr/bin/env python
# KAT-CQ3 = KAT-CQ2 trunk (verbatim bytes) + pristine-BF16 MTP head from
# gbuzhf original-mtp-head.safetensors, requantized OUR way:
#   head matmuls -> Q8_0 (our validated quantizer), head routed experts ->
#   Q4_K RTN (build_rebase.quant_q4_k), norms/routers -> F32.
#
# Why: full provenance (no community middleman quant), our precision map,
# quality-gated lineage. Donor head provenance: Qwen3.6-35B-A3B trained MTP
# head (gbuzhf graft = same source; verified 76%/48%/73% acceptance on KAT).
#
# Tensor mapping (safetensors -> GGUF blk.40.*):
#   mtp.fc.weight                -> blk.40.nextn.eh_proj.weight        Q8_0
#     NOTE sf [2048,4096] = (out,in); GGUF eh_proj is (in=4096,out=2048) ->
#     transpose. Verified vs byteshape donor orientation.
#   mtp.pre_fc_norm_hidden.weight-> blk.40.nextn.hnorm.weight           F32
#   mtp.pre_fc_norm_embedding.w  -> blk.40.nextn.enorm.weight           F32
#   mtp.norm.weight              -> blk.40.nextn.shared_head_norm.weight F32
#   mtp.layers.0.input_layernorm -> blk.40.attn_norm.weight             F32
#   mtp.layers.0.post_attention_layernorm -> blk.40.post_attention_norm F32
#   self_attn.q_proj [8192,2048]  -> blk.40.attn_q.weight  (in,out)?? see NOTE
#   self_attn.k_proj [512,2048]   -> blk.40.attn_k.weight
#   self_attn.v_proj [512,2048]   -> blk.40.attn_v.weight
#   self_attn.o_proj [2048,4096]  -> blk.40.attn_output.weight
#   self_attn.q_norm/k_norm       -> attn_q_norm/attn_k_norm            F32
#   mlp.experts.gate_up_proj [256,1024,2048] fused -> split into
#     blk.40.ffn_gate_exps [2048,512,256] + blk.40.ffn_up_exps           Q4_K
#   mlp.experts.down_proj [256,2048,512] -> blk.40.ffn_down_exps         Q4_K
#   mlp.gate.weight [256,2048]    -> blk.40.ffn_gate_inp.weight  (needs transpose to [2048,256]) F32
#   mlp.shared_expert.{gate,up,down} -> ffn_{gate,up,down}_shexp         Q8_0
#   mlp.shared_expert_gate [1,2048]-> blk.40.ffn_gate_inp_shexp.weight   F32
#
# AXIS CONVENTION (verified against byteshape donor dims):
#   donor GGUF attn_q = (2048, 8192) = (in, out) per ggml ne convention
#   (ne[0]=row=in). safetensors q_proj = [8192, 2048] = [out, in] torch.
#   => GGUF tensor = torch.T (transpose).
#   eh_proj: donor GGUF (4096, 2048) = (in,out); sf fc [2048,4096]=[out,in]
#   => transpose as well.
#   fused gate_up [256,1024,2048]: per-expert [1024,2048]=[out,in] where
#   out 1024 = gate(512)+up(512) INTERLEAVED? or concat? Qwen3.6 MTP fuses
#   as [gate; up] concat along out axis per HF module impl. Split halves:
#   first 512 rows = gate, last 512 = up. Cross-checked: donor separate
#   gate/up exist -> if mismatch, perplexity gate catches it.
import json, struct, sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_gguf import SrcGGUF
import e0
import build_rebase as BR  # quant_q4_k (validated RTN)

SF   = "C:/merge/kat-mtp-head-bf16.safetensors"
BASE = "D:/merge/out/KAT-CQ2.gguf"
OUT  = "C:/merge/KAT-CQ3-MTP.gguf"
ALIGN = 32

def q8b(x):
    x = np.ascontiguousarray(x, dtype=np.float32).ravel()
    assert x.size % 32 == 0
    n = x.size // 32
    xb = x.reshape(n, 32)
    amax = np.abs(xb).max(axis=1)
    d = np.maximum(np.where(amax > 0, amax / 127.0, 1.0), 1.1754944e-38)
    d = np.where(np.isfinite(d), d, 1.0)
    q = np.rint(xb * (1.0 / d)[:, None]).astype(np.int8)
    blk = np.zeros((n, 34), dtype=np.uint8)
    blk[:, :2] = d.astype("<f2").view(np.uint8).reshape(n, 2)
    blk[:, 2:] = q.view(np.uint8).reshape(n, 32)
    return blk.tobytes()

def sf_read(hdr, f, name):
    info = hdr[name]
    dt, shp = info["dtype"], info["shape"]
    s0, s1 = info["data_offsets"]
    f.seek(8 + hdr_len + s0)
    n = int(np.prod(shp)) if shp else 1
    assert dt == "BF16", (name, dt)
    u = np.frombuffer(f.read(s1 - s0), dtype="<u2").astype(np.uint32)
    return (u << 16).view(np.float32).reshape(shp)

def main():
    global hdr_len
    t0 = time.time()
    # ---- read safetensors ----
    f = open(SF, "rb")
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))
    R = lambda n: sf_read(hdr, f, n)

    base = SrcGGUF(BASE)

    # ---- build MTP tensor list: (gguf_name, ggml_type, bytes, dims) ----
    mtp = []  # (name, dims, ggml_type, payload)
    def add(name, dims, gt, payload):
        mtp.append((name, dims, gt, payload))

    # norms / routers (F32)
    for sfn, ggn in [
        ("mtp.pre_fc_norm_hidden.weight", "blk.40.nextn.hnorm.weight"),
        ("mtp.pre_fc_norm_embedding.weight", "blk.40.nextn.enorm.weight"),
        ("mtp.norm.weight", "blk.40.nextn.shared_head_norm.weight"),
        ("mtp.layers.0.input_layernorm.weight", "blk.40.attn_norm.weight"),
        ("mtp.layers.0.post_attention_layernorm.weight", "blk.40.post_attention_norm.weight"),
        ("mtp.layers.0.self_attn.q_norm.weight", "blk.40.attn_q_norm.weight"),
        ("mtp.layers.0.self_attn.k_norm.weight", "blk.40.attn_k_norm.weight"),
        ("mlp_shared_expert_gate", "blk.40.ffn_gate_inp_shexp.weight"),
    ]:
        if sfn == "mlp_shared_expert_gate":
            x = R("mtp.layers.0.mlp.shared_expert_gate.weight").reshape(-1)
        else:
            x = R(sfn).reshape(-1)
        dims = [len(x)]
        add(ggn, dims, 0, x.astype("<f4").tobytes())

    # router: mlp.gate.weight [256,2048] -> GGUF ffn_gate_inp (2048,256) = T
    g = R("mtp.layers.0.mlp.gate.weight")          # [256(e), 2048(h)]
    gi = np.ascontiguousarray(g.T)                  # [2048, 256]
    add("blk.40.ffn_gate_inp.weight", [2048, 256], 0, gi.astype("<f4").tobytes())

    # fc / eh_proj: sf [2048,4096]=[out,in] -> GGUF (in,out) = T
    fc = R("mtp.fc.weight")                         # [2048, 4096]
    eh = np.ascontiguousarray(fc.T)                 # [4096, 2048]
    add("blk.40.nextn.eh_proj.weight", [4096, 2048], 8, q8b(eh))

    # attention matmuls: sf [out,in] -> GGUF (in,out)=T ; Q stays PACKED (Q||gate)
    for sfn, ggn in [
        ("mtp.layers.0.self_attn.q_proj.weight", "blk.40.attn_q.weight"),
        ("mtp.layers.0.self_attn.k_proj.weight", "blk.40.attn_k.weight"),
        ("mtp.layers.0.self_attn.v_proj.weight", "blk.40.attn_v.weight"),
        ("mtp.layers.0.self_attn.o_proj.weight", "blk.40.attn_output.weight"),
    ]:
        x = np.ascontiguousarray(R(sfn).T)
        add(ggn, list(x.shape), 8, q8b(x))

    # shared expert: sf gate [512,2048]=[out,in] -> GGUF ffn_gate_shexp (in=2048,out=512)=T
    for sfn, ggn in [
        ("mtp.layers.0.mlp.shared_expert.gate_proj.weight", "blk.40.ffn_gate_shexp.weight"),
        ("mtp.layers.0.mlp.shared_expert.up_proj.weight",   "blk.40.ffn_up_shexp.weight"),
    ]:
        x = np.ascontiguousarray(R(sfn).T)          # [2048, 512]
        add(ggn, [2048, 512], 8, q8b(x))
    # down: sf [2048,512]=[in? out?] HF down = [model, hidden] => [out=2048? no]
    # HF Linear down_proj: in=512(inter), out=2048(hidden): sf [2048,512]=[out,in]
    # GGUF ffn_down_shexp donor = (512, 2048) = (in, out) -> T
    x = np.ascontiguousarray(R("mtp.layers.0.mlp.shared_expert.down_proj.weight").T)
    add("blk.40.ffn_down_shexp.weight", [512, 2048], 8, q8b(x))

    # routed experts: fused gate_up [256,1024,2048] per-expert [out=1024,in=2048]
    # split concat halves: gate = rows 0:512 -> GGUF (in,out)=T ; up = 512:1024
    gu = R("mtp.layers.0.mlp.experts.gate_up_proj")   # [256, 1024, 2048]
    dn = R("mtp.layers.0.mlp.experts.down_proj")      # [256, 2048, 512] [in=2048,out=512]?
    E = gu.shape[0]
    gate_e = np.empty((E, 2048, 512), dtype=np.float32)  # (in,out) per exp
    up_e   = np.empty((E, 2048, 512), dtype=np.float32)
    for e in range(E):
        gate_e[e] = gu[e, :512, :].T                     # [2048,512]
        up_e[e]   = gu[e, 512:, :].T
    # donor GGUF ffn_gate_exps = (2048,512,256) = (in,out,expert)
    gate_t = np.ascontiguousarray(gate_e.transpose(1, 2, 0))  # (2048,512,256)
    up_t   = np.ascontiguousarray(up_e.transpose(1, 2, 0))
    add("blk.40.ffn_gate_exps.weight", [2048, 512, 256], 12, BR.quant_q4_k(gate_t.reshape(-1)))
    add("blk.40.ffn_up_exps.weight",   [2048, 512, 256], 12, BR.quant_q4_k(up_t.reshape(-1))
        if hasattr(BR.quant_q4_k, "__call__") else b"")
    # down: donor GGUF ffn_down_exps = (512,2048,256) = (out? in?)...
    # donor: (512, 2048, 256): ne0=512(inter-out), ne1=2048(hidden-in)? 
    # GGUF conv stores down as [inter, hidden, expert] with row=in=hidden? 
    # llama.cpp reads ffn_down_exps as ne=[n_ff, n_embd, n_exp]: matmul down(in=n_ff)->out=n_embd
    # torch down_proj [out=2048, in=512] per expert, sf [256, 2048, 512] = [e, out, in]
    # GGUF wants (in=512, out=2048, e) => per-expert transpose then permute
    down_t = np.ascontiguousarray(dn.transpose(2, 1, 0))  # (512,2048,256)
    add("blk.40.ffn_down_exps.weight", [512, 2048, 256], 12, BR.quant_q4_k(down_t.reshape(-1)))

    print(f"MTP tensors prepared: {len(mtp)}")

    # ---- assemble: base tensors verbatim + MTP appended ----
    entries = []  # (name, dims, type, src_file, src_abs_off, nbytes)
    cur = 0
    for n, (dims, tt, toff, tp, op) in base.entries.items():
        nb = e0.tensor_nbytes(dims, tt)
        entries.append((n, dims, tt, BASE, base.header_len + toff, nb))
        cur = (cur + nb + ALIGN - 1) // ALIGN * ALIGN
    for name, dims, gt, payload in mtp:
        nb = len(payload)
        entries.append((name, dims, gt, None, 0, nb))
        cur = (cur + nb + ALIGN - 1) // ALIGN * ALIGN
    total_data = cur
    print(f"data plan: {total_data/2**30:.2f} GiB ({len(entries)} tensors)")

    # ---- header: clone CQ2's original 54-KV section, patch counts, add 2 KVs ----
    hdr_bytes = bytearray(base.header[:10943779])  # through end of original KV section
    struct.pack_into("<Q", hdr_bytes, 8, len(entries))    # n_tensors
    struct.pack_into("<Q", hdr_bytes, 16, 55)             # n_kv = 54 + 1 (only nextn_predict_layers added)
    struct.pack_into("<I", hdr_bytes, 1000, 41)           # block_count 40 -> 41

    def kv_str(k, v):
        b = k.encode(); vb = v.encode()
        return struct.pack("<Q", len(b)) + b + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
    def kv_u32(k, v):
        b = k.encode()
        return struct.pack("<Q", len(b)) + b + struct.pack("<I", 4) + struct.pack("<I", v)
    hdr_bytes += kv_u32("qwen35moe.nextn_predict_layers", 1)

    # tensor table (RELATIVE offsets — llama.cpp convention, verified)
    def build_tbl():
        t = bytearray(); o = []; c = 0
        for (n, dims, gt, sf_, soff, nb) in entries:
            b = n.encode()
            t += struct.pack("<Q", len(b)) + b
            t += struct.pack("<I", len(dims))
            for d in dims: t += struct.pack("<Q", d)
            t += struct.pack("<I", gt)
            t += struct.pack("<Q", c)
            o.append(c)
            c = (c + nb + ALIGN - 1) // ALIGN * ALIGN
        return t, o
    data_start = 0
    for _ in range(3):
        tbl, offs = build_tbl()
        data_start = (10943779 + (len(hdr_bytes)-10943779) + len(tbl) + ALIGN - 1) // ALIGN * ALIGN
    header = bytes(hdr_bytes) + bytes(tbl)
    header += b"\x00" * (data_start - len(header))
    assert len(header) == data_start

    # ---- write ----
    bf = open(BASE, "rb")
    with open(OUT, "wb") as out:
        out.write(header)
        for i, (n, dims, gt, sf_, soff, nb) in enumerate(entries):
            out.seek(data_start + offs[i])
            if sf_ is BASE:
                bf.seek(soff)
                remaining = nb
                while remaining > 0:
                    chunk = bf.read(min(1 << 24, remaining))
                    if not chunk: raise RuntimeError(f"short {n}")
                    out.write(chunk); remaining -= len(chunk)
            else:
                payload = dict((m[0], m[3]) for m in mtp)[n]
                out.write(payload)
            if (i + 1) % 120 == 0:
                print(f"  [{i+1}/{len(entries)}] ({time.time()-t0:.0f}s)", flush=True)
    bf.close()
    print(f"DONE {OUT} in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
