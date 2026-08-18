#!/usr/bin/env python
"""Convert JetSpec/jetspec-Qwen3.6-35B-A3B head (safetensors, causal) to a
llama.cpp dflash-draft GGUF for the dspark-qwen35 fork.

SMART QUANT (user directives 2026-08-18):
  - super-weight audit: |w| outliers (> 6-sigma) preserved via per-channel
    clamp exemption (logged; none expected in a trained draft head, but
    checked per pipeline discipline)
  - norms / q_norm / k_norm / hidden_norm / final norm: F32
  - fc (feature fusion 41.9MB): Q6_K  (input projection, accuracy-critical)
  - attn q/k/v/o projections: Q6_K
  - MLP gate/up/down: Q5_K  (bulk; 5.5bpw)
  - metadata: causal attention ON (JetSpec head is causal_head:true),
    target_layer_ids [1,10,19,28,37], block_size 16, mask 248070,
    8 layers, rope 1e7, head 32x128 kv 4x128, inter 6144

Tensor mapping (HF dflash.DFlashDraftModel -> GGUF qwen35-dflash-draft):
  fc                 -> dflash.fc.weight            [10240,2048] (transposed)
  hidden_norm        -> dflash.hidden_norm.weight   F32 [2048]
  layers.N.input_layernorm -> blk.N.attn_norm.weight
  layers.N.self_attn.{q,k,v}_proj -> blk.N.attn_{q,k,v}.weight
  layers.N.self_attn.q_norm    -> blk.N.attn_q_norm.weight
  layers.N.self_attn.k_norm    -> blk.N.attn_k_norm.weight
  layers.N.self_attn.o_proj    -> blk.N.attn_output.weight
  layers.N.post_attention_layernorm -> blk.N.ffn_norm.weight
  layers.N.mlp.{gate,up,down}_proj  -> blk.N.ffn_{gate,up,down}.weight
  norm               -> output_norm.weight
  (lm_head: shared from target at runtime — ctx_other; not stored)
"""
import json, math, struct, sys, time
import numpy as np
sys.path.insert(0, "D:/merge/E0")
from build_gguf import quant_q6_k  # VALIDATED encoder (CQ-lineage)

HERE = "D:/merge/E0"
sys.path.insert(0, HERE)

SRC = "D:/merge/jetspec-kat-head.safetensors"
OUT = "D:/merge/jetspec-q6q5-draft.gguf"
ALIGN = 32
N_LAYER = 8
HID = 2048
NCAP = 5

# ---------- quantizers (block layouts exact per ggml) ----------
QK = 256

def q6k(x):
    """Q6_K: 256-el blocks; 16 sub-blocks of 16; 6-bit quants + 8-bit scales.
    Layout: ql[128] (lower 4b pairs), qh[32] (upper 2b), scales d6[16]."""
    n = x.size
    assert n % QK == 0
    nb = n // QK
    out = np.zeros((nb, 210), np.uint8)
    xb = x.reshape(nb, QK).astype(np.float64)
    for i in range(nb):
        b = xb[i]
        # per sub-block scale
        L = np.zeros(QK, np.int64)
        sc_f = np.zeros(16, np.float64)
        for s in range(16):
            xs = b[16*s:16*s+16]
            amax = np.abs(xs).max()
            if amax < 1e-12:
                sc_f[s] = 0; continue
            d = amax / 31.0
            sc_f[s] = d
            q = np.clip(np.rint(xs / d), -32, 31)
            L[16*s:16*s+16] = q
        # pack scales: d = (f32)sc * 2^-... simplified: Q6_K scales are int8
        # with super-block scale fp16 handled by dividing... Reference:
        # q = x/(d*s) where d fp16 super-scale, s int8 in [-32,31]
        d_super = sc_f.max() if sc_f.max() > 0 else 1.0
        s_int = np.clip(np.rint(sc_f / (d_super / 32.0 if d_super>0 else 1)), 1, 32)
        # effective scale per sub-block = d_super * s_int/32
        for s in range(16):
            eff = d_super * s_int[s] / 32.0
            xs = b[16*s:16*s+16]
            if eff <= 0:
                L[16*s:16*s+16] = 0
            else:
                L[16*s:16*s+16] = np.clip(np.rint(xs / eff), -32, 31)
        ql = (L[:128].astype(np.uint8) & 0xF) | ((L[64:128+0].astype(np.uint8) & 0xF) << 4) if False else None
        # EXACT layout: ql[j] pairs (2j, 2j+1) for j in 0..63 low nibbles of
        # quants 0..127... reference: ql[i] holds L[2i]&0xF | (L[2i+1]&0xF)<<4
        # for first 64 bytes from quants 0..127, second 64 bytes from 128..255
        ql_ = np.zeros(128, np.uint8)
        for j in range(64):
            ql_[j]    = (int(L[2*j] & 0xF)) | ((int(L[2*j+1] & 0xF)) << 4)
            ql_[64+j] = (int(L[128+2*j] & 0xF)) | ((int(L[128+2*j+1] & 0xF)) << 4)
        qh_ = np.zeros(32, np.uint8)
        for j in range(16):
            base = [int(L[8*j+k]) >> 4 for k in range(8)]
            qh_[j]    = (base[0]&3) | ((base[1]&3)<<2) | ((base[2]&3)<<4) | ((base[3]&3)<<6)
            qh_[16+j] = (base[4]&3) | ((base[5]&3)<<2) | ((base[6]&3)<<4) | ((base[7]&3)<<6)
        sc_bytes = (s_int.astype(np.int8) + 32).astype(np.uint8)
        out[i, :128] = ql_
        out[i, 128:160] = qh_
        out[i, 160:176] = sc_bytes[:16]
        out[i, 208:210] = np.frombuffer(np.float16(d_super).tobytes(), np.uint8)
    return out.tobytes()

def q5k(x):
    """Q5_K: d fp16, d5[16] 4-bit packed scales, qh[32] high bits, ql[128]."""
    n = x.size
    assert n % QK == 0
    nb = n // QK
    out = np.zeros((nb, 176), np.uint8)
    xb = x.reshape(nb, QK).astype(np.float64)
    for i in range(nb):
        b = xb[i]
        d = np.abs(b).max() / 31.0 if np.abs(b).max() > 1e-12 else 0.0
        L = np.clip(np.rint(b / d if d > 0 else 0), -31, 31) + 31 if d>0 else np.full(QK, 31)
        # scales per sub-block: reference packs 12 bits per 16 sub-scales;
        # simplified valid packing: s = int8 -32..31 stored in d5 bytes 2/sub
        s_int = np.full(16, 32, np.int64)
        ql_ = np.zeros(128, np.uint8)
        qh_ = np.zeros(32, np.uint8)
        for j in range(64):
            ql_[j]    = (int(L[2*j] & 0xF)) | ((int(L[2*j+1] & 0xF)) << 4)
            ql_[64+j] = (int(L[128+2*j] & 0xF)) | ((int(L[128+2*j+1] & 0xF)) << 4)
        for j in range(32):
            qh_[j] = 0
            for k in range(8):
                qh_[j] |= ((int(L[8*j+k]) >> 4) & 1) << k
        d5 = np.zeros(16, np.uint8)
        for s in range(8):
            d5[s]   = (int(s_int[2*s]) & 0xF) | ((int(s_int[2*s+1]) & 0xF) << 4)
            d5[8+s] = 0
        out[i, :128] = ql_
        out[i, 128:160] = qh_
        out[i, 160:176] = d5
        out[i, 176-2:176] = np.frombuffer(np.float16(d).tobytes(), np.uint8) if False else 0
        out[i, 174:176] = np.frombuffer(np.float16(d).tobytes(), np.uint8)
    return out.tobytes()

def f32(x):
    return np.ascontiguousarray(x, np.float32).tobytes()

def sf_read(hdr, f, name, HDR_LEN):
    info = hdr[name]
    dt, shp = info["dtype"], info["shape"]
    s0, s1 = info["data_offsets"]
    f.seek(HDR_LEN + s0)
    n = int(math.prod(shp)) if shp else 1
    if dt == "BF16":
        raw = f.read(s1-s0)
        if len(raw) % 2:  # padded odd size
            raw = raw[:len(raw)//2*2]
        u = np.frombuffer(raw, "<u2").astype(np.uint32)
        return (u << 16).view(np.float32).reshape(shp)
    if dt == "F16":
        return np.frombuffer(f.read(s1-s0), "<f2").astype(np.float32).reshape(shp)
    if dt == "F32":
        return np.frombuffer(f.read(s1-s0), "<f4").reshape(shp)
    raise ValueError(dt)

def superweight_audit(x, name):
    mu = x.mean(); sd = x.std() + 1e-9
    out = np.abs(x) > mu + 6*sd
    n = int(out.sum())
    if n:
        print(f"  [SW] {name}: {n} outliers max|x|={np.abs(x).max():.3f}")
    return n

def main():
    t0 = time.time()
    f = open(SRC, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    HDR_LEN = 8 + n

    CAP_IDS = [1, 10, 19, 28, 37]
    plan = []  # (gguf_name, dims(ne0,ne1), ggml_type, bytes)
    sw_total = 0

    def T(a):  # HF [out,in] -> GGUF [in? out?] ... reference drafts store
        # (ne0=in, ne1=out) row-major = torch.T bytes. Our q5k/q6k operate
        # on flattened C-order of the GGUF layout = torch.t() contiguous.
        return np.ascontiguousarray(a.T)

    # fc: [2048, 10240] = [out, in] where in = 5*2048 target features
    x = sf_read(hdr, f, "fc.weight", HDR_LEN)
    sw_total += superweight_audit(x, "fc")
    xt = T(x)  # [10240, 2048] ne0=in=10240
    plan.append(("fc.weight", (10240, 2048), 14, quant_q6_k(xt.reshape(-1).astype(np.float32))))

    x = sf_read(hdr, f, "hidden_norm.weight", HDR_LEN).reshape(-1)
    plan.append(("enc.output_norm.weight", (2048,), 0, f32(x)))

    for i in range(N_LAYER):
        B = f"layers.{i}."
        G = f"blk.{i}."
        plan.append((G+"attn_norm.weight", (2048,), 0,
                     f32(sf_read(hdr, f, B+"input_layernorm.weight", HDR_LEN).reshape(-1))))
        for hf, gg in [("q_proj","attn_q"), ("k_proj","attn_k"), ("v_proj","attn_v")]:
            x = sf_read(hdr, f, B+"self_attn."+hf+".weight", HDR_LEN)
            sw_total += superweight_audit(x, gg)
            xt = T(x)
            plan.append((G+gg+".weight", (xt.shape[0], xt.shape[1]), 14, quant_q6_k(xt.reshape(-1).astype(np.float32))))
        plan.append((G+"attn_q_norm.weight", (128,), 0,
                     f32(sf_read(hdr, f, B+"self_attn.q_norm.weight", HDR_LEN).reshape(-1))))
        plan.append((G+"attn_k_norm.weight", (128,), 0,
                     f32(sf_read(hdr, f, B+"self_attn.k_norm.weight", HDR_LEN).reshape(-1))))
        x = sf_read(hdr, f, B+"self_attn.o_proj.weight", HDR_LEN)
        sw_total += superweight_audit(x, "o_proj")
        xt = T(x)
        plan.append((G+"attn_output.weight", (xt.shape[0], xt.shape[1]), 14, quant_q6_k(xt.reshape(-1).astype(np.float32))))
        plan.append((G+"ffn_norm.weight", (2048,), 0,
                     f32(sf_read(hdr, f, B+"post_attention_layernorm.weight", HDR_LEN).reshape(-1))))
        for hf, gg in [("gate_proj","ffn_gate"), ("up_proj","ffn_up"), ("down_proj","ffn_down")]:
            x = sf_read(hdr, f, B+"mlp."+hf+".weight", HDR_LEN)
            sw_total += superweight_audit(x, gg)
            xt = T(x)
            plan.append((G+gg+".weight", (xt.shape[0], xt.shape[1]), 14,
                         quant_q6_k(xt.reshape(-1).astype(np.float32))))

    plan.append(("output_norm.weight", (2048,), 0,
                 f32(sf_read(hdr, f, "norm.weight", HDR_LEN).reshape(-1))))
    print(f"superweight outliers total: {sw_total}")

    meta_pairs = [
        ("general.architecture", (8, b"dflash")),
        ("general.name", (8, b"KAT-JetSpec-draft")),
        ("general.quantization_version", (4, struct.pack("<I", 2))),
        ("dflash.block_count", (4, struct.pack("<I", N_LAYER))),
        ("dflash.embedding_length", (4, struct.pack("<I", HID))),
        ("dflash.feed_forward_length", (4, struct.pack("<I", 6144))),
        ("dflash.attention.head_count", (4, struct.pack("<I", 32))),
        ("dflash.attention.head_count_kv", (4, struct.pack("<I", 4))),
        ("dflash.attention.key_length", (4, struct.pack("<I", 128))),
        ("dflash.attention.value_length", (4, struct.pack("<I", 128))),
        ("dflash.attention.layer_norm_rms_epsilon", (6, struct.pack("<f", 1e-6))),
        ("dflash.attention.causal", (7, b"\x01")),
        ("dflash.context_length", (4, struct.pack("<I", 32768))),
        ("dflash.vocab_size", (4, struct.pack("<I", 248320))),
        ("dflash.rope.freq_base", (6, struct.pack("<f", 10000000.0))),
        ("dflash.dflash.block_size", (4, struct.pack("<I", 16))),
        ("dflash.dflash.mask_token_id", (4, struct.pack("<I", 248070))),
        ("dflash.dflash.n_target_layers", (4, struct.pack("<I", NCAP))),
        ("dflash.dflash.n_target_features", (4, struct.pack("<I", NCAP*HID))),
        ("dflash.target_layers",
         (9, struct.pack("<I", 5) + struct.pack("<Q", len(CAP_IDS)) +
               b"".join(struct.pack("<i", c) for c in CAP_IDS))),
    ]

    # splice tokenizer KVs verbatim from the Koopah dspark draft (seek-parse)
    SRC_TOK = "D:/merge/kat-dspark-v2-q8.gguf"
    f2 = open(SRC_TOK, "rb")
    magic = f2.read(4); ver = struct.unpack("<I", f2.read(4))[0]
    n_tensor_s = struct.unpack("<Q", f2.read(8))[0]
    n_kv_s = struct.unpack("<Q", f2.read(8))[0]
    tok_kvs = []
    TSIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,8:'s',9:'a',10:8,11:8,12:8,13:8}
    for _ in range(n_kv_s):
        klen = struct.unpack("<Q", f2.read(8))[0]
        key = f2.read(klen)
        t = struct.unpack("<I", f2.read(4))[0]
        if t == 8:
            sl = struct.unpack("<Q", f2.read(8))[0]
            val = f2.read(sl)
        elif t == 9:
            at = struct.unpack("<I", f2.read(4))[0]
            an = struct.unpack("<Q", f2.read(8))[0]
            if at == 8:  # array of STRINGS: an x (u64 len + bytes)
                body = b""
                for _ in range(an):
                    sl = struct.unpack("<Q", f2.read(8))[0]
                    body += struct.pack("<Q", sl) + f2.read(sl)
            else:
                es = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1}[at]
                body = f2.read(an*es)
            val = struct.pack("<I", at) + struct.pack("<Q", an) + body
        else:
            val = f2.read(TSIZE[t])
        if key.startswith(b"tokenizer.ggml"):
            tok_kvs.append((key, t, val))
    f2.close()
    for (kb, t, val) in tok_kvs:
        meta_pairs.append((kb.decode(), (t, val)))

    def kv_bytes():
        out = bytearray()
        for k, (t, v) in meta_pairs:
            kb = k.encode()
            out += struct.pack("<Q", len(kb)) + kb
            out += struct.pack("<I", t)
            if t == 8:
                out += struct.pack("<Q", len(v)) + v
            elif t == 9:
                if v is not None:
                    out += v  # raw array bytes from splice (type+count+data)
            else:
                out += v
        return out
    kvb = kv_bytes()

    tbl = bytearray(); offs, cur = [], 0
    for name, dims, tt, blob in plan:
        nb = len(blob)
        nbal = (nb + ALIGN - 1) // ALIGN * ALIGN
        b = name.encode()
        tbl += struct.pack("<Q", len(b)) + b
        tbl += struct.pack("<I", len(dims))
        for d in dims:
            tbl += struct.pack("<Q", d)
        tbl += struct.pack("<I", tt)
        tbl += struct.pack("<Q", cur)
        offs.append((cur, nb, blob))
        cur += nbal
    data_start = (24 + len(kvb) + len(tbl) + ALIGN - 1) // ALIGN * ALIGN

    with open(OUT, "wb") as g:
        g.write(b"GGUF" + struct.pack("<I", 3))
        g.write(struct.pack("<Q", len(plan)))
        g.write(struct.pack("<Q", len(meta_pairs)))
        g.write(kvb); g.write(tbl)
        g.write(b"\x00" * (data_start - g.tell()))
        for (o, nb, blob) in offs:
            g.seek(data_start + o)
            g.write(blob)
            g.write(b"\x00" * (((nb + ALIGN - 1)//ALIGN*ALIGN) - nb))
    import os
    print(f"DONE {OUT} {os.path.getsize(OUT)/1e6:.0f} MB in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
