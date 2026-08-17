#!/usr/bin/env python
# Build a lucebox DFlash-draft GGUF from the byteshape MTP block (blk.40),
# using the SHARED expert path only (dense, always-on) + block attention.
#
# RATIONALE: lucebox's draft loader is dense-FFN-only (ffn_gate/up/down
# required). The MTP block's MoE has 256 routed experts (can't merge cheaply
# without quality loss) but ONE shared expert (ffn_*_shexp) that fires on
# every token — a legitimate dense single-layer drafter. Attention comes
# from the block's full-attn tensors (blk.40 IS a full-attention layer:
# attn_q/k/v present, no ssm_*).
#
# SCHEMA (matches dflash-draft-3.6-q8_0 exactly):
#   metadata: general.architecture=qwen35-dflash-draft, embedding_length=2048
#     (our model's hidden), block_count=1, dflash.n_target_layers=40,
#     dflash.block_size=16, mask_token_id from donor, vocab 248320,
#     head_count=8 (2048/256), head_count_kv=2 (512/256), key_length=256
#   tensors:  dflash.fc.weight        <- nextn.eh_proj (4096=2*2048: [h;emb] proj)
#             dflash.hidden_norm      <- nextn.hnorm
#             output_norm.weight      <- nextn.shared_head_norm
#             blk.0.attn_q/k/v/output, norms  <- blk.40.attn_*
#             blk.0.ffn_gate/up/down  <- blk.40.ffn_gate/up/down_shexp
#             blk.0.ffn_norm.weight   <- blk.40.post_attention_norm
#   NOTE eh_proj input is [hidden(2048); embed(2048)] = 4096 -> 2048 out.
#        dflash.fc takes [hidden; embed] too (25600x5120 = 5*5120 = target
#        features...). Their fc maps target hidden-features -> draft hidden.
#        For single-layer: fc = eh_proj[2048-out, 4096-in] transposed to
#        their layout (out,in) = (2048, 4096). Loader checks n_embd=fc.out?
#        Their fc dims (25600,5120): 25600 = 5 layers * 5120 target feats.
#        => fc maps ALL target layer outputs -> draft input. Ours: 40 layers
#        * 2048 = 81920?? No — keep n_target_layers=1: fc = eh_proj (2048,4096).
import struct, sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_gguf import SrcGGUF
import e0

SRC = "D:/merge/out/bs-mtp-iq4xs.gguf"
OUT = "C:/merge/kat-mtp-shexp-draft.gguf"
ALIGN = 32

def deq(src, name):
    """Read tensor from donor GGUF, dequantize to f32 (BF16/F16/Q8_0/Q4_K...)."""
    dims, tt, toff, tp, op = src.entries[name]
    src.f.seek(src.header_len + toff)
    n = int(np.prod(dims))
    raw = src.f.read(e0.tensor_nbytes(dims, tt))
    if tt == 0:   return np.frombuffer(raw, "<f4").astype(np.float32).reshape(dims)
    if tt in (1, 30):  # F16 / BF16
        u16 = np.frombuffer(raw, "<u2")
        if tt == 1:
            return (u16.astype(np.uint32) << np.where(u16 & 0x8000, 16, 0)
                    if False else np.frombuffer(raw, "<f2").astype(np.float32)).reshape(dims)
        # BF16: upper 16 bits of f32
        return ((u16.astype(np.uint32) << 16).view(np.float32)).reshape(dims)
    if tt == 8:   # Q8_0
        blk = np.frombuffer(raw, np.uint8).reshape(-1, 34)
        d = blk[:, :2].copy().view(np.float16).ravel().astype(np.float32)
        q = blk[:, 2:].copy().view(np.int8)
        return (q.astype(np.float32) * d[:, None]).ravel().reshape(dims)
    if tt == 13:  # Q5_K (e0 validated dequantizer; returns f32 raveled)
        return e0.deq_q5k(raw).reshape(dims)
    raise ValueError(f"unhandled donor type {tt} for {name}")

def q8(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.size // 32
    xb = x.reshape(n, 32)
    amax = np.abs(xb).max(axis=1)
    d = np.maximum(np.where(amax > 0, amax / 127.0, 1.0), 1.1754944e-38)
    q = np.rint(xb * (1.0 / d)[:, None]).astype(np.int8)
    blk = np.zeros((n, 34), dtype=np.uint8)
    blk[:, :2] = d.astype("<f2").view(np.uint8).reshape(n, 2)
    blk[:, 2:] = q.view(np.uint8).reshape(n, 32)
    return blk.tobytes()

def f32(x):
    return np.ascontiguousarray(x, dtype=np.float32).tobytes()

def main():
    t0 = time.time()
    src = SrcGGUF(SRC)
    B = "blk.40."

    # draft tensor plan: (gguf_name, donor_name, layout_fn, quant)
    plan = [
        # dflash.fc: lucebox multiplies fc [out, target_n_embd] x target_feats[2048].
        # Donor eh_proj is (2048 out, 4096 in = [hidden; embed] concat) — take the
        # hidden-half columns so the matmul shape matches (2048, 2048).
        # Implemented via special-case below (donor sliced, not copied whole).
        ("dflash.fc.weight",            None,                             None, q8, 8),
        ("dflash.hidden_norm.weight",   B + "nextn.hnorm.weight",        None, f32, 0),
        ("output_norm.weight",          B + "nextn.shared_head_norm.weight", None, f32, 0),
        ("blk.0.attn_q.weight",         B + "attn_q.weight",             None, q8, 8),
        ("blk.0.attn_q_norm.weight",    B + "attn_q_norm.weight",        None, f32, 0),
        ("blk.0.attn_k.weight",         B + "attn_k.weight",             None, q8, 8),
        ("blk.0.attn_k_norm.weight",    B + "attn_k_norm.weight",        None, f32, 0),
        ("blk.0.attn_v.weight",         B + "attn_v.weight",             None, q8, 8),
        ("blk.0.attn_output.weight",    B + "attn_output.weight",        None, q8, 8),
        ("blk.0.attn_norm.weight",      B + "attn_norm.weight",          None, f32, 0),
        ("blk.0.ffn_norm.weight",       B + "post_attention_norm.weight",None, f32, 0),
        ("blk.0.ffn_gate.weight",       B + "ffn_gate_shexp.weight",     None, q8, 8),
        ("blk.0.ffn_up.weight",         B + "ffn_up_shexp.weight",       None, q8, 8),
        ("blk.0.ffn_down.weight",       B + "ffn_down_shexp.weight",     None, q8, 8),
    ]

    blobs = []
    for gguf_name, donor, layout, quant, gt in plan:
        if gguf_name == "dflash.fc.weight":
            eh = deq(src, B + "nextn.eh_proj.weight")   # (2048, 4096)
            x = eh[:2048, :].copy()                     # input-hidden half (axis0=in per ggml ne convention)
        elif gguf_name == "blk.0.attn_q.weight":
            # Q||gate is PER-HEAD INTERLEAVED (qwen35_target_graph.cpp:684):
            # out axis = n_head * [q(256); gate(256)]. Take q of every head:
            # reshape out-axis to (16, 2, 256), keep [:, 0, :].
            qg = deq(src, B + "attn_q.weight")          # (2048, 8192)
            n_h, hd = 16, 256
            qg = qg.reshape(2048, n_h, 2, hd)           # in, head, q|gate, dim
            x = np.ascontiguousarray(qg[:, :, 0, :])    # (2048, 16, 256)
            x = x.reshape(2048, n_h * hd)               # (2048, 4096) Q-only
        else:
            x = deq(src, donor)
        blobs.append((gguf_name, x.shape, quant(x)))
        print(f"  {gguf_name}: {x.shape} from {donor}")

    # metadata
    meta_pairs = [
        ("general.architecture",                (8, b"qwen35-dflash-draft")),
        ("general.name",                        (8, b"KAT-MTP-shexp-draft")),
        ("general.quantization_version",        (4, struct.pack("<I", 2))),
        ("qwen35-dflash-draft.block_count",     (4, struct.pack("<I", 1))),
        ("qwen35-dflash-draft.embedding_length",(4, struct.pack("<I", 2048))),
        ("qwen35-dflash-draft.feed_forward_length", (4, struct.pack("<I", 512))),
        ("qwen35-dflash-draft.attention.head_count", (4, struct.pack("<I", 16))),  # 16x256=4096 = true q_dim (donor q is Q||gate packed)
        ("qwen35-dflash-draft.attention.head_count_kv", (4, struct.pack("<I", 2))),
        ("qwen35-dflash-draft.attention.key_length", (4, struct.pack("<I", 256))),
        ("qwen35-dflash-draft.attention.value_length", (4, struct.pack("<I", 256))),
        ("qwen35-dflash-draft.attention.layer_norm_rms_epsilon", (6, struct.pack("<f", 1e-6))),
        ("qwen35-dflash-draft.context_length",  (4, struct.pack("<I", 32768))),
        ("qwen35-dflash-draft.vocab_size",      (4, struct.pack("<I", 248320))),
        ("qwen35-dflash-draft.rope.freq_base",  (6, struct.pack("<f", 1000000.0))),  # type 6 F32 4B (reference uses 6; F64 breaks get_val<float>)
        ("qwen35-dflash-draft.dflash.block_size", (4, struct.pack("<I", 16))),
        ("qwen35-dflash-draft.dflash.mask_token_id", (4, struct.pack("<I", 248070))),
        ("qwen35-dflash-draft.dflash.n_target_layers", (4, struct.pack("<I", 40))),
    ]

    # NOTE on fc dims: donor eh_proj is (2048 out, 4096 in [h;e]). The
    # reference draft's fc is (25600, 5120) = (n_tgt_layers*target_hidden, ?).
    # Our n_target_layers=40 * 2048 = 81920 != 2048 rows. The loader derives
    # n_tgt_layers from fc.out / n_embd when unset; we WANT 1-layer semantics,
    # so set embedding_length=2048 and fc rows=2048 -> derived n_tgt=1.
    # n_target_layers metadata left consistent = 1:
    for i,(k,(t,v)) in enumerate(meta_pairs):
        if k.endswith("dflash.n_target_layers"):
            meta_pairs[i] = (k, (4, struct.pack("<I", 1)))

    # header
    def kv_bytes():
        out = bytearray()
        for k, (t, v) in meta_pairs:
            kb = k.encode()
            out += struct.pack("<Q", len(kb)) + kb
            out += struct.pack("<I", t)
            if t == 8:  # STRING values carry their own 8-byte length prefix
                out += struct.pack("<Q", len(v)) + v
            else:
                out += v
        return out
    kvb = kv_bytes()

    tbl = bytearray()
    offs, cur = [], 0
    for (gguf_name, dims, blob), (_, _, _, _, gt) in zip(blobs, plan):
        nb = len(blob)
        nb_al = (nb + ALIGN - 1) // ALIGN * ALIGN
        b = gguf_name.encode()
        tbl += struct.pack("<Q", len(b)) + b
        tbl += struct.pack("<I", len(dims))
        for d in dims: tbl += struct.pack("<Q", d)
        tbl += struct.pack("<I", gt)
        tbl += struct.pack("<Q", cur)
        offs.append((cur, nb))
        cur += nb_al
    data_start = (24 + len(kvb) + len(tbl) + ALIGN - 1) // ALIGN * ALIGN

    with open(OUT, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(blobs)))   # n_tensors
        f.write(struct.pack("<Q", len(meta_pairs)))  # n_kv
        f.write(kvb)
        f.write(tbl)
        f.write(b"\x00" * (data_start - f.tell()))
        for (o, nb), (_, _, blob) in zip(offs, blobs):
            f.seek(data_start + o)
            f.write(blob)
            f.write(b"\x00" * (((nb + ALIGN - 1)//ALIGN*ALIGN) - nb))
    print(f"DONE {OUT} in {(time.time()-t0):.1f}s")

if __name__ == "__main__":
    main()
