# PURPOSE: Side-by-side arch/bytes comparison of KAT-CQ3-MTP.gguf vs
# Ornith15-Q4KM.gguf. User correction (V112b): both are hybrid gated-DeltaNet
# + full-attention qwen3_5_moe — the "DeltaNet CPU cost" hypothesis is WRONG.
# Find the real delta: ssm shapes, attention shapes, byte totals per class.
import struct, collections

def parse(path):
    f = open(path, "rb")
    buf = f.read(96 << 20)
    p = 24
    def rstr(b, o):
        n, = struct.unpack_from("<Q", b, o)
        return b[o+8:o+8+n].decode(), o+8+n
    def skip(b, o, t):
        m = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8,13:2}
        if t == 8:
            n, = struct.unpack_from("<Q", b, o); return o+8+n
        if t == 9:
            at, = struct.unpack_from("<I", b, o); n, = struct.unpack_from("<Q", b, o+4); o += 12
            for _ in range(n): o = skip(b, o, at)
            return o
        return o + m[t]
    kvs = {}
    n_tensors, = struct.unpack_from("<Q", buf, 8)
    n_kv, = struct.unpack_from("<Q", buf, 16)
    for _ in range(n_kv):
        k, p = rstr(buf, p)
        t, = struct.unpack_from("<I", buf, p); p += 4
        if t == 8:
            v, p = rstr(buf, p)
        elif t == 7:
            v = bool(struct.unpack_from("<B", buf, p)[0]); p += 1
        elif t in (4,):
            v, = struct.unpack_from("<I", buf, p); p += 4
        elif t in (10,):
            v, = struct.unpack_from("<Q", buf, p); p += 8
        elif t == 9:
            at, = struct.unpack_from("<I", buf, p); n, = struct.unpack_from("<Q", buf, p+4)
            v = f"array[{at}]x{n}"; p = skip(buf, p, t); continue
        else:
            p = skip(buf, p, t); v = f"<t{t}>"
        kvs[k] = v
    tensors = []
    for _ in range(n_tensors):
        name, p = rstr(buf, p)
        nd, = struct.unpack_from("<I", buf, p); p += 4
        dims = struct.unpack_from(f"<{nd}Q", buf, p); p += 8*nd
        t, = struct.unpack_from("<I", buf, p); p += 4
        off, = struct.unpack_from("<Q", buf, p); p += 8
        tensors.append((name, dims, t))
    # type -> bytes/elem for GGUF quant types
    TB = {0:4,1:2,8:1,12:(18+2)/32*4,13:(2+2)/32*4,14:(2+2)/32*2,15:1/2,16:1,24:(4+2)/32*4,28:(4+12)/64*4,29:1}
    def nbytes(dims, t):
        n = 1
        for d in dims: n *= d
        bpe = {0:4,1:2,14:1.0625,12:0.5625,13:0.5,24:0.5625,28:0.5625}.get(t, None)
        if bpe is None: return None
        return int(n * bpe)
    return kvs, tensors, nbytes

for label, path in [("KAT-CQ3", r"C:\merge\KAT-CQ3-MTP.gguf"),
                    ("Ornith-Q4KM", r"D:\merge\out\Ornith15-Q4KM.gguf")]:
    kvs, tensors, nbytes = parse(path)
    print(f"\n===== {label} =====")
    for k in sorted(kvs):
        if k.startswith("qwen3_5_moe.") and "tokenizer" not in k:
            print(f"  {k} = {kvs[k]}")
    # bytes by class
    cls = collections.Counter(); shapes = {}
    for name, dims, t in tensors:
        nb = nbytes(dims, t)
        if nb is None: continue
        parts = name.split(".")
        if "blk" in parts:
            suf = ".".join(parts[2:])
            # classify
            if "ffn_" in suf and "exps" in suf: c = "routed-experts"
            elif "shexp" in suf: c = "shared-experts"
            elif "ffn_gate_inp" in suf: c = "routers"
            elif suf.startswith("attn_q") or suf.startswith("attn_k") or suf.startswith("attn_v") or suf.startswith("attn_o") or suf.startswith("attn_gate"):
                c = "attention-proj"
            elif "ssm" in suf: c = "ssm"
            elif "norm" in suf: c = "norms"
            elif "nextn" in suf: c = "mtp-head"
            else: c = "other"
            key = c
        elif name in ("token_embd.weight", "output.weight"): key = "embeddings"
        elif name == "output_norm.weight": key = "norms"
        else: key = "other"
        cls[key] += nb
        if key in ("ssm", "attention-proj") and suf not in shapes and "blk.0." in name or (key=="ssm" and name.startswith("blk.0.")):
            shapes[name] = (dims, t)
    tot = sum(cls.values())
    for c, b in cls.most_common():
        print(f"  {c:16s} {b/1e9:7.3f} GB  ({100*b/tot:.1f}%)")
    print(f"  TOTAL           {tot/1e9:7.3f} GB")
    print("  blk.0 ssm/shape samples:")
    for name, dims, t in tensors:
        if name.startswith("blk.0.") and ("ssm" in name or "attn" in name):
            print(f"    {name:40s} {dims} t{t}")
