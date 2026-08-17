# Local GGUF inventory parser — corrected GGML metadata value-type map.
# BUG FIXED [2026-08-17]: earlier version used wrong value-type ids
# (8=STRING not UINT64, 9=ARRAY not INT64, 10/11/12=U64/I64/F64).
# That desync caused "general.architecture = 9" nonsense + MemoryError.
# PURPOSE: verify KAT-CQ1 tensor names/dtypes vs lucebox qwen35moe loader.
import struct, sys
from collections import Counter, defaultdict

GGML_TENSOR_TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",
                     8:"Q8_0",9:"Q8_1",10:"IQ2_XXS",14:"Q2_K",15:"Q3_K",
                     16:"Q4_K",17:"Q5_K",18:"Q6_K",19:"IQ2_XS",30:"BF16"}

def read_gguf(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, = struct.unpack("<I", f.read(4))
        n_tensors, n_meta = struct.unpack("<QQ", f.read(16))
        def rstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")
        def rval(t):
            if t == 8: return rstr()                    # STRING
            if t == 9:                                   # ARRAY
                sub, = struct.unpack("<I", f.read(4))
                cnt, = struct.unpack("<Q", f.read(8))
                return [rval(sub) for _ in range(cnt)]
            fmt = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",
                   7:"<B",10:"<Q",11:"<q",12:"<d"}[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        meta = {}
        for _ in range(n_meta):
            k = rstr(); t, = struct.unpack("<I", f.read(4))
            meta[k] = rval(t)
        tensors = []
        for _ in range(n_tensors):
            name = rstr()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8*nd))
            ttype, = struct.unpack("<I", f.read(4))
            roff, = struct.unpack("<Q", f.read(8))
            tensors.append((name, dims, ttype, roff))
        return ver, meta, tensors, f.tell()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "D:/merge/out/KAT-CQ1.gguf"
    ver, meta, tensors, ds = read_gguf(path)
    print(f"GGUF v{ver} | {len(tensors)} tensors | data @{ds}")
    for k in ("general.architecture","general.name","general.basename",
              "qwen35moe.expert_count","qwen35moe.expert_used_count",
              "qwen35moe.block_count","qwen35moe.embedding_length",
              "qwen35moe.context_length"):
        if k in meta: print(f"  {k} = {meta[k]}")
    tc = Counter(GGML_TENSOR_TYPES.get(t,str(t)) for _,_,t,_ in tensors)
    print("\ntensor type counts:", dict(tc))
    # control plane (non-expert) dtype map by role
    cp = defaultdict(lambda: Counter())
    for name, dims, t, off in tensors:
        if "ffn_src" in name: continue  # routed experts
        parts = name.split(".")
        role = parts[1] if len(parts) > 1 else name
        cp[role][GGML_TENSOR_TYPES.get(t,str(t))] += 1
    print("\ncontrol-plane roles x dtype:")
    for role in sorted(cp):
        print(f"  {role:28s} {dict(cp[role])}")
    # sample a few attn_q / ssm tensor shapes for loader-layout match
    print("\nlayout probe (first attn_q, ssm_conv1d, ffn_gate_shexp if any):")
    seen = set()
    for name, dims, t, off in tensors:
        for probe in ("attn_q","ssm_conv1d","ffn_gate_shexp","ffn_down_shexp"):
            if probe in name and probe not in seen:
                print(f"  {name}  {GGML_TENSOR_TYPES.get(t,t)}  dims={dims}")
                seen.add(probe)
        if len(seen) == 4: break
