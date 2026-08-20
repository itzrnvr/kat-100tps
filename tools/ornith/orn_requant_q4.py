# PURPOSE: In-place requant experiment — convert the 21 Q6_K ffn_down_exps
# tensors in the official Ornith Q4_K_M to Q4_K, writing smaller blobs at the
# same offsets (dead hole left behind, offsets untouched), patching the type
# field 14 (Q6_K) -> 12 (Q4_K). Isolates the quant-type variable: predicted
# ~1.08x traffic cut on expert reads (V112b math).
# Import names use Unicode subscripts; resolved via dir().
import os, struct, sys, importlib.util
import numpy as np

PATH = r"D:\merge\out\Ornith15-Q4KM.gguf"
E0 = r"D:\merge\E0"
sys.path.insert(0, E0)

import e0
spec = importlib.util.spec_from_file_location("br", os.path.join(E0, "build_rebase.py"))
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)
q4k_fn = next(getattr(br, n) for n in dir(br) if n.startswith("quant_q4"))
print("quantizer:", q4k_fn.__name__)

ALIGN = 32
f = open(PATH, "r+b")
buf = f.read(96 << 20)
n_tensors, = struct.unpack_from("<Q", buf, 8)
n_kv, = struct.unpack_from("<Q", buf, 16)
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

for _ in range(n_kv):
    _, p = rstr(buf, p); t, = struct.unpack_from("<I", buf, p); p += 4
    if t == 8: _, p = rstr(buf, p); continue
    p = skip(buf, p, t)

tensors = []
for _ in range(n_tensors):
    name, p = rstr(buf, p)
    nd, = struct.unpack_from("<I", buf, p); p += 4
    dims = struct.unpack_from(f"<{nd}Q", buf, p); p += 8*nd
    type_pos = p
    t, = struct.unpack_from("<I", buf, p); p += 4
    off, = struct.unpack_from("<Q", buf, p); p += 8
    tensors.append((name, dims, t, off, type_pos))

data_start = (p + ALIGN - 1) // ALIGN * ALIGN
print(f"data_start={data_start}  n_tensors={len(tensors)}")

Q6KB = getattr(e0, "Q6K_BLOCK", None)
print("Q6K_BLOCK from e0:", Q6KB)

targets = [(n, d, t, o, tp) for (n, d, t, o, tp) in tensors
           if t == 14 and "ffn_down_exps" in n]
print(f"targets: {len(targets)} Q6_K down_exps tensors")
assert len(targets) == 21, len(targets)

for name, dims, t, off, type_pos in targets:
    n_elem = 1
    for d in dims: n_elem *= d
    assert n_elem % 256 == 0
    old_sz = n_elem // 256 * Q6KB
    f.seek(data_start + off)
    raw = f.read(old_sz)
    assert len(raw) == old_sz, (name, len(raw), old_sz)
    x = e0.deq_q6k(raw)
    assert len(x) == n_elem
    new = q4k_fn(x)
    new_sz = n_elem // 256 * 144
    assert len(new) == new_sz, (name, len(new), new_sz)
    f.seek(data_start + off)
    f.write(new)
    f.seek(type_pos)
    f.write(struct.pack("<I", 12))
    print(f"{name}: Q6K {old_sz} -> Q4K {new_sz}", flush=True)

f.close()
print("DONE — file patched in place")
