# PURPOSE: Parse the LOCAL Ornith-1.5 Q4_K_M GGUF header; dump tensor
# inventory grouped by layer-class (attention vs linear-attn/ssm vs MoE),
# confirm the tensor set matches what gypsy-dragon's qwen35moe arch expects,
# and record data-start offset for later graft surgery.
import struct, sys, collections

PATH = r"D:\merge\out\Ornith15-Q4KM.gguf"

buf = open(PATH, "rb").read(64 << 20)  # header + tensor infos + tokenizer KVs
assert buf[:4] == b"GGUF", buf[:4]
ver, = struct.unpack_from("<I", buf, 4)
n_tensors, = struct.unpack_from("<Q", buf, 8)
n_kv, = struct.unpack_from("<Q", buf, 16)
p = 24

def rstr(b, o):
    n, = struct.unpack_from("<Q", b, o)
    return b[o+8:o+8+n].decode("utf-8"), o + 8 + n

kvs = {}
for _ in range(n_kv):
    k, p = rstr(buf, p)
    t, = struct.unpack_from("<I", buf, p); p += 4
    if t == 0:   v = struct.unpack_from("<B", buf, p)[0]; p += 1
    elif t == 1: v, = struct.unpack_from("<b", buf, p); p += 1
    elif t == 2: v, = struct.unpack_from("<H", buf, p); p += 2
    elif t == 3: v, = struct.unpack_from("<h", buf, p); p += 2
    elif t == 4: v, = struct.unpack_from("<I", buf, p); p += 4
    elif t == 5: v, = struct.unpack_from("<i", buf, p); p += 4
    elif t == 6: v, = struct.unpack_from("<f", buf, p); p += 4
    elif t == 7: v = bool(struct.unpack_from("<B", buf, p)[0]); p += 1
    elif t == 8: v, p = rstr(buf, p)
    elif t == 9:
        et, = struct.unpack_from("<I", buf, p); p += 4
        n, = struct.unpack_from("<Q", buf, p); p += 8
        v = ("array", et, n)
        if et == 8:
            for _ in range(n): _, p = rstr(buf, p)
        else:
            p += n * {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(et, 1)
    elif t == 10: v, = struct.unpack_from("<Q", buf, p); p += 8
    elif t == 11: v, = struct.unpack_from("<q", buf, p); p += 8
    elif t == 12: v, = struct.unpack_from("<d", buf, p); p += 8
    else: raise ValueError(f"kv type {t}")
    kvs[k] = v

print(f"GGUF v{ver}  tensors={n_tensors}  kv={n_kv}")
for k in sorted(kvs):
    if k.startswith("qwen3_5_moe.") and "tokenizer" not in k:
        v = kvs[k]
        if isinstance(v, tuple): v = f"{v[0]}[{v[1]}]x{v[2]}"
        print(f"  {k} = {v}")

tensors = []
for _ in range(n_tensors):
    name, p = rstr(buf, p)
    nd, = struct.unpack_from("<I", buf, p); p += 4
    dims = struct.unpack_from(f"<{nd}Q", buf, p); p += 8 * nd
    ttype, = struct.unpack_from("<I", buf, p); p += 4
    off, = struct.unpack_from("<Q", buf, p); p += 8
    tensors.append((name, dims, ttype, off))

# group suffixes across trunk layers 0..39
suf = collections.Counter()
for name, dims, ttype, off in tensors:
    if ".blk." in name:
        parts = name.split(".")
        suf[".".join(parts[2:])] += 1
    else:
        suf[name] += 1

print("\n-- trunk tensor suffixes (count over layers) --")
for s, c in sorted(suf.items()):
    print(f"  {c:3d}  {s}")

data_start = min(t[3] for t in tensors)
print(f"\ndata_start (first tensor offset) = {data_start}")
print(f"file size = {len(open(PATH,'rb').read(0)) or __import__('os').path.getsize(PATH)}")
# type histogram
tc = collections.Counter(t[2] for t in tensors)
print("type histogram:", dict(tc))
