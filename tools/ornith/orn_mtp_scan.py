# PURPOSE: Parse a remote GGUF header via HTTP range requests and list tensors.
# Used to inventory the native MTP head tensors in mudler's Ornith-1.5 APEX-MTP
# BF16 GGUF without downloading the 71GB file.
import struct, sys, urllib.request

URL = "https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF/resolve/main/Ornith-1.5-35B-A3B-BF16-MTP.gguf"

def fetch(pos, n):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={pos}-{pos+n-1}"})
    return urllib.request.urlopen(req, timeout=120).read()

buf = fetch(0, 48 << 20)  # header + tensor infos + big tokenizer KVs
assert buf[:4] == b"GGUF", buf[:4]
ver, = struct.unpack_from("<I", buf, 4)
n_tensors, = struct.unpack_from("<Q", buf, 8)
n_kv, = struct.unpack_from("<Q", buf, 16)
p = 24

def rstr(b, o):
    n, = struct.unpack_from("<Q", b, o)
    s = b[o+8:o+8+n].decode("utf-8")
    return s, o + 8 + n

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
        if et == 8:  # array of strings: each is u64 len + bytes
            for _ in range(n):
                _, p = rstr(buf, p)
        else:
            p += n * {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(et, 1)
    elif t == 10: v, = struct.unpack_from("<Q", buf, p); p += 8
    elif t == 11: v, = struct.unpack_from("<q", buf, p); p += 8
    elif t == 12: v, = struct.unpack_from("<d", buf, p); p += 8
    else: raise ValueError(f"kv type {t}")
    kvs[k] = v

print(f"GGUF v{ver}  tensors={n_tensors}  kv={n_kv}")
for k in ("general.architecture", "general.name", "qwen3_5_moe.block_count",
          "qwen3_5_moe.expert_count", "qwen3_5_moe.embedding_length"):
    if k in kvs: print(f"  {k} = {kvs[k]}")

tensors = []
for _ in range(n_tensors):
    name, p = rstr(buf, p)
    nd, = struct.unpack_from("<I", buf, p); p += 4
    dims = struct.unpack_from(f"<{nd}Q", buf, p); p += 8 * nd
    ttype, = struct.unpack_from("<I", buf, p); p += 4
    off, = struct.unpack_from("<Q", buf, p); p += 8
    tensors.append((name, nd, dims, ttype, off))

ai = kvs.get("general.architecture", "?")
data_off_hint = None
# tensors are sorted by offset; header end = first tensor offset
align = 32
print("\n-- tensors beyond trunk (blk >= 40) or MTP-ish --")
for name, nd, dims, ttype, off in tensors:
    is_extra = False
    if "blk." in name:
        try:
            idx = int(name.split("blk.")[1].split(".")[0])
            is_extra = idx >= 40
        except ValueError:
            pass
    if is_extra or any(s in name.lower() for s in ("mtp", "draft")):
        print(f"  {name:60s} nd={nd} dims={dims} type={ttype} off={off}")

print(f"\nfirst tensor off={tensors[0][4]}  last tensor off={tensors[-1][4]} name={tensors[-1][0]}")
# total = last offset + its size estimate
print(f"n_tensors parsed = {len(tensors)}")
