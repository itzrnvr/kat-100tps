# PURPOSE: Compact-rewrite the in-place-patched Ornith GGUF into a valid file.
# LESSON (V114a): llama.cpp gguf_init_from_reader enforces STRICT offset
# adjacency — tensor N must start exactly at tensor N-1's end. Shrinking
# tensors in place (Q6_K->Q4_K at same offset) leaves holes that invalidate
# every following tensor's offset check -> loader rejects the file.
# This script reads each tensor's CURRENT type to compute its (new) size,
# copies that many bytes from the ORIGINAL offset (valid bytes live at slot
# start), and writes a fresh file with compact sequential layout + patched
# offset fields. Reclaims ~1.45GB of holes.
import os, struct, time

SRC = r"D:\merge\out\Ornith15-Q4KM.gguf"
OUT = r"D:\merge\out\Ornith15-Q4K-CQ.gguf"
ALIGN = 32

f = open(SRC, "rb")
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

tensors = []  # (name, dims, type, old_off, off_pos_in_header)
for _ in range(n_tensors):
    name, p = rstr(buf, p)
    nd, = struct.unpack_from("<I", buf, p); p += 4
    dims = struct.unpack_from(f"<{nd}Q", buf, p); p += 8*nd
    t, = struct.unpack_from("<I", buf, p); p += 4
    off_pos = p
    off, = struct.unpack_from("<Q", buf, p); p += 8
    tensors.append((name, dims, t, off, off_pos))

old_data_start = (p + ALIGN - 1) // ALIGN * ALIGN

# bytes per element per ggml type (matches loader's ggml_nbytes)
BPE = {0:4, 1:2, 2:1, 3:1, 6:1, 7:1, 8:1.0625, 9:1, 10:1.0625, 11:1,
       12:0.5625, 13:0.6875, 14:0.8203125, 15:0.5625, 16:1.0, 24:0.5625,
       25:0.5, 26:0.5, 27:0.5, 28:0.59375, 29:0.5, 30:2, 36:1.015625, 37:1.03125}

def tsize(dims, t):
    n = 1
    for d in dims: n *= d
    bpe = BPE[t]
    nb = int(n * bpe)
    assert abs(nb - n*bpe) < 1.0 or (n*bpe)%1==0, (t, n)
    # int types must be exact
    nb = int(round(n*bpe)) if (n*bpe)%1 else int(n*bpe)
    return nb

new_offs = []
cur = 0
for name, dims, t, off, off_pos in tensors:
    sz = tsize(dims, t)
    cur = (cur + ALIGN - 1) // ALIGN * ALIGN
    new_offs.append((cur, sz))
    cur += sz
new_data_len = cur

total_in = sum(sz for _, sz in new_offs)
print(f"tensors={len(tensors)} new data length={new_data_len/1e9:.3f} GB "
      f"(holes reclaimed: {(os.path.getsize(SRC)-old_data_start-new_data_len)/1e9:.3f} GB)", flush=True)

# write output: header copy with patched offsets, then compact data
g = open(OUT, "wb")
g.write(buf[:old_data_start])  # header incl. tensor dir; offsets patched after
# patch offsets in the header copy
g.seek(0)
for (name, dims, t, off, off_pos), (noff, sz) in zip(tensors, new_offs):
    g.seek(off_pos)
    g.write(struct.pack("<Q", noff))
g.seek(old_data_start)  # new data section starts at same aligned position

t0 = time.time()
copied = 0
CH = 64 << 20
for i, ((name, dims, t, off, off_pos), (noff, sz)) in enumerate(zip(tensors, new_offs)):
    assert g.tell() == old_data_start + noff, (name, g.tell(), noff)
    f.seek(old_data_start + off)
    left = sz
    while left > 0:
        chunk = f.read(min(CH, left))
        if len(chunk) == 0:
            raise IOError(f"short read {name}")
        g.write(chunk)
        left -= len(chunk)
    copied += sz
    # re-align for next tensor
    pad = (-sz) % ALIGN
    if pad:
        g.write(b"\0" * pad)
    if i % 50 == 0 or i == len(tensors)-1:
        el = time.time()-t0
        print(f"[{i+1}/{len(tensors)}] {copied/1e9:.2f} GB  {copied/1e6/el:.0f} MB/s", flush=True)

g.close(); f.close()
print(f"DONE {os.path.getsize(OUT)} bytes", flush=True)
