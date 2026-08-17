#!/usr/bin/env python
# KAT-CQ2-MTP v2: byte-patch CQ2's ORIGINAL header (all 54 KV pairs intact),
# bump block_count 40->41 (u32 @ off 1000), append nextn_predict_layers=1 as
# KV #55, then append the rebuilt tensor table (733 base + 20 MTP tensors).
# Data section: byte-copy CQ2 tensors, then grafted blk.40.nextn.* tensors.
#
# BUG THIS FIXES [2026-08-17]: v1 wrote only 4 KV pairs but declared n_kv=3,
# and dropped 50 required keys -> gguf "failed to read tensor info".
# v2 keeps every original KV byte and only patches what must change.
import struct, sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_gguf import SrcGGUF
import e0
tensor_nbytes = e0.tensor_nbytes  # BUGFIX: build_gguf version divides all blocks by QK_K=256; Q8_0 uses 32-el blocks (1/8 size error)

SRC  = "D:/merge/out/bs-mtp-iq4xs.gguf"
BASE = "D:/merge/out/KAT-CQ2.gguf"
OUT  = "C:/merge/KAT-CQ2-MTP.gguf"
ALIGN = 32

def main():
    t0 = time.time()
    base = SrcGGUF(BASE)
    src  = SrcGGUF(SRC)

    # ---- collect MTP tensors (rename -> blk.40.nextn.*) ----
    mtp = []
    for name, (dims, tt, toff, tp, op) in src.entries.items():
        if not name.startswith("blk.40."):
            continue
        # BUGFIX [2026-08-17]: keep donor names VERBATIM. The donor's own
        # layout (which stock draft-mtp already loads: 61-68% acceptance
        # measured) has plain blk.40.<name> for block tensors and nextn.*
        # only on the 4 special MTP tensors. The earlier rename to
        # blk.40.nextn.* was lucebox's convention and stock rejected it.
        new_name = name
        mtp.append((new_name, name, dims, tt, toff, tensor_nbytes(dims, tt)))
    mtp_dims = {nn: (dims, tt) for (nn, sname, dims, tt, toff, nb) in mtp}
    print(f"base {len(base.entries)} + mtp {len(mtp)} tensors")

    # ---- plan data offsets ----
    entries = []   # (new_name, src_file, src_abs_off, nbytes)
    cur = 0
    for n, (dims, tt, toff, tp, op) in base.entries.items():
        nb = tensor_nbytes(dims, tt)
        entries.append((n, BASE, base.header_len + toff, nb))
        cur = (cur + nb + ALIGN - 1) // ALIGN * ALIGN
    for new_name, sname, dims, tt, toff, nb in mtp:
        entries.append((new_name, SRC, src.header_len + toff, nb))
        cur = (cur + nb + ALIGN - 1) // ALIGN * ALIGN
    total_data = cur

    # ---- build header: original bytes + patches ----
    hdr = bytearray(base.header[:base.header_len])  # full original: magic..kv..tensor-table
    # We rebuild FROM the kv end (tensor table replaced anyway).
    # Locate KV end + original n_kv/n_tensors positions:
    #   magic(4) ver(4) n_tensors(8) n_kv(8)  => n_tensors @12, n_kv @20
    # GGUF v3: magic@0 ver@4 n_tensors@8 n_kv@16 (v1 wrongly patched 12/20)
    struct.pack_into("<Q", hdr, 8, len(entries))          # n_tensors: 733 -> 753
    struct.pack_into("<Q", hdr, 16, 55)                   # n_kv: 54 -> 55
    struct.pack_into("<I", hdr, 1000, 41)                 # block_count u32 -> 41

    # KV section ended at 10943779 (verified). Insert new KV pair AT THE END
    # of KV section, then new tensor table after it.
    kv_end = 10943779
    new_kv = bytearray()
    key = b"qwen35moe.nextn_predict_layers"
    new_kv += struct.pack("<Q", len(key)) + key
    new_kv += struct.pack("<I", 4)                        # BUGFIX: type 4 = UINT32 (type 0 is UINT8/1-byte — caused 3-byte walker desync)
    new_kv += struct.pack("<I", 1)                        # value 1

    tbl = bytearray()
    offs = []
    cur = 0  # data offsets are relative to data_start; compute after header len known
    def build_tbl():
        # offsets in the TABLE are relative to data-section start (0-based)
        t = bytearray()
        o = []   # relative offsets
        c = 0
        for (nn, sf, soff, nb) in entries:
            b = nn.encode()
            t += struct.pack("<Q", len(b)) + b
            if sf == BASE:
                dims, ttype = base.entries[nn][0], base.entries[nn][1]
            else:
                dims, ttype = mtp_dims[nn]
            t += struct.pack("<I", len(dims))
            for d in dims: t += struct.pack("<Q", d)
            t += struct.pack("<I", ttype)
            t += struct.pack("<Q", c)
            o.append(c)
            c = (c + nb + ALIGN - 1) // ALIGN * ALIGN
        return t, o, c

    # iterate to fixpoint on data_start (header size affects alignment only)
    data_start = kv_end  # ALIGN=32 matches CQ2's own data-start alignment
                          # (SrcGGUF computed header_len with the same 32)
    for _ in range(3):
        tbl, offs, end = build_tbl()
        data_start = (kv_end + len(new_kv) + len(tbl) + ALIGN - 1) // ALIGN * ALIGN
    header = bytes(hdr[:kv_end]) + bytes(new_kv) + bytes(tbl)
    header += b"\x00" * (data_start - len(header))
    assert len(header) == data_start

    # ---- write ----
    files = {BASE: open(BASE, "rb"), SRC: open(SRC, "rb")}
    with open(OUT, "wb") as out:
        out.write(header)
        for i, (nn, sf, soff, nb) in enumerate(entries):
            out.seek(data_start + offs[i])   # offs are relative now
            f = files[sf]
            f.seek(soff)
            remaining = nb
            while remaining > 0:
                chunk = f.read(min(1 << 24, remaining))
                if not chunk: raise RuntimeError(f"short read {nn}")
                out.write(chunk); remaining -= len(chunk)
            if (i + 1) % 120 == 0:
                print(f"  [{i+1}/{len(entries)}] {out.tell()/2**30:.1f} GiB ({time.time()-t0:.0f}s)", flush=True)
    for f in files.values(): f.close()
    print(f"DONE {OUT} {total_data/2**30:.2f} GiB data + header {data_start} in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
