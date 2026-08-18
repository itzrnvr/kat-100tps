#!/usr/bin/env python
"""Build the DSCT token file for dspark-capture from the gh_* coding corpus.
Tokenizes via the running llama-server /tokenize endpoint, writes:
  magic u32 'DSCT', version u32=1, n_samples u32
  per sample: sample_id u64, seq_len u32, ids i32[seq_len], loss_mask u8[seq_len]
loss_mask: 0 for prompt part, 1 for the tail 25% (train region marker;
passed through untouched by the forward).
Usage: python make_dsct.py [n_files] [out.bin]
"""
import glob, json, struct, sys, time, urllib.request

PORT = "8035"
NFILES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = sys.argv[2] if len(sys.argv) > 2 else "D:/merge/train/dsct_tokens.bin"
MAXLEN = 1600  # tokens per sample (capture n_ctx=2048)

def tokenize(text):
    body = json.dumps({"content": text}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/tokenize",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["tokens"]

def main():
    files = sorted(glob.glob("D:/merge/train/gh_*.parquet"))
    import pyarrow.parquet as pq
    samples = []
    sid = 0
    for fp in files[:NFILES]:
        try:
            tbl = pq.read_table(fp)
        except Exception:
            continue
        texts = []
        if "task" in tbl.column_names:
            texts += ["TASK: " + str(t) for t in tbl.column("task").to_pylist() if t]
        if "messages" in tbl.column_names:
            for msgs in tbl.column("messages").to_pylist():
                for m in msgs or []:
                    c = (m or {}).get("content") or ""
                    if c:
                        texts.append(str(c))
        for t in texts:
            if len(t) < 200:
                continue
            try:
                ids = tokenize(t[:20000])
            except Exception:
                continue
            # chunk long texts
            for s0 in range(0, min(len(ids), 6400), MAXLEN):
                chunk = ids[s0:s0+MAXLEN]
                if len(chunk) < 64:
                    continue
                mask = [0]*len(chunk)
                for i in range(int(len(chunk)*0.75), len(chunk)):
                    mask[i] = 1
                samples.append((sid, chunk, mask))
                sid += 1
            if sid % 400 == 0:
                print(f"  {sid} samples", flush=True)
        if len(samples) > 6000:
            break
    print(f"total samples: {len(samples)}")
    with open(OUT, "wb") as f:
        f.write(struct.pack("<III", 0x54435344, 1, len(samples)))
        for sidc, ids, mask in samples:
            f.write(struct.pack("<QI", sidc, len(ids)))
            f.write(struct.pack(f"<{len(ids)}i", *ids))
            f.write(bytes(mask))
    import os
    print(f"DONE {OUT} {os.path.getsize(OUT)/1e6:.0f} MB")

if __name__ == "__main__":
    main()
