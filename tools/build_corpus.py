#!/usr/bin/env python
# Build spark calibration corpus (JSONL of text chunks) from scale_swe.parquet
# real agent trajectories. Chunks mirror actual serving traffic:
# system prompt + tool definitions + problem + assistant turns.
# Output: one JSON string per line, each ~2-4KB of text.
import json, sys
import pyarrow.parquet as pq

SRC = "D:/merge/train/scale_swe.parquet"
OUT_TRAIN = "D:/lucebox/optimizations/spark/corpus/kat_train.jsonl"
OUT_TEST = "D:/lucebox/optimizations/spark/corpus/kat_test.jsonl"
N_TRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 400   # trajectories to use
CHUNK_CHARS = 3500                                        # ~900-1100 tokens each

def chunk_text(t: str):
    return [t[i:i+CHUNK_CHARS] for i in range(0, len(t), CHUNK_CHARS) if t[i:i+CHUNK_CHARS].strip()]

def main():
    t = pq.read_table(SRC, columns=["instance_id","messages","tools","problem_statement","solved"])
    n = t.num_rows
    import pandas as pd
    idx = list(range(0, n, max(1, n // N_TRAJ)))
    rows = t.take(idx).to_pandas().head(N_TRAJ * 2)
    train_chunks, test_chunks = [], []
    # alternate split at trajectory level (every 5th -> test)
    tri = 0
    for _, r in rows.iterrows():
        msgs = list(r["messages"]) if r["messages"] is not None else []
        parts = []
        for m in msgs[:14]:  # cap turns per traj
            role = m.get("role", "")
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            if isinstance(c, str) and c.strip():
                parts.append(f"[{role}] {c[:2000]}")
        text = "\n\n".join(parts)
        chunks = chunk_text(text)
        if tri % 5 == 0:
            test_chunks += chunks[:2]
            train_chunks += chunks[2:]
        else:
            train_chunks += chunks
        tri += 1
        if len(train_chunks) > 2600:  # Spark used 333 chunks; we use more for richness
            break
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for c in train_chunks:
            f.write(json.dumps(c) + "\n")
    with open(OUT_TEST, "w", encoding="utf-8") as f:
        for c in test_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"train: {len(train_chunks)} chunks -> {OUT_TRAIN}")
    print(f"test:  {len(test_chunks)} chunks -> {OUT_TEST}")

if __name__ == "__main__":
    main()
