#!/bin/bash
# Collect routing traces from KAT-CQ2 on dflash_server for:
#   (a) CCT table (build_cct.py)  (b) spark.csv warm start  (c) pre-gate training
# Feeds calibration corpus chunks as prompts, collects per-layer routing.
set -e
BIN=D:/lucebox/server/build/Release/dflash_server.exe
MODEL=D:/merge/out/KAT-CQ2.gguf
P=8022
TRACE=D:/merge/train/kat_routing.bin
CORPUS=D:/lucebox/optimizations/spark/corpus/kat_train.jsonl

psmux kill-session -t dftrace 2>/dev/null || true
psmux new-session -d -s dftrace -x 220 -y 40
psmux send-keys -t dftrace "DFLASH_COLLECT_ROUTING=$TRACE $BIN $MODEL --port $P --target-device cuda:0 --spark 2>&1 | tee -a D:/merge/train/dftrace.log" Enter

echo "waiting for load..."
for i in $(seq 1 40); do
  sleep 10
  curl -s -m 2 http://127.0.0.1:$P/health >/dev/null 2>&1 && { echo "READY ($((i*10))s)"; break; }
done

python - "$P" "$CORPUS" <<'EOF'
import json, sys, time, urllib.request
port, corpus = sys.argv[1], sys.argv[2]
chunks = [json.loads(l) for l in open(corpus, encoding="utf-8")][:120]
print(f"feeding {len(chunks)} chunks...")
t0 = time.time()
for i, c in enumerate(chunks):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "dflash", "temperature": 0,
                         "max_tokens": 32,
                         "messages": [{"role": "user", "content": c[:3000]}]}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=180).read()
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(chunks)} ({time.time()-t0:.0f}s)", flush=True)
print(f"fed all in {time.time()-t0:.0f}s")
EOF

sleep 3
psmux send-keys -t dftrace C-c
sleep 2
ls -la "$TRACE" 2>/dev/null && echo "NEXT: python D:/merge/train/build_cct.py $TRACE D:/merge/train/kat_cct.bin"
