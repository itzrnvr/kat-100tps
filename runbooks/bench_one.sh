#!/bin/bash
# PURPOSE: Single-config launcher + bench, strictly one at a time.
#          Usage: bench_one.sh <n-max> <logname>
W=$1
LOG=C:/merge/train/$2.log
SRV=/c/src/lmdspark/build/bin/Release/llama-server.exe
cd /c/src/lmdspark/build/bin/Release || exit 1
for i in $(seq 1 6); do
  KAT_PIPELINE=1 "$SRV" -m C:/merge/KAT-CQ3-MTP.gguf \
    -md D:/merge/kat-dspark-v2-q8.gguf \
    --spec-type draft-dspark,ngram-mod --spec-draft-n-max $W \
    --spec-draft-p-min 0.75 \
    --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 24 --spec-ngram-mod-n-match 48 \
    -ngl 99 -cmoe -t 8 -ctk q8_0 -ctv q8_0 -fa on -c 8192 --port 8036 >> $LOG 2>&1
  sleep 3
  if curl -s -m 2 http://127.0.0.1:8036/health | grep -q '"ok"'; then echo "UP W=$W" >> $LOG; break; fi
done
cd /d/merge/train && python bench_novel.py >> $LOG 2>&1
echo "BENCH_DONE W=$W" >> $LOG
powershell -NoProfile -c "Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue"
