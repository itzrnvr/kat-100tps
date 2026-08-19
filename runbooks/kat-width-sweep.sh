#!/bin/bash
# PURPOSE: Novel verify-width sweep. Cost model: step = 17ms draft + 33ms/row;
#          novel mean-len 2.4-2.5 => width 8 wastes 5 rows/step. Sweep n-max.
LOG=C:/merge/train/dswidth.log
cd /c/src/lmdspark/build/bin/Release || exit 1
for W in 3 4 5 6 7; do
  for i in $(seq 1 6); do
    KAT_PIPELINE=1 ./llama-server.exe -m C:/merge/KAT-CQ3-MTP.gguf \
      -md D:/merge/kat-dspark-v2-q8.gguf \
      --spec-type draft-dspark,ngram-mod --spec-draft-n-max $W \
      --spec-draft-p-min 0.75 \
      --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 24 --spec-ngram-mod-n-match 48 \
      -ngl 99 -cmoe -t 8 -ctk q8_0 -ctv q8_0 -fa on -c 8192 --port 8036 >> $LOG 2>&1
    sleep 2
    if curl -s -m 2 http://127.0.0.1:8036/health | grep -q '"ok"'; then
      echo "=== WIDTH $W UP ===" >> $LOG; break
    fi
  done
  cd /d/merge/train && python bench_novel.py 2>&1 | grep tg= | sed "s/^/W=$W /" >> $LOG
  powershell -c "Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue"
  sleep 4
done
echo SWEEP_DONE >> $LOG
