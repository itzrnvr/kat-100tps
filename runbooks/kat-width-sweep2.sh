#!/bin/bash
# PURPOSE: Complete width curve W=5,6 (W=7 clean: 17.5-21.0; W=3/4 suspect).
#          Serial, RAM-gated, one server at a time.
LOG=C:/merge/train/dswidth2.log
SRV=/c/src/lmdspark/build/bin/Release/llama-server.exe

wait_ram () {
  for r in $(seq 1 30); do
    FREE=$(powershell -NoProfile -c "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r')
    [ -z "$FREE" ] && FREE=0
    if awk "BEGIN{exit !($FREE > 10)}"; then return 0; fi
    sleep 5
  done
  return 1
}

powershell -NoProfile -c "Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue" 2>/dev/null
sleep 6

for W in 5 6 7; do
  wait_ram || { echo "RAM GATE FAILED W=$W" >> $LOG; continue; }
  for i in $(seq 1 6); do
    KAT_PIPELINE=1 "$SRV" -m C:/merge/KAT-CQ3-MTP.gguf \
      -md D:/merge/kat-dspark-v2-q8.gguf \
      --spec-type draft-dspark,ngram-mod --spec-draft-n-max $W \
      --spec-draft-p-min 0.75 \
      --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 24 --spec-ngram-mod-n-match 48 \
      -ngl 99 -cmoe -t 8 -ctk q8_0 -ctv q8_0 -fa on -c 8192 --port 8036 >> $LOG 2>&1
    sleep 2
    if curl -s -m 2 http://127.0.0.1:8036/health | grep -q '"ok"'; then
      echo "=== WIDTH $W UP ===" >> $LOG
      break
    fi
  done
  (cd /d/merge/train && python bench_novel.py 2>&1 | grep tg= | sed "s/^/W=$W /" >> $LOG)
  powershell -NoProfile -c "Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue" 2>/dev/null
  sleep 6
done
echo SWEEP_DONE >> $LOG
