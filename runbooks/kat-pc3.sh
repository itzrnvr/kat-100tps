#!/bin/bash
LOG=C:/merge/train/dsppc3.log
for i in $(seq 1 8); do
  KAT_PIPELINE=1 ./llama-server.exe -m C:/merge/KAT-CQ3-MTP.gguf \
    -md D:/merge/kat-dspark-v2-q8.gguf \
    --spec-type draft-dspark,ngram-mod --spec-draft-n-max 8 \
    --spec-draft-p-min 0.75 \
    --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 24 --spec-ngram-mod-n-match 48 \
    -ngl 99 -cmoe -t 8 -ctk q8_0 -ctv q8_0 -fa on -c 8192 --port 8035 >> $LOG 2>&1
  sleep 2
  if curl -s -m 2 http://127.0.0.1:8035/health | grep -q ok; then echo UP >> $LOG; break; fi
done
