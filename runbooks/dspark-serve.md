# DSpark serving runbook (best speed configs, measured 2026-08-18)

## Best COPY/REPETITIVE workloads — 47.4 med / 52.2 peak t/s e2e (3.9x stock AR)
```bash
cd C:/src/lmdspark/build/bin/Release
./llama-server.exe -m C:/merge/KAT-CQ3-MTP.gguf \
  -md D:/merge/kat-dspark-v2-q8.gguf \
  --spec-type draft-dspark,ngram-mod --spec-draft-n-max 8 \
  --spec-draft-p-min 0.75 \
  --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 24 --spec-ngram-mod-n-match 48 \
  -ngl 99 -cmoe -t 8 -ctk q8_0 -ctv q8_0 -fa on -c 16384 --port 8035
```
NOTE: intermittent "invalid vector subscript" at draft load — retry loop
launches until /health ok (usually 1-3 tries).

## Best NOVEL workloads — ~22 t/s (tie with lucebox AR)
```bash
./llama-server.exe ... --spec-type draft-dspark --spec-draft-n-max 3 ... (rest same)
```
w=3 beats w=8 on novel text (17.0 vs 10.6 t/s): acceptance decays
0.71->0.10 across the block; deep verify rows are waste (33ms/row).

## Known tradeoffs (measured)
- NOT bit-exact vs AR at temp 0 (DFlash K/V injection numerics); outputs
  remain coherent. Use stock CQ3 recipe when bit-exactness matters.
- -ngld all = 2x REGRESSION (GPU-resident draft contends with target).
- conf-min 0.35: parity (not adopted).
- Union-v2 (KAT_UNION_K): safe w/ warmup, net-neutral at K>=24.

## Cost model (V49): step = 17ms draft + 33ms x verify_rows.
## 100 t/s needs 33->~10ms/row (Bole-class fused/tree-verify patch).
