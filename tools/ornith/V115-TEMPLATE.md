# V115 verdict template (REVISED — pre-commitment after advisory)

## GATE DESIGN CORRECTION (recorded before numbers land)
1. KAT-vs-Ornith PPL is CROSS-MODEL: it shows quality territory, not requant
   damage (Ornith's baseline may differ from training alone). The true
   isolation requires Ornith-official-PPL vs Ornith-CQ-PPL (same model,
   only 21 down_exps differ).
2. The pre-patch Ornith baseline is GONE — we patched in place and the file
   is offset-corrupt (unloads). Recovering the true baseline = re-download
   official (pardl2, ~16 min) + one more PPL run (~57 min).
3. Thresholds re-aligned to campaign standard (V98: strict +0.05, marginal
   +0.30). The earlier +0.5/+1.0 was unjustifiably loose and left a dead
   band. New gates, measured as Ornith-CQ minus Ornith-official:
     - PASS:      delta <= +0.05
     - MARGINAL:  +0.05 < delta <= +0.30  (judgment: consider re-Q6_K on
                  worst-N layers of the 21, or accept for copy workloads)
     - FAIL:      delta > +0.30           (revert; serve official quant)
4. KAT-CQ3 same-run PPL is retained as CONTEXT (binary/corpus/flags anchor),
   not as the gate.

## MEASUREMENT ORDER (machine dedicated, no disk contention)
1. [running] KAT-CQ3 PPL — anchor + context (~50 min remaining)
2. delete broken Ornith15-Q4KM.gguf (unloadable, dead for serving+PPL)
3. re-download official Ornith Q4_K_M via pardl2 (~16 min)
4. Ornith-official PPL (~57 min)  <- TRUE baseline
5. Ornith-CQ PPL (~57 min)        <- patched
6. Verdict from delta(5 vs 4)

## FILL ON COMPLETION
- KAT-CQ3 (context):            PPL = ____
- Ornith-official (baseline):   PPL = ____
- Ornith-CQ (patched):          PPL = ____
- Requant delta (5-4):          ____  -> PASS / MARGINAL / FAIL

## IF PASS — champion Ornith serve command:
llama-server -m D:\merge\out\Ornith15-Q4K-CQ.gguf -md D:\merge\kat-dspark-v2-q8.gguf
  --spec-type draft-dspark,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75
  --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 24 --spec-ngram-mod-n-match 48
  -ngl 99 -cmoe -fa on -ctk q8_0 -ctv q8_0 -t 12 -c 8192
(resident: add --load-mode none, needs ~18GB free; mmap works from ~10GB)
Measured: novel 23.2 med / verbatim 53.5 med (mmap, warm-cache caveat).

## CAVEATS CARRIED INTO THE VERDICT
- V114 speeds are upper bounds (warm-cache confound unquantified)
- Ornith card benchmarks self-reported (+5-19 over Qwen3.6 base)
- Sweeping quality eval (verify_swe.py Docker protocol) not yet run; PPL is
  necessary-not-sufficient
- RTN Q4_K on those 21 layers vs official imatrix Q6_K: imatrix had
  activation-weighted calibration we didn't replicate — this is exactly what
  the delta measures
