# Ornith-1.5-35B-A3B evaluation tooling (V112+)

- pardl2.py — parallel range downloader, direct-offset writes, Windows-safe
  (os.pwrite absent; per-thread handles). Resume via .pstate.json sidecar.
- orn_mtp_scan.py — remote GGUF header parser via HTTP ranges (correct GGUF
  v3 KV type map incl. string-array walking). Used on mudler APEX-MTP 71GB.
- orn_local_scan.py — same, local file; tensor inventory + type histogram.
- orn_vs_kat_arch.py — side-by-side arch/bytes comparison KAT-CQ3 vs Ornith
  (found: identical shapes; official quant has 21/41 down_exps at Q6_K).
- orn_requant_q4.py — IN-PLACE GGUF surgery: deq Q6_K -> quant Q4_K at same
  offset, patch type field. Dead hole left behind (harmless).
- orn_bench.py — identical-protocol bench (3 novel + 2 copy, warmup discard,
  ORN_SPEC=mtp env switches spec stack).
- orn_ppl.py — paired PPL gate, same binary/corpus/flags both models.

KEY FACTS:
- official Ornith-1.5 Q4_K_M ships the complete MTP head (blk.40.*, 41 blocks)
- ggml type ids: 8=Q8_0, 12=Q4_K, 14=Q6_K, 15=Q4_0 (t8 is NOT Q6_K)
- draft-mtp on Ornith: acceptance 0.196-0.36, hangs server -> dspark champion

## Tooling addendum (V114-V115 session)
- orn_compact.py — GGUF full-rewrite compactor; BPE map ground-truthed for
  types {0,12,14}; lesson: audit constants against measured tensor bytes.
- orn_v115_orchestrator.py — strictly-serial V115 pipeline (waits on KAT PPL,
  delete broken, re-download official via pardl2 with pstate-based completion
  check, official PPL, CQ PPL, verdict). Idempotent per stage; relaunch-safe.
- V115-TEMPLATE.md — pre-registered gates (PASS <=+0.05 / MARGINAL <=+0.30 /
  FAIL >+0.30 on official-vs-CQ same-model delta).
