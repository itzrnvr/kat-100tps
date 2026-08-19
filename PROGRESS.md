# PROGRESS LOG — KAT 100 t/s Campaign
Running findings ledger. Newest last. Every entry: measured, mechanism, verdict.
Goal: novel + copy BOTH 70-100+ t/s at Q4_K quality.

## System
RTX 3070 Ti 8GB (sm_86, PCIe4 x8 ~16GB/s) + 31GB DDR5 RAM (~55GB/s).
KAT-Coder-V2.5-Dev 35B MoE (40L x 256 experts top-8, GDN hybrid).
Target 19.8GB > VRAM 8GB => experts RAM-resident (-cmoe).

## Milestones (measured)
- stock AR: 12.3 t/s | lucebox AR: 22.4 | CQ3+gbuzhf spec: copy 46.5
- DSpark v2 head (Koopah, cross-model from Qwen3.6-35B): acc 0.4-0.5 novel
- DSpark + ngram compose: copy 47.4/52.2 e2e, 68.5tg | novel 17-18.6
- PIPELINE x COMPOSE (V64): copy 70.0tg/58.1 e2e | novel 22.9tg  <- CURRENT
- mechanics: 82 GPU<->CPU splits/fwd; step=17ms draft + 33ms/verify-row;
  verify cost superlinear in width (expert fragmentation)

## Findings (what doing what — all profiled)
1. Fate prefetch v1: 5x WORSE (sync D2H in decode loop = self-stall).
   Technique not wrong — placement was. v2 designed (async, arbiter).
2. Union restriction v1: acc HELD 0.79 but 9 graph-ops/layer ate gains.
   v3 FUSED (1 static add/layer): overhead gone; K32 neutral on these
   workloads (keep in tree; the fix mattered).
3. Width: w=8 optimal for compose; w12/16 collapse (superlinear verify).
   GOOSE paper: spine should be ngram-deep not head-deep -> sweep queued.
4. CQ4 (Q3_K experts): PPL 7.35 vs 5.71 Q4K floor -> DISCARDED (quality
   gate is binding; byte-cut at <4.5bpw is off-limits).
5. JetSpec causal head: converted clean, but fork's dflash runtime is
   non-causal-marginal-shaped; head outputs input-independent noise.
   Control (Koopah head same binary): acc 0.27, confident -> runtime
   verdict, not conversion bug. Needs causal decoder-graph feature.
6. Cascade v1 (length cap): no-op (ngram never exceeded horizon).
   v2 (per-token logit-scored truncation) BUILT, untested (env gate).
7. Load race: RAM-pressure driven (4GB free=100% crash, 17GB=zero).
   Operational rule: >10GB free before -md launches.
8. Pipeline + no-big-offload guard: AR +33%; guard BREAKS draft-context
   fit -> compose runs pipeline WITHOUT guard (works, faster).
9. Capture->finetune lane: DSCT 6230 samples; disk caps capture at 100
   samples (3GB) — enough to validate. Trainer: fc+L4/L5 (151M params).
   ft pass1: acc 0.19->1.00 train; save crashed (attr bug), fixed,
   rerun with held-out val split in flight.

## In flight
- ft pass2 (val-split, 20min) -> export GGUF -> serve -> novel bench.
  If engine acc follows (0.5->0.7+): novel scales proportionally.

## Queued (user directive: no discards, solve issues)
- GOOSE ngram-spine depth sweep (n-max 12/16/24 ngram-only)
- Cascade v2 bench (built)
- Fate v2: async arbiter prefetch (design in DESIGN.md V58-adjacent)
- Hot-expert VRAM cache (SOFT_MAX readback infra built; needs fork port)
- Expert replication (2605.11537) on hot set
- CATS memory-adaptive verify depth | Bole tree kernels | NInfer port

## Best configs (runbooks/)
- copy+novel: kat-pc3.sh (pipeline x dspark+ngram w8)

## V70 — Head finetune lane closed (2026-08-19)
- Trainer bugs found+fixed: ggml ne1-fastest deq permutation (advisory,
  confirmed: step-0 0.000 -> 0.219 after fix), k_norm skipped, q-gate
  geometry (32 heads no gate — dims prove it), device placement.
- Retrained clean: VAL acc 0.697 held-out (base 0.219 same metric).
- A/B CONTROL (identical stack/flags/prompts, n=3 each):
    ft head:   acc 0.21, tg 10.7-11.6
    base head: acc 0.195, tg 7.7-11.1
  => IDENTICAL. Finetune wash on novel. The 0.30-0.37 "novel baseline"
  was a different protocol (copy-agentic prompts); true novel-story
  acceptance floor ~0.20 both heads.
- Root cause of wash: 100-sample capture (6% of 6230) covers too little
  of KAT's novel distribution; the head saturates what it saw.
- VERDICT: acceptance lever via head finetune = CLOSED at this data
  scale. Do not retrain without full capture.
- Server-protocol traps logged: TIME_WAIT zombie curl "UP", taskkill
  //F arg mangling, pipeline buffer-fallback under RAM pressure
  (contaminates benches — always grep for sched_reserve failure).

## V71 — MARKOV BIAS WAS SILENTLY OFF; now engaged (2026-08-19)
- Found in server log: "DSpark markov bias skipped: runtime block width
  exceeds trained block_size 8" — EVERY prior run used --spec-draft-n-max 8
  => anchor+8 = 9 > 8 => bigram bias (the head's acceptance booster)
  disabled since day one.
- Fix: n-max 7 => exact 8-token blocks. Skip warning: 0 occurrences.
- Measured (identical 3-prompt novel protocol, clean single-client):
    n-max 8 (markov off): tg 10.7-11.6
    n-max 7 (markov on):  tg 17.5-21.0   (+60-80%)
- Also: concurrent-client benching CONTAMINATES (two parallel benches
  showed 5.3-8.1 while solo window showed 16.5-21.3) — always serialize.
- Note: mk7 differs two ways (markov + width 7); width sweep next to
  attribute. Constraint: markov requires n-max <= 7.

## V72 — markov claim FALSIFIED; harness protocol established (2026-08-19)
- Serial python harness (launch->bench->kill, no bash races): nmax=7 vs 8,
  3 trials x 3 novel prompts each.
- RESULT: identical within noise. nmax=7 median 13.10 (max 21.11);
  nmax=8 median 12.97 (max 23.19). BOTH show cold~11 -> warm~21-23 trend.
- V71's "+60% from markov" was WARM-SERVER vs COLD-SERVER confound. Real
  effect of markov bias on novel: none measurable. Claim retracted.
- Warmup = mmap page-in of 19.8GB experts (first prompts fault from disk).
  Steady-state novel decode = 21-23 t/s. Matches old 22.9 ledger figure.
- PROTOCOL RULES (now enforced): (1) streaming decode-phase timing only,
  (2) discard first trial after server load (page-in), (3) one client,
  (4) python-subprocess launches (bash shim silently blocks the exe —
  "command not found" with exit 0 was DLL/env masking).
- ENV FIX: cudart64_12/cublas64_12/cublasLt64_12 copied beside exe
  (fresh shells lost CUDA PATH -> ggml-cuda.dll unloadable).
- Remaining novel gap to 100: need 4.3x. Top lever = verify-cost (width 8
  at mean-len 2.5 wastes ~5 rows/step): cascade v2 next, then hot-expert
  cache + Fate v2 prefetch (expert RAM reads dominate).

## V73 — Cascade v2: 2 bugs fixed, verdict = no-op for greedy (2026-08-19)
- BUG 1 (fixed): keep==0 empty-draft wedge (re-draft/re-truncate loop).
- BUG 2 (fixed): dp.result null-deref on idle slots — segfaulted server
  on first bench request whenever benching != slot 0. Both fixes built.
- MEASURED (harness, warmup-discarded, 3x3):
    base8:    median 13.92, max 27.18
    casc015d: median 14.20, max 28.23
  => parity within noise.
- WHY no-op: result_probs only populate when temp>0 (rejection-sampling
  path) or LLAMA_SPEC_HEADROOM. Greedy decode => probs empty => cascade
  never scores => 0 truncations (confirmed: truncs logged = 0).
- VERDICT: cascade v2 helps only sampled generation. For greedy novel
  the verify-cost lever needs a different mechanism (per-position
  confidence from the head exists as conf[] — the conf_min gate is the
  greedy-native equivalent and was already tested at parity).
- Acceptance on novel steady-state: 0.24-0.38, mean-len ~3.0.

## V74 — ROUTING IS UNIFORM: expert-locality family CLOSED (2026-08-19)
- Instrumented topk logger (KAT_TOPK_LOG env; per-layer selected expert
  ids per graph) — required fixing a REAL pre-existing bug: the union-v3
  observer was DEAD CODE inside the graph_compute error branch (missing
  brace after `return nullptr;`). Union-v3 "parity" measurements were
  actually union-never-ran. Observer now live (and union results must be
  re-validated before ever citing them again).
- MEASURED (novel decode, 227,840 activations):
    global entropy 15.67 bits (max 16.10)
    global top-128 experts: 1.0% of activations
    per-layer top-24 coverage: 0.4% median
- CONCLUSION: no exploitable hot set. Hot-expert cache, expert
  replication (2605.11537), union restriction — all structurally
  inapplicable to KAT. KAT's load-balanced routing sees to that.
- RAM-read physics now the sole novel bottleneck: per committed token
  ~544MB expert reads (40L x top-8 x ~1.7MB Q4 expert) at ~55GB/s.
  Step commits ~3 (mean-len) for width-8 verify -> ~21-28 t/s warm.
  Theoretical perfectly-overlapped ceiling ~55-60; 100 novel at Q4_K
  quality needs bytes/token halved beyond what quantization floor allows.
- Remaining levers: (a) width economics under clean protocol (fewer
  wasted rows), (b) Fate v2 overlap (bandwidth pipelining, not locality).

## V75 — WIDTH CURVE (clean protocol): W=3 optimal, +78% novel (2026-08-19)
- Harness: warmup discarded, 3 trials x 3 novel prompts, streaming decode
  rate. Markov active for all W<=7 (block<=8).
    W=1: median 20.18 (max 33.7)
    W=2: median 21.24 (max 31.7)
    W=3: median 24.71 (max 52.6)  <- OPTIMAL
    W=4: median 19.53 (max 27.2)
    W=8: median 13.92 (max 28.2)  <- old flagship
- Novel acceptance ~0.25-0.38, mean-len ~3.0 => W=3 matches actual commit
  length; every wider step buys rejected rows at 33ms each.
- Prompt-dependent spread: story prompts (repetitive structure, ngram
  hits) reach 30-52; hardest novel prompt floors 16-21 at any width
  (pure RAM-bandwidth bound).
- OLD width sweep (V66-era "w=8 optimal") was measured under the
  contaminated protocol + pre-warmup; superseded.
- NEXT: W=3 confirm (5 trials), copy bench at W=3 (copy wants wide for
  its 21-24 mean-len — expect regression; if so, adaptive width is the
  answer, lucebox has adaptive_verify_width.h to port).

## V76 — W=3 CONFIRMED (n=15): novel median 29.06, max 56.72 (2026-08-19)
- 5 trials x 3 prompts: median 29.06 (vs 24.71 first run; min 17.22).
- Novel stack now: pipeline x dspark+ngram W=3 = 2.1x old flagship median.
- 56.72 peak = best novel decode ever recorded on this system.

## V77 — FLAGSHIP VALIDATED + no-mmap parity (2026-08-19)
- Definitive harness run (lifecycle-owned, warmup discarded, 3 trials x
  3 novel + 2 copy): W=3 flagship
    novel: median 31.73, max 60.30
    copy:  26.0-47.4
  Second confirmation run: novel median 31.85, max 62.92.
- --no-mmap A/B: 31.85 vs 31.73 => PARITY. mmap page-fault tax is not
  in the warm steady-state path. Thread closed.
- NEXT: the 3x question. Verify row costs 33ms measured; physics floor
  (544MB expert reads @ ~55GB/s DDR5) = ~10ms. Where do 23ms/row go?
  Suspects: thread count underfeeding DDR5 (t8 on 16-thread part),
  unvectorized dequant-scalar path, per-row expert gather overhead,
  draft 17ms serialization. Profiling the CPU MoE path next.

## V79 — BANDWIDTH CEILING FOUND; system at hardware limit (2026-08-19)
- Practical DDR5 bandwidth measured (numpy streaming, this machine):
    8 threads: 20.0 GB/s | 16 threads: 23.7 GB/s | copy r+w: 14.6 GB/s
  => ~24 GB/s practical, NOT the 55 GB/s datasheet figure used in all
  earlier cost models (laptop power/thermal limits).
- DECOMPOSITION at flagship (W=3, t12, novel 33.3 t/s, mean-len ~3):
    step ~90ms; verify reads 4 rows x 544MB = 2.2GB
    2.2GB / 24 GB/s = 89.6ms == measured step time
  => verify runs at ~100% OF PRACTICAL BANDWIDTH. No kernel overhead
  left. The "33ms vs 10ms floor" gap was a wrong denominator, not slack.
- CONSEQUENCES:
  (a) Software prefetch/Fate-style read tricks: cannot help — the bus
      is already saturated.
  (b) Union/VRAM caching of hot experts: DEAD for KAT — uniform routing
      (V74) means a fixed K-set covers ~0.5% of natural activations;
      restriction would collapse quality. (Old "K=32 neutral" data was
      from the dead-observer bug — invalid.)
  (c) Remaining levers must REDUCE BYTES/READS: lower expert bits (Q4_K
      floor blocks), higher acceptance (fewer wasted rows), or VRAM
      residency for ALL experts (impossible: 40L x 256 x 1.7MB = 17.4GB
      >> 8GB VRAM).
- STATUS: novel 33.3 med/66.1 peak, copy 51.5 peak at Q4_K quality —
  ~2-3x from 100 by pure hardware ceiling, absent architecture change.

## V80 — FINAL CONFIG MATRIX (t12) (2026-08-19)
- W=3 t12 (UNIVERSAL BEST): novel med 33.3 / peak 66.1; copy 27.6-51.5
- W=8 t12 (copy-only): copy peak 49.9 but median 21-23 (wasted-row tax
  when ngram misses); novel med 18.6. No reason to prefer it anymore.
- SINGLE-CONFIG SERVING: W=3, t12, pipeline, dspark+ngram markov-on.
- Campaign totals vs session start: novel 13.9 -> 33.3 med (+140%),
  peaks to 66; copy holds 51.5 peak. At Q4_K quality floor throughout.
- HOLDING for user instruction per directive (no Fate / new lanes).

## V81 — MTP graft deep-dive: 2 real bugs fixed, still 0% (2026-08-19)
- Trigger: gbuzhf/KAT-Coder-V2.5-Dev-MTP-GGUF README claims donor-head
  acceptance 48-76% on KAT. Controlled A/B on identical engine:
    their build: 0.57-0.75 (sampled AND greedy)
    our CQ3-MTP: 0.00000 (251 drafts, 0 accepted, every protocol)
- OUR GRAFT HAD 2 REAL BUGS (both fixed, byte-verified):
  1. Routed experts (ffn_{gate,up,down}_exps) written expert-FASTEST
     (transpose(1,2,0) C-order) instead of ggml expert-SLOWEST layout —
     every expert was a shuffle of all 256. Round-trip-verified decoder
     showed cos~0.00 vs donor; fixed to expert-major (cos 0.997).
  2. ALL 2D head matmuls stored transposed (builder applied .T on
     [out,in] donor which is already ggml-correct flat order).
  Note: earlier "head verified cos 0.9999" was wrong-vs-wrong (compared
  transposed donor against transposed graft). Lesson recorded.
- AFTER fixes (full head rewrite from verified donor, correct
  orientation, router in-major): acceptance STILL 0.00000.
- Remaining suspects (checked, not guilty): metadata diff (cosmetic
  only), vocab/mask token (identical), head wiring (loads clean, same
  tensor set as their build), q-gate packing (fixed by correct layout).
- CONCLUSION: the delta is the TRUNK's hidden-state contract. Their
  trunk = APEX quant of stock KAT. Ours = CQ2 lineage (BF16 re-encode +
  DeltaNet per-layer head-interleave transforms + re-encoded control
  plane). The head consumes the trunk's h (via embeddings_nextn) —
  self-consistent for AR but evidently misaligned for the donor head.
- NEXT DIAGNOSTIC (queued): logits dump — run target and draft head on
  the same context, compare top-k. Separates "h extraction path differs"
  from "h values drifted".
- dspark comparison verdict UNCHANGED: dspark head (0.36-0.82) remains
  the only working novel drafter on our trunk.
