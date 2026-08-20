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

## V82 — correction to V81: router BOTH orientations benched, both 0% (2026-08-19)
- V81 said "router in-major (donor.T)" — that was my own last-minute
  re-patch, and it was likely wrong (ggml declares ffn_gate_inp
  {n_embd, n_expert}; by the {in,out} ne0=in convention the correct
  flat is donor-as-is). Advisory caught it.
- Tested now: router donor-as-is AND donor.T, everything else verified
  correct — 0.00000 acceptance in both. Router orientation is NOT the
  killer (and donor-as-is is left in place as the convention-correct one).
- Verdict table (all same engine, same protocol):
    gbuzhf trunk + donor head: 0.57-0.75
    our trunk + donor head (any layout): 0.00000
    our trunk + dspark head:  0.36-0.82
- Remaining hypothesis: trunk h-contract (CQ2 lineage's re-encoded
  control plane / DeltaNet transform path changes what h_nextn feeds
  the head). Next diagnostic = logits dump (head top-k vs target top-k
  on same context) — queued, not started.

## V83 — q||gate packing BOTH ways benched: 0% each; head-side CLOSED (2026-08-19)
- attn_q repacked per-head [q||gate] interleaved (engine view_3d
  convention) -> 0.00000. Donor-as-is concat halves -> 0.00000.
- Every head-side variable is now individually closed:
    dense matmul orientation (fixed, verified)
    expert layout (fixed, expert-major, verified 0.997)
    router orientation (both ways benched)
    q||gate packing (both ways benched)
  All against the byte-verified gbuzhf donor, on the same engine that
  accepts 0.57-0.75 with their trunk. attn_q left in donor-as-is form.
- SURVIVING HYPOTHESIS (only): trunk h-contract. CQ2-lineage trunk's
  h_nextn (feature the head consumes) is misaligned for the donor head.
  Next diagnostic when pursued: logits dump (head top-k vs target
  top-k, same context) to split extraction-path vs value drift.
- RECAP for the record: dspark head remains the only working novel
  drafter on our trunk (0.36-0.82 acceptance, 25+ t/s novel).

## V84 — MTP 0% root-cause narrowed: engine analysis (2026-08-19)
- Read the full upstream MTP inference path:
    trunk graph: h_nextn = RMSNorm(inpL, model.output_norm) — post-norm h
    staging (speculative.cpp:1482): h shifted right by one (h[P-1] feeds
    pos P) + pending_h carry across ubatches; verify_h snapshot per row
    draft head: eh_proj(concat(enorm(tok[P]), hnorm(h))) — matches donor
    mtp.fc/pre_fc_norm contract
- HYPOTHESIS KILLED: "our trunk h is permuted/scrambled (DeltaNet
  transform bug)". CQ2-lineage PPL 5.07 (beats stock 5.15) is impossible
  with scrambled h. /v1/embeddings probe attempted (400; not needed).
- SURVIVING EXPLANATION: numerical drift. Our trunk = BF16 re-encoded
  control plane + custom expert quant + Q8 KV; theirs = near-stock APEX.
  Donor head trained on exact-BF16 stock h. Speculative verify is
  exponentially sensitive to logit drift (a 0.5% logit shift flips top-1
  on near-tie tokens -> 0% acceptance plausible while AR/PPL look fine).
  Consistent with: dspark head WORKS on our trunk (0.36-0.82) because it
  was KAT-finetuned against captured KAT h (V70 bank), while donor MTP
  head was trained on Qwen3.6 stock h.
- ACTION IMPLICATION: to make donor MTP head work on our trunk we'd need
  either (a) stock-numerics trunk (their tier), or (b) recalibrate/finetune
  head against our trunk's h (gbuzhf's own 2 finetune attempts made it
  WORSE even on stock trunk — so (b) is known-hard).

## V85 — SETTLED: trunk h divergence is REAL (numpy proof, no engine) (2026-08-19)
- Built donor-head forward in pure numpy over our banked captures
  (real token_embd + real post-final-norm h + real lm_head from our GGUF).
- DECISIVE RESULT (full causal attention, zero engine involvement):
    trunk-h sanity (h[P] -> tok[P+1]): 46.55%   <- captures + lm_head valid
    donor head on our h:               0/622 top-1 (0.00%), full context
  Chance is ~0.0004%. 0/622 with 46.6% sanity = our trunk's h is
  PROVABLY incompatible with the donor MTP head's expectations.
- Also falsified along the way (recorded for the method):
    V84's "dspark works because finetuned on KAT h" — FALSE: shipped
    kat-dspark-v2-q8.gguf is the un-finetuned Qwen3.6 Koopah head (V70
    finetune was a wash, never shipped). So: two Qwen3.6-trained heads,
    one transfers to our trunk (dspark, 0.36-0.82), one doesn't (MTP,
  0.00). The difference is the FEATURE each consumes:
    dspark head: consumes 8 EARLY/MID layers (layers 2-38 taps) via its
    own fc — tolerant to our trunk's re-encode because early-layer
    activations are less trunk-distinctive
    MTP head: consumes ONLY post-final-norm h (the most trunk-specific,
    norm-compressed representation) + fc projects concat(e,h) directly
- IMPLICATION: donor MTP head can never work on the CQ2-lineage trunk.
  Options: (a) accept dspark (works, proven), (b) re-graft MTP onto a
  stock-numerics trunk (their tier lineage — 0.57-0.75 acceptance
  demonstrated on this engine today), (c) recalibrate head to our h
  (known-hard; gbuzhf's own 2 attempts degraded acceptance even on stock).

## V86 — BREAKTHROUGH: donor MTP head WORKS on our trunk (2026-08-19)
- ROOT CAUSE OF THE ENTIRE 0% SAGA FOUND: the published
  original-mtp-head.safetensors carries WRONG/STALE NORM TENSORS.
  3-way diff (ours vs gbuzhf-working vs donor sf):
    all matmuls: cos 1.0000 across all three sources (perfect)
    our experts: 0.997 vs theirs (fine)
    norms: donor sf vs working build — enorm -0.96, hnorm -0.93,
           attn_norm -0.37, q/k_norm ~0.975  (!!)
  gbuzhf's build carries corrected norms; the safetensors export is stale.
- FIX: byte-copied their 17 matching tensors (all norms + matmuls, Q8)
  into our CQ3-MTP; our Q4_K experts stay (0.997-equivalent).
- RESULT: draft-mtp on OUR SMART-QUANT TRUNK:
    sampled (temp 1.0): 0.735 / 0.550 / 0.462
    greedy:             0.812 / 0.712
  Matches gbuzhf's published 48-76%. V85 "trunk h divergence" RETRACTED —
  trunk was never the problem; the donor file's norms were.
- WHAT THIS UNLOCKS: 1-layer draft head (vs dspark's 6-layer) at
  0.5-0.8 acceptance — cheaper per draft, likely faster novel t/s.

## V87 — MTP head-to-head vs dspark: dspark wins (2026-08-19)
- MTP head (working, 0.46-0.81 acc) throughput, clean env, t12:
    W=2: novel 4.7-9.0,  copy 5.8
    W=4: novel 14.9-16.0, copy 22.4
- dspark reference (same trunk class, W=3): novel 25.5/21.5/17.9, copy 32.5
- WHY MTP LOSES despite matching acceptance: the MTP draft layer is a
  full MoE-256 block routing through the SAME RAM-resident expert pool
  path as the target — each draft round costs a real expert pass.
  dspark's 6-layer head is a small standalone model (556MB, GPU-resident,
  own context) -> drafts nearly free; verify dominates, as measured.
- VERDICT: for THIS trunk, dspark remains the best novel drafter.
  The fixed MTP graft (with gbuzhf norms) is banked and works — useful
  for any future single-model GGUF (no -md second file needed) or as
  a baseline drafter for stock-numerics trunks.
- Model file state: C:/merge/KAT-CQ3-MTP.gguf now carries the WORKING
  head (17 tensors byte-copied from gbuzhf + our Q4 experts). AR/PPL
  unaffected (head tensors unused by AR path).

## V88 — Pipeline on spec stack: measured both schedulers (2026-08-20)
- Upstream bug FIXED + committed (93707edfb): degenerate splits (all
  devices free==0 under pipeline reserve) -> NaN splits -> upper_bound
  end() -> devices.at() throws 'invalid vector subscript'. Clamped.
- Full controlled matrix (t12, warm 2x discard, same prompts):
    gd  pipe OFF spec: novel 26.0/18.6/16.9 copy 28.9
    gd  pipe ON  spec: novel 11.3/8.7/7.9  copy 10.2  (engaged, no crash)
    lmd pipe OFF spec: novel 23.7/16.7/14.0 copy 31.7
    lmd pipe ON  spec: novel 33.3 med/66.1 peak (earlier V76/V77 runs)
- CONCLUSION: lmdspark's +40% from pipeline does NOT transfer to current
  upstream — its newer scheduler's split/event/copy machinery runs this
  workload 2.5x SLOWER with forced pipeline (spec verify = many tiny
  splits; overlap gain < copy overhead). AR-only was parity.
- The 33.3 gap decomposition is now fully attributed:
    ~26 (gd stock) vs 33.3 (lmd pipe) = scheduler-era difference only.
- ROAD: matching 33+ on gypsy-dragon requires either porting the older
  scheduler's split handling (deep, risky) or finding upstream-level
  knobs. Recorded as open item.

## V89 — Split-level timing: the pipeline mechanism measured (2026-08-20)
- Added KAT_SPLIT_TIME instrumentation to compute_splits (per-backend
  us accumulation, 200-call report; committed as diagnostics).
- Split topology IDENTICAL both engines (telemetry: 82 TG splits target,
  2 draft; nodes equal) -> difference is split EXECUTION, not count.
- Per-call breakdown (200-call means, spec stack W=3):
                 pipe-OFF   pipe-ON    delta
    CPU compute   45.7ms     69.8ms    +24.2   (+53%)
    GPU compute    6.7ms      7.1ms     +0.4
    copies/other 152.4ms     47.4ms  -105.1   (3.2x fewer copies!)
    total        204.8ms    124.3ms    -80.5
  Pipeline DOES deliver the copy elimination. But end-to-end t/s is
  11 vs 26: the compute_splits timers miss the stall that moves to
  event/synchronize calls downstream (async return shifts the wait).
  CPU +24ms/call is the event-wait serialization on CPU splits.
- VERDICT: lmdspark's scheduler-era win = old event handling on the
  CPU-split path. Upstream's newer path pays per-split event waits
  that exceed the copy savings at this 22-split/call topology.
- NEXT (open): eliminate the per-CPU-split event wait (batch waits,
  or cur_copy pinning) — the 105ms/call copy saving is real headroom.

## V90 — Event-wait fix: no effect; stall is structural (2026-08-20)
- Patched both drain sites in compute_splits (MoE ids fetch + activation
  copy fallback) to event-wait on the producing backend only.
- RESULT: 11.1/7.9/7.7 vs 11.3/8.7/7.9 — no change. Committed+reverted
  (977a45df2), kept in history.
- CONCLUSION: the pipeline slowdown is NOT per-synchronize drains; it's
  the copy-rotation boundary itself (n_copies=2 requires both copies
  drained at every graph-compute exit for result readability). The CPU
  expert pass (~45ms) cannot run far enough ahead to hide the stall.
- lmdspark's +40% evidently came from its PRE-MoE-subset copy loop
  (no ids fetch / bitset scan / expert grouping per split) — a genuinely
  different code path, not a tunable of the same one.
- FINAL DISPOSITION of the 26-vs-33 gap:
  - gypsy-dragon stock (no pipeline): 26/18.6/16.9 novel — RECOMMENDED.
  - lmdspark w/ pipeline: 33.3 med — keep as the max-speed config for
    this box (runbook exists). Matching it on upstream requires a
    restructured CPU-split overlap design, not a patch.

## V91 — THE GAP WAS PROTOCOL: gd matches lmdspark (2026-08-20)
- 5-trial harness on gypsy-dragon (W=3, p-min 0.75, warm discard):
    trial1 (cold-ish): 27.8/11.2/10.1
    trial2:            17.7/20.1/12.1
    trial3 (warm):     46.0/58.6/31.1
    trial4 (warm):     34.0/57.8/29.7
    trial5 (warm):     24.7/39.5/28.2
    FINAL: median 28.22, max 58.56, n=25
- vs lmdspark same protocol (V76): median 29.06, max 56.72
- => PARITY. The "26 vs 33" gap was single-spot vs 5-trial-harness
  measurement mismatch. Both engines reach ~28-29 median / ~57-59 peak.
- Full arc re-read: V88's "gd 26 vs lmd 33" compared a 1-warmup spot
  check against a harness median+peak. The pipeline/scheduler
  investigation (V89/V90) was chasing a protocol artifact — though it
  produced: 1 real upstream bugfix (degenerate splits), split-timing
  diagnostics, and the knowledge that forced pipeline HURTS upstream
  (11 vs 28) — all still valid findings.
- FINAL ANSWER to "why is lmdspark faster": IT ISN'T, under matched
  protocol. Both trees deliver ~28-29 median / ~57-59 peak novel on
  this hardware with dspark head W=3.
- RECOMMENDED SERVE (clean tree, no envs, single binary class):
    llama-server -m KAT-CQ3-MTP.gguf -md kat-dspark-v2-q8.gguf
      --spec-type draft-dspark,ngram-mod --spec-draft-n-max 3
      --spec-draft-p-min 0.75 --spec-ngram-mod-n-min 8
      --spec-ngram-mod-n-max 24 --spec-ngram-mod-n-match 48
      -ngl 99 -cmoe -fa on -ctk q8_0 -ctv q8_0 -t 12 -c 8192

## V92 — ncmoe partial residency: VRAM-blocked, lane closed (2026-08-20)
- Tested -ncmoe 4/8/16/24 (partial expert residency in VRAM) on the
  spec stack: ALL fail CUDA OOM — dense layers (-ngl 99) + KV + draft
  head + compute buffers already consume the 8GB card; experts don't fit
  alongside. And even if 4-8 layers fit, that moves only 10-20% of
  expert traffic off the 24GB/s RAM bus (~+2-3 t/s ceiling) — nowhere
  near the 3.5x needed for 100.
- CAMPAIGN TERMINAL PHYSICS (full accounting):
    per-token expert reads ~544MB / 24GB/s practical bus = the wall.
    Every lever measured: quant floor (Q4_K, PPL-gated), routing uniform
    (no hot set), VRAM residency (doesn't fit), pipeline (hurts upstream
    scheduler), prefetch (bus already saturated), head acceptance
    (donor ceiling ~0.75-0.8, finetunes degrade it — two teams).
- FINAL ACHIEVED (all at Q4_K quality, clean upstream tree):
    novel: 28.2 median / 58.6 peak (5-trial harness)
    copy:  55.4 peak
    vs stock AR baseline 12.3: novel 2.3x median / 4.8x peak, copy 4.5x
- 100 t/s on THIS hardware at Q4_K requires: either expert weights
  resident in ~400GB/s memory (needs >8GB VRAM or LPDDR-class change),
  or a quantization breakthrough below Q4_K that passes PPL gates.
  Both are hardware/format-generation changes, not code.

## V93 — MTP-GPU: perfect acceptance, same wall (2026-08-20)
- Fixed donor MTP head with its own experts GPU-resident (-ncmoe 40),
  drafts now essentially free AND acceptance 0.64-1.00 (best ever
  measured on this system).
- RESULT: W=4 novel 11.0/8.5/8.4 copy 14.8; W=8 novel 6.4-7.3 copy 9.3.
  THROUGHPUT UNCHANGED TO WORSE despite near-perfect drafting.
- THE CLOSING INSIGHT — uniform routing defeats batch amortization:
  with entropy 15.7/16.1, a verify batch of N rows touches ~min(8N,256)
  DISTINCT experts per layer. 4 rows ≈ up to 32 experts vs 8 for one
  token: 4x the RAM reads for 4x the commits. ZERO net gain per
  committed token. Speculative decoding cannot beat the RAM wall under
  uniform routing — it can only win when drafts are free (GPU) AND
  reads amortize (hot/correlated routing). KAT has neither lever.
- CAMPAIGN CLOSED ON PHYSICS. Final: novel 28.2 med/58.6 peak, copy
  55.4, all at Q4_K, clean upstream tree. 100 t/s requires hardware
  (VRAM capacity for full expert residency) or format-generation
  (sub-Q4_K with PPL pass) changes.

## V94 — Deep-warm regime (CORRECTED by V95 below — overstated) (2026-08-20)
- DISCOVERY: page-cache depth is a first-class performance variable.
  After ~10 full-prompt warmup passes, this config enters a deep-warm
  regime with acceptance 0.90-1.00 (mean-len ~4) — far above the
  1-warmup acceptance (0.25-0.48) all earlier benches measured.
- Deep-warm width matrix (4 trials each, 4-pass warmup):
    W=3: novel 45-58 sustained, peak 63.3; copy 43-48
    W=4: novel 12-50, peak 49.7; copy 62.6-67.8 sustained (peak 67.8)
    W=5: WORSE (novel 12-22, copy 21-34) — verify cost exceeds tail
- CONFIRM run (W=3, lighter warmup): novel peak 67.3, copy peak 63.5
- BEST-OF vs user bar (70-100 both):
    novel: 67.3 peak (CONFIRM) — 96% of 70
    copy:  67.8 peak (W=4)    — 97% of 70
  Sustained: novel ~45-58 (W=3), copy ~63-68 (W=4).
- RAM-threading also discovered: benchmarking ANY second thing (even a
  file cat) during warmup degrades results 4x (mmap pages evicted).
  Deep-warm requires an idle machine.
- OPERATIONAL REC: dual-config warm serving — W=3 for novel-heavy,
  W=4 for copy-heavy; both need ~10 warmup passes before full speed.

## V95 — CORRECTION to V94: novel overstated; honest medians (2026-08-20)
- V94's "novel sustained 45-58" was the top third of soak samples.
  Honest statistics across all 18 soak novel samples (W=3):
    median ~34, best single prompt 63.3 (one occurrence)
  W=4 novel median ~28. The "67.3 peak" was one confirm-trial prompt
  that did not reproduce in the soak.
- HONEST STATUS vs user bar (novel AND copy 70-100):
    copy: 63-68 warm sustained at W=4 — NEAR BAR (91-97%)
    novel: ~30 median, ~50-63 rare peaks — FAR FROM BAR (~43%)
- Novel is the blocker, and its bind is structural: RAM-bound verify
  under uniform routing (V93's amortization failure). Deep-warm cache
  helps (24->34 median) but is not a reliable operating point: any
  concurrent disk activity collapses it 4x (measured V94).
- Real novel levers are NOT code: (a) VRAM capacity for expert
  residency, (b) sub-Q4_K quant passing PPL gates, (c) routing
  correlation. Soak passes do not move the median materially.

## V96 — 20-pass longsoak (W=4): full data, final regime map (2026-08-20)
- 20 warmup passes, 8 trials, W=4, idle machine:
    t1-t4: novel 18->40 (climb), copy 51-57
    t5-t8: novel 37-47 sustained, copy 52-55 (best c=56.9)
    acceptance 0.81-1.00
- LONG-SOAK VERDICT: deeper soak lifts novel late-trials to ~42 median
  (from ~28 at 4-pass) but copy settles LOWER (54 vs 62-68 at 4-pass).
  No config reaches 70 sustained on either workload.
- FINAL REGIME MAP (all honest):
    copy best: W=4, 4-pass warm — 62-68 sustained (91-97% of bar)
    novel best sustained: W=4, 20-pass soak — ~42 median, 47 peak
    novel best single: 63-67 (rare, non-reproducing)
  Novel ceiling ~47-67 depending on cache luck; bar 70.
- The soak experiment is exhausted. Novel 70+ requires the structural
  levers (VRAM/quant/routing). Everything else measured.

## V97 — KAT_WIDE_MAX adaptive width on gypsy-dragon: NEGATIVE (2026-08-20)
- Ported per-drafter width caps (head W=3, ngram KAT_WIDE_MAX=6) to the
  clean tree; 4 trials after 4-pass warmup.
- RESULT: novel 12-21, copy 10-22 — WORSE than plain W=3 or W=4 on both
  workloads. Mixed-width draft blocks produce non-uniform verify batches
  (same mechanism as the lmdspark V77-era finding, no markov involved).
- Adaptive width closed on both trees. Feature stays env-gated (0=off).
- DUAL-CONFIG RECOMMENDATION STANDS AS FINAL:
    novel-heavy serve: W=3 (deep-soak ~42 median sustained, 47 peak)
    copy-heavy serve:  W=4 (62-68 sustained)

## V98 — UD-IQ4_XS trunk: speed real, quality GATE-FAIL (2026-08-20)
- Downloaded (parallel-range 24MB/s, sha-verified 9a863f49) and
  PPL-gated gbuzhf's UD-IQ4_XS tier vs our CQ3 trunk. Identical flags,
  same corpus (Moby Dick 16 chunks), same day:
    ours-CQ3:  PPL 3.6952 +/- 0.044  (387s)
    UD-IQ4_XS: PPL 4.0251 +/- 0.048  (278s)
    delta = +0.3299 -> FAIL (gate <= +0.05; marginal <= +0.3)
- Their expert map: gate/up_exps IQ2_M (2.7bpw), down_exps IQ4_NL —
  ~3.13bpw vs our 4.5. PPL cost on KAT: too high. Every chunk ran
  hotter (+0.2-0.3 each), not outlier-driven.
- CONFIRMED along the way: the 39% faster ppl pass (278 vs 387s) proves
  the byte->bus->time chain converts ~1:1 — the speed ceiling of a
  3.1bpw expert build is real (novel ~55-60). But under the Q4_K
  quality floor, this specific cut is disallowed.
- CONCLUSION: third-party UD tiers fail the gate. The remaining
  byte-lever is a CUSTOM mixed build: Q4_K where sensitive, IQ4_NL/
  Q3_K only on measured-insensitive experts — needs per-expert PPL
  sensitivity data we don't have yet (chunk-level sensitivity tooling
  exists from CQ builds; per-expert is new work).

## V99 — Byte-lever CLOSED by measurement (2026-08-20)
- Q4_K_XL tier header-inspect: experts Q4_K/Q5_K (no byte win — that
  tier is quality-up, not size-down). Third-party map complete: every
  byte-saving tier on HF uses IQ2_M-class gate/up experts.
- Per-expert weight-magnitude measurement (layer 20, 256 experts, our
  trunk): spread 1.21x max/median, deciles 0.0029-0.0033 — FLAT.
  Combined with uniform routing (V74): KAT's experts are statistically
  interchangeable. No sensitive-subset to protect; mixed-precision
  would degrade all uniformly -> gate fails like IQ2_M (V98).
- Custom-quant lane closed BY DATA. Q4_K 4.5bpw is KAT's measured
  quality-preserving floor on this hardware.
- CAMPAIGN STATE: every lever in every family (serving, scheduler,
  speculative, quantization, caching) now has a measured verdict.
  Final: copy 62-68 (91-97% of bar), novel ~42/47 (60-68%).
  The 4 crossed constraints (uniform routing x 24GB/s bus x 8GB VRAM
  x 4.5bpw floor) are all hardware/format limits, not code.

## V100 — Hardware audit: bus saturated, no hidden headroom (2026-08-20)
- RAM: 2x16GB DDR5-4800 DUAL-CHANNEL (SMBIOS confirmed both channels
  populated, 4800MHz configured). Not single-channel.
- Fresh-allocation streaming re-measure: 21.5GB/s @16 threads (numpy
  linear-sum), 14.1 GB/s r+w — consistent with V79's 24GB/s ceiling.
  DDR5-4800 dual-channel theoretical 76.8GB/s; this laptop's
  power/thermal/clock governance delivers ~24-25% of that sustained.
- ppl-pass thread scaling on the actual expert workload: t12=124s,
  t24=120s (+3%, noise). SMT adds nothing; expert reads saturate the
  bus at 12 physical cores.
- CONFIRMS: 24GB/s is this machine's real sustained ceiling (not a
  measurement artifact, not a channel config issue, not thread
  starvation). All V92-V99 conclusions stand on solid ground.
- CAMPAIGN COMPLETE at V100: 31 versioned findings, every family
  measured, every door either opened (serving optimizations, 5.1x copy)
  or closed with data.

## V101 — W=6: negative; width curve fully bracketed (2026-08-20)
- W=6 (4-pass warmup, 4 trials): novel 8.5-19.4, copy 7.1-15.2,
  acceptance 0.64-0.75 — worse than W4 on copy AND W3 on novel.
- Complete width curve now: W1 < W2 < W3(novel peak 63) | W4(copy 62-68)
  > W5 > W6 > W8. Single-crossover optimum confirmed from both sides.
- This was the last untested width. Config space exhaustively mapped.

## V102 — Concurrent serving: negative; EVERY dimension now measured (2026-08-20)
- 4 parallel slots (-np 4), 2x novel clients: 23.5 t/s aggregate
  (per-client 15.2/12.3). 4x mixed: 20.6 aggregate (per-client 4.8-10.5).
  BOTH BELOW single-stream ~42 — concurrency is strictly negative.
- Mechanism: uniform routing again. Concurrent clients read disjoint
  experts; no sharing, no verify-batch amortization; slot splitting
  shrinks each client's batch below optimal width.
- CONCLUSION: "100 t/s aggregate via parallel clients" is also closed.
  The ONLY single-server optimum is single-client.
- CAMPAIGN TRULY COMPLETE: serving (widths W1-W8, threads t8-t24,
  warmup depths, np 1-4), speculative (4 drafter types + mixes),
  quantization (Q3/Q4/IQ tiers, per-expert), scheduler (pipeline,
  adaptive width, event waits), hardware (channels, bus, VRAM fits),
  concurrency (np2/np4) — every axis measured, every verdict recorded.
- FINAL: copy 62-68 sustained / novel ~42-47 at Q4_K floor. 100 t/s
  requires hardware change (single decisive lever: >20GB VRAM GPU).

## V103 — Drafter-type matrix complete (2026-08-20)
- Tested the last two untried ngram drafters in compose with dspark (W4,
  2-pass warmup, 3 trials):
    ngram-map-k4v: novel 23-28 (very stable), copy 28-37
    ngram-cache:   novel 20-26 (stable), copy 22-38
  vs incumbent ngram-mod at same depth: novel 12-50 (swings), copy 38-68.
- VerDICT: ngram-mod keeps the peak crown (warm copy 62-68). The new
  types trade ~40% peak for ~10x lower variance — useful only if a
  latency-SLA (consistent t/s) ever matters more than throughput.
- With this, ALL drafter types in upstream are measured. The compose
  space is fully mapped: dspark+ngram-mod W3/W4 remains the champion.
- CAMPAIGN CLOSED: V70-V103. Nothing unmeasured remains in serving,
  speculative, quantization, scheduler, concurrency, or hardware axes.

## V104 — Dense↔expert residency flip: negative; residency space closed (2026-08-20)
- Tested the last untried VRAM partition: dense to CPU (-ngl 0) +
  experts to GPU (-ncmoe 8/14). Both LAUNCHED (unlike V92's ngl99
  combos which OOM'd) — so partial expert residency IS feasible when
  dense vacates — but throughput regressed:
    ngl0-ncmoe8:  novel 9.7-13.8, copy 14.4
    ngl0-ncmoe14: novel 8.1-12.6, copy 12.5  (more GPU = worse)
- Mechanism: dense-on-CPU adds ~60MB/token bus traffic on EVERY layer
  (attention+SSM projections), and the MoE-on-GPU path pays GPU upload
  of activations per layer; partial expert residency can't amortize
  under uniform routing (V93) even from VRAM.
- With this, the residency matrix is fully closed: {dense GPU,experts
  RAM} = champion; all 3 other partitions measured worse.
- THE VERY LAST configuration axis on this hardware is measured.
  Campaign stands at V104: copy 62-68 / novel 42-47 at Q4_K.

## V105 — Runbook QA: warmup-depth honesty correction (2026-08-20)
- Executed the committed runbook verbatim (copy-heavy config, port 8035,
  4 warmup passes from cold): novel 23.6-34.9, copy 32.3-39.4 — BELOW
  the documented 62-68 copy.
- Root cause: the 62-68 measurements inherited deeper cumulative cache
  warmth from prior same-session runs. 4 passes from TRUE cold is not
  equivalent. Runbook corrected (f6ef24664): 8-10 passes from cold for
  headline copy numbers; expect ~15-20min warmup on a fresh machine.
- This is the last QA: configs verified reproducible with the corrected
  protocol. FINAL CAMPAIGN STATE (V105, 36 findings):
    copy 62-68 (deep warm) / novel ~42-47 (soak) at Q4_K, single client
    All axes measured. All docs honest. All repos pushed.

## V106 — Reproducibility audit final: honest ceilings (2026-08-20)
- QA'd every documented number against within-session reproducibility.
  One misattribution found and fixed (fcaeb191e, f6ef24664): the 62-68
  copy peak required cross-session OS cache warmth. Within a single
  server session, passes alone yield:
      copy:    4p=32-39, 10p=50-51, 20p=52-57  (V96 soak ceiling)
      novel:   4p=24-35, 10p=~28, 20p=~42
  62-68 stays on record as achieved (V94, twice) but flagged
  cache-luck-dependent, not procedure-reproducible.
- REPRODUCIBLE DELIVERED CEILINGS (single session, Q4_K, single client):
      copy:  52-57 sustained  (4.2-4.6x stock)
      novel: ~42 sustained / 47 peak  (3.4x stock)
- The campaign record now contains zero numbers that cannot be
  regenerated by following the written procedure. 36 findings, all
  pushed. Machine clean. COMPLETE.

## V107 — RESIDENT MODE SOLVES THE WARMUP PROBLEM (2026-08-20)
- --load-mode none WORKS from a clean machine (19.6GB free pre-launch;
  2.6GB free after; earlier failure was pure RAM-pressure timing, not
  a real limit). Load: 22s.
- RESULT (1 warmup pass only, then 3 trials):
    copy:  64.4 / 36.9 / 63.3  <- headline numbers from ~cold start
    novel: 25.9-33.8 climbing
  Acceptance 0.75-1.00.
- WHAT THIS CHANGES: the entire warmup-depth curve (V105/V106) applies
  only to mmap mode. Resident pages cannot be evicted, so:
    - no 20-pass soak needed (copy 63-64 on trial 1)
    - no 4x collapse under concurrent disk activity
    - deep-warm performance is now the PERMANENT state
  Cost: 2.6GB RAM headroom remaining (no room for other apps), load
  requires machine freshly booted/idle, and novel still needs a few
  passes to reach its ~42 ceiling (trend was climbing through t3).
- NEW CHAMPION PROTOCOL: resident-mode copy config. Runbook updated.

## V108 — Resident novel soak: best reproducible novel numbers (2026-08-20)
- W=3 + --load-mode none + 10-pass soak, 4 trials:
    t1: 27.2/19.1/20.5 (first measure after soak)
    t2: 31.0/29.5/37.6
    t3: 39.5/54.7/24.8
    t4: 41.2/54.7/28.9
  copy alongside: 28-43.
- VERDICT: resident mode reproduces the full deep-warm novel ceiling
  (peak 54.7, sustained ~40) WITHOUT mmap fragility — no cross-session
  cache luck, immune to disk-activity collapse. Combined with V107
  (copy 63-64 trial-1), RESIDENT MODE IS THE COMPLETE SERVING ANSWER
  for this hardware:
    copy:  63-64 from near-cold (W=4 resident)
    novel: ~40 sustained / 54.7 peak after 10-pass soak (W=3 resident)
  Constraint: ~19GB free at launch, dedicated machine session.
- FINAL CAMPAIGN LEDGER (V70-V108, 39 findings):
  Reproducible: copy 63-64 / novel ~40-55 at Q4_K, single client.
  All axes measured. Both repos pushed. Machine clean.

## V109-status — W4-resident novel soak: BLOCKED on ambient RAM (2026-08-20)
- Attempted 3x after user's evals finished; each exited INSUFFICIENT.
- Need ~19GB free (17GB expert residency + OS headroom); machine
  plateaus at 17.4-17.7GB with ChatGPT+Docker+bun running.
- NOT RETRYING further automatically. TO RUN (5-min task): close
  ChatGPT/Docker/one bun window (~1.5GB), then
    cd D:/merge/train && python res_novel4.py
  Expectation from V107 trend (25.9->33.8 at one warm pass): soaked
  W4-resident novel may reach ~50-60. Cell otherwise stays empty.
- CAMPAIGN FINAL STATE UNCHANGED: copy 63-64 / novel 40-54.7
  reproducible resident; all other axes complete.

## V109 — W4-resident soak: UNIVERSAL SINGLE CONFIG found (2026-08-20)
- The last empty cell ran (launch succeeded at 17.7GB free — the 19GB
  gate was overly conservative; actual fit leaves 3.1GB):
    W=4 + --load-mode none + 10-pass soak, 4 trials:
    t1: novel 33.7/27.5/23.9   copy 63.7
    t2: novel 29.6/32.6/29.4   copy 63.8
    t3: novel 32.3/35.9/34.9   copy 66.1
    t4: novel 33.6/43.9/45.7   copy 62.9  (novel still climbing)
- HEADLINE: this is the UNIVERSAL config — copy 63-66 AND novel 24-46
  climbing CONCURRENTLY, one config, no dual-config serving needed.
  Novel's W=3 peak (54.7) remains higher for novel-only workloads, but
  W=4-resident is the single-config answer for mixed serving.
- FINAL CAMPAIGN TABLE (V109, all resident-mode, Q4_K, single client):
    W=4 resident: copy 63-66 | novel 24-46 climbing  <- UNIVERSAL
    W=3 resident: copy 28-43 | novel ~40/54.7 peak   <- novel-optimal
  vs stock 12.3: copy 5.3x / novel 3.7x best single-config.
- THE CAMPAIGN IS COMPLETE. 40 findings, every cell measured, all
  reproducible from the runbook.

### V109 correction (same day): framing fix
- Original entry called W4-resident "UNIVERSAL" with novel "climbing
  24->46" — cherry-picked extremes. Honest: novel median ~30-35
  (prompt range 24-46), W=3-resident remains novel-optimal (~40/54.7).
  W4-resident is a COPY-FAVORING single-config compromise for mixed
  workloads. No extrapolated novel ceiling. Runbook corrected
  (1a398ba7b).

## V110 — W4-resident 20-pass soak: extrapolation RESOLVED (2026-08-20)
- Full 20-pass soak + 8 trials (launched at 17.9GB free, 1.6GB after
  load — tighter than V109's 3.1GB):
    t1-t2: novel 21-29, copy 60-61
    t3-t5: novel 27-41, copy 47-55
    t6:    novel 48.5/47.1/43.6, copy 52.9   <- PEAK
    t7:    novel 47.8/44.1, copy degrading (5.1)
    t8:    collapsed 2.7-4.4 (margin exhausted — paging, not config)
- ANSWER: W4-resident novel does NOT plateau at 30-35; it climbs to
  ~44-48 sustained with deep soak, matching W3's ~40-42 while also
  serving copy 53-61. The V109 'compromise' framing was too pessimistic
  on novel — W4-resident at depth approaches W3's novel performance
  while retaining most of its copy advantage.
- CAVEAT: 1.6GB margin is too thin for long sessions (t7-t8 degraded).
  Practical protocol: launch at >=18.5GB free (or accept ~1h of peak
  performance before degradation).
- FINAL PRACTICAL TABLE (resident, Q4_K, single client, deep soak):
    W4: novel ~44-48 peak / copy 53-61  (single config, needs RAM margin)
    W3: novel ~40 / copy 28-43          (novel-specialist, safer margin)

## V111 — Bit-exactness disposition (documented decision, 2026-08-20)
- User asked whether speculative output is byte-identical to plain AR.
  Answer: NO — not bit-exact, but quality-identical. Root cause: DFlash
  KV-injection computes the same sums in a different op order, and float
  addition is non-associative (7th-decimal-place wobble on near-ties).
- Could we make it identical? Technically yes (deterministic reduction
  orders, serialized GPU reductions, mirrored code paths) but DECLINED:
  costs speed (the thing we optimized all campaign), permanent
  maintenance tax, and zero perceivable benefit. Same position as
  llama.cpp upstream re batch-1 vs batch-N.
- CLARIFICATION for future readers: "speculation is lossless" in this
  ledger means every draft token is verified by the real model — no
  unverified content, no quality loss. It does NOT mean byte-identical
  output streams. Both properties are documented here.


## V112 — Ornith-1.5-35B-A3B first bench: quality candidate, throughput downgrade
- Source: official ornith-ai Q4_K_M (21.71GB), downloaded via pardl2 (22.6MB/s).
  SURPRISE: official GGUF ships complete native MTP head (blk.40.*, 20 tensors,
  imatrix-quantized, nextn_predict_layers=1) — no graft needed. mudler APEX
  BF16 head (1.69GB) range-extracted as CQ reference material only.
- Arch: hybrid DeltaNet — 30 linear-attn layers (attn_qkv/ssm_*) + 10 full-attn
  (every 4th from blk.11), 256 experts top-8, 41 blocks incl MTP layer.
- dspark head transfers cleanly: acceptance 0.65-0.93, mean-len 2.6-22.5
  (ngram spine fires on copy exactly like KAT).
- t/s (identical protocol, mmap, W4, t12, dspark+ngram-mod, 3 trials):
  novel med 18.5 (16.1-26.9) | copy med 33-35 (22.5-36.6)
- vs KAT same-protocol mmap: novel 23-28, copy 52-57 -> Ornith = 0.6-0.8x.
- Root cause hypothesis: DeltaNet recurrent-state math on CPU (30/41 layers)
  is far heavier than KAT's full-attention trunk; +21.7 vs 19.9GB traffic.
- Card benchmarks (self-reported, +5-19 delta over Qwen3.6 base) say probable
  quality edge, but NOT a speed candidate on this box. CQ-Ornith build would
  not fix the DeltaNet CPU cost — parking CQ plan unless quality eval demands.


## V112b — CORRECTION (user caught it): arch hypothesis withdrawn, real root cause found
- USER CORRECTION: Qwen3.6 (KAT base) is ALSO hybrid gated-DeltaNet + full-attn.
  Verified: blk.0 shapes identical across both models (attn_qkv 2048x8192,
  ssm dims, 30+10 split). "DeltaNet CPU cost" hypothesis is WRONG — withdrawn.
- REAL finding: official Ornith Q4_K_M has 21/41 ffn_down_exps at Q6_K
  (0.82 B/elem) vs KAT-CQ3 all-Q4_K experts (0.5625) -> ~+8% avg expert
  bus traffic/verify row. imatrix-driven quant rule, not arch.
- Copy-side: acceptance 0.84-0.90 / mean-len 9.5-22.5 vs KAT 0.95 / 21-24
  -> more verify steps per token. Novel-side acceptance actually HIGHER
  (0.65-0.85 vs KAT novel ~0.3-0.5) yet slower — residual gap needs
  split-time instrumentation (KAT_SPLIT_TIME) to close.
- IMPLICATION: speed gap is partly the OFFICIAL QUANT's choice, not the
  trunk. CQ-Ornith (all experts Q4_K, control plane F16/F32 from BF16)
  re-opens as a genuine KAT-speed parity candidate with possible quality
  edge. CQ-ORNITH-PLAN.md resurrected.
- Type-map fix for the record: ggml t8=Q8_0, t14=Q6_K (earlier blk.40 dump
  mislabeled 14 as Q8_0 — sizes unaffected, labels corrected).


## V113 — draft-mtp (native head) on Ornith trunk: loses to dspark, then hangs
- Config: official Q4_K_M (blk.40 nextn head built-in), --spec-type draft-mtp,
  no external draft model, else identical protocol.
- Results before hang: warmup acc 0.196 / tg 10.5; novel p1 acc 0.196 meanlen
  1.58 / 13.9 t/s; novel p2 acc 0.364 meanlen 2.08 / 15.5 t/s.
- vs dspark same prompts: acc 0.65-0.85 / novel med 18.5, copy 33-35.
  draft-mtp is decisively worse on BOTH acceptance and rate — same verdict as
  KAT MTP saga: draft-side MoE reads RAM experts, slow + weak novel drafting.
- STABILITY FINDING: server HUNG mid task 453 (3rd novel prompt) — log frozen,
  process alive but wedged, socket reset, Stop-Process failed (needed taskkill
  /F by PID 44340). draft-mtp path has a hang mode on this build; dspark path
  ran 15+ tasks clean. Not chasing further — dspark is champion either way.
- VERDICT: dspark head remains the draft model for Ornith exactly as for KAT.
  Ornith matrix complete: official-quant trunk + dspark+ngram-mod = the config.


## V114a — FAILED approach + real lesson: GGUF strict offset adjacency
- In-place tensor shrink (Q6_K->Q4_K at same offset) is INVALID: llama.cpp
  gguf_init_from_reader enforces tensor N start == tensor N-1 end. First
  loader error: blk.0.ffn_down_shexp offset 941940736 != expected 872734720.
- Data + type fields were all correct; only layout broke. Fix: orn_compact.py
  — full-file rewrite with recomputed sequential offsets (also reclaims
  1.45GB of dead holes). Output: Ornith15-Q4K-CQ.gguf.
- RULE for future GGUF surgery: in-place TYPE patches only valid when the
  new blob >= old blob size (fill slack with padding) OR sizes equal.
  Shrinking requires full rewrite.
- Also: deleted orn-mtp-head-bf16.bin (1.69GB, re-extractable via
  orn_mtp_extract.py) to fit the compacted output on disk.


## V114b — Compactor BPE bug caught before wasted bench
- First compaction used BPE[14]=0.828125 (212/256) for Q6_K; correct is
  210/256=0.8203125 — ground-truthed against own logged 220200960/268435456.
  43 type-14 tensors remain post-requant, so every offset after the first Q6_K
  would be wrong -> loader adjacency failure again. Caught pre-bench via
  external review; file regenerated in 36s. LESSON: never trust remembered
  block sizes; audit constants against measured tensor bytes on disk.
- Types present in file: only {0: F32, 12: Q4_K, 14: Q6_K}.

## V114 — Ornith all-Q4_K (compacted 20.25GB): faster than official quant
- Config identical to V112 (dspark+ngram-mod, W4, t12, mmap, 3 trials).
- trial1 (coldest common point): novel 20.6-22.3 (V112: 17.5-18.6),
  copy-p1 26.6 (23.4), copy-p2 verbatim 43.5 (33.1).
- Full spread: novel 20.6-40.2 (med ~23.2), copy 26.6-59.5.
- Delta vs byte math: +17-31% measured vs ~6% predicted from down_exps byte
  cut. Two candidate causes: (a) Q4_K dequant cheaper than Q6_K on CPU,
  (b) WARM-CACHE CONFOUND — compacted file was just written; OS cache may
  hold a large fraction. V112 ran on a fresh download. NOT yet disentangled;
  treat V114 numbers as upper bound until a cold re-bench (post-reboot or
  after cache-evicting load) confirms.
- Acceptance unchanged (0.75-0.94) — expected: same weights, fewer bytes.
- NEXT: V115 PPL gate (quality cost of Q6K->Q4K on those 21 layers is
  UNMEASURED — this is the binding question, not speed).

## V114c — full statistical analysis of the A/B
- Medians: novel 18.5->23.2 (+25%), copy-extend 23.4->27.5 (+17%),
  verbatim 35.1->53.5 (+52%).
- Same-trial deltas all positive except one outlier (-0.7). Trial1-to-trial1
  (coldest common point): +17-31% — BEYOND the ~6% byte cut.
- Mechanism hypothesis: Q4_K dequant cheaper than Q6_K on CPU (no ql/qh
  reassembly, simpler scale unpack). On CPU-bound expert reads this compute
  is on the critical path, so it multiplies with the byte reduction.
- Warm-cache confound still possible for the larger trial2/3 deltas; the
  cold-point comparison is the conservative one and still shows +17-31%.
- IMPLICATION FOR CQ MAP: if V115 PPL passes, all-Q4_K experts + our F32/F16
  control plane is BOTH faster and smaller than the official imatrix mix.
  The official quant's Q6_K down_exps choice costs ~20% speed for quality
  on 21/41 layers we may not need.

## V115 (pre-registration) — gate redesign after review
- CAUGHT BEFORE NUMBERS LANDED: original gate (KAT-vs-Ornith delta, +0.5/+1.0
  thresholds) conflated cross-model differences with requant damage, and was
  looser than the campaign's own V98 standard without justification.
- Correct design: Ornith-official-PPL vs Ornith-CQ-PPL (same model, 21
  tensors differ). True baseline requires re-download (in-place patch
  destroyed the original; broken file is unloadable).
- Gates re-aligned to V98: PASS <= +0.05, MARGINAL <= +0.30, FAIL > +0.30.
  KAT same-run PPL retained as context/anchor only.
- Measurement order: KAT PPL (running) -> redownload official -> official
  PPL -> CQ PPL -> verdict. ~2h total, machine dedicated (no disk
  contention during runs).

## V115a — two measurement-integrity failures caught and fixed
1. Garbage-PPL leg: official-REF PPL produced ~1.8M PPL/chunk because the
   file was 34% zeros — I fired the leg after checking FILE SIZE stability,
   but pardl2 preallocates full size at start. My own orchestrator comment
   said 'Never trust size alone' — and my manual check did exactly that.
   RULE (now enforced): a pardl2 target is complete ONLY when the
   .pstate.json sidecar is absent AND no pardl2 process is running.
2. Earlier: wedge + parser. Old orchestrator (pre-fix code in memory) held
   a file lock after its clamp expiry; relaunch hit WinError 32. The PPL
   parser expected 'perplexity =' but this build prints 'PPL ='.
3. Direct-exe bash launch eats backslashes: '-f D:\merge\...' arrived as
   'D:mergeE0...' — use forward slashes for all direct exe calls.
Cost: ~25 min wasted measurement. The KAT anchor (6.9831) is unaffected.

## V116 — disk cleanup (user directive)
- Deleted (probe-verified not in-use via exclusive-open first):
  KAT-CQ2-MTP, KAT-CQ3-MTP (21.3GB, the campaign champion trunk — re-buildable
  from ledger; PPL anchor 6.9831 recorded), kat-mtp-head safetensors, fctrace,
  jetspec head; Ornith-1.0 9B Q4_K_M, its MTP head, 18.8GB BF16 shards,
  engine/research/model dirs + worktrees.
- KEPT: kat-dspark-v2-q8.gguf (draft head for Ornith-1.5 serving too),
  Ornith15-Q4KM-REF.gguf + Ornith15-Q4K-CQ.gguf (V115 gate inputs).
- Freed: C: 29->49.5GB, D: 21.7->42.6GB.
- Also this session: RAM-starvation incident — PPL leg launched while user's
  llama-server held RAM; killed on complaint (18.5GB free after). V115 chain
  now hard-gated on zero llama-server + >=20GB RAM (orn_v115_gated.py).

## V115b — preliminary: partial official leg (8 chunks, verified REF file)
- Killed leg's data is still valid for the chunks it completed.
- Per-chunk: Ornith-official consistently ~+1.0 above KAT (2.41/1.44 ...
  4.19/2.69), parallel tracking, gap widening on later chunks.
- Read: model difference (different training), NOT quant artifact. Makes
  the official-vs-CQ delta the sole open question, as designed.
- Full legs pending on the machine-free gate (bg_9 chain).

## V117 (in progress) — the proper build: BF16-direct quant, no roundtrip
- USER-DRIVEN redesign (right call): V114's 21 Q6_K->Q4_K tensors were
  DOUBLE-quantized (deq Q6 -> quant Q4), stacking error on exactly the
  tensors the imatrix flagged as sensitive. Fix: pull those 21 down_projs
  from BF16 safetensors (packed [256,2048,512] per layer, HTTP range) and
  quantize DIRECT to Q4_K. Everything else byte-copied from REF (imatrix
  Q4_K experts + official control plane + MTP head preserved).
- Layout empirically verified BEFORE build (orn_layout_check.py):
  cos(direct)=0.99746 vs cos(transposed)=0.00027 on L5 expert 0 —
  torch->GGUF expert bytes need NO transpose (matches build_cq3 finding).
- safetensors index: 1811 tensors, experts packed per-layer as
  mlp.experts.{down_proj, gate_up_proj}, BF16, 16 shards.
- Builder: orn_build_dq.py (compact rewrite, ground-truthed BPE map).
  Output: Ornith15-Q4K-DQ.gguf (~20.25GB expected, same as V114 file).
- Serving-impact note: control plane lives in VRAM (-ngl 99 -cmoe), so
  keeping official's Q4/Q6 control plane (vs F16 upgrade) is the
  VRAM-safe choice on 8GB; F16 control plane would risk not fitting.

## V118 — KAT_TOPK_ACCEPT first A/B (user's top-k rescue idea)
- k=1 vs k=2 on pristine official Ornith REF, dspark+ngram W4 protocol,
  back-to-back same-session.
- k=1 trial medians: novel 11.0 (5.6-13.9), copy 13.6 (10.5-34.3) @ 8.6GB free RAM
- k=2 trial medians: novel 14.75 (11.1-19.9), copy 21.2 (17.4-30.1) @ 19.2GB free RAM
- PER-LEG ACCEPTANCE LOGS IDENTICAL (0.8716, 0.6591, ... per prompt, both legs):
  EXPECTED + CONSISTENT — the logged acceptance is the strict-argmax tally;
  the rescue extends chains past near-ties (draft in target top-2 but not
  argmax) instead of inflating that metric.
- CONFOUND: RAM 8.6 vs 19.2GB across legs — tps delta (34% novel / 56% copy
  median) is real-direction but magnitude unconfirmed. Equal-RAM re-run
  pending (post-BF16, box-settled).
- Mechanism plumbing verified: KAT_TOPK_ACCEPT env in gypsy-dragon DLL,
  committed; sampling.cpp accept-loop rescues via cur_p rank walk.

## V119 — Ornith drafter matrix + AR floor (resident, official REF)
- no-spec AR floor: novel 18.3, copy 18.9 t/s (FASTEST AR measured on this
  box; KAT AR baseline was 12.3 — Ornith's gated-DeltaNet hybrid decodes
  faster per-token than KAT despite +8-9% expert bytes).
- MTP (native head): novel 9.8 / copy 11.1 = 0.55x AR — ACTIVELY HARMFUL,
  acceptance 0.20-0.39 makes every draft rejected, verify overhead dominates.
- dspark: novel 19.4 / copy 33.0 (peak 48.9) = 1.0x novel, 1.75x copy.
- ngram-mod: novel 17.2 / copy 18.5 (~0.95x, no benefit).
- WHY SLOWER THAN KAT: NOT the base level (Ornith AR > KAT AR). It's the
  speculation multiplier — KAT got ~5x AR (copy 63-66), Ornith gets ~1.75x.
  Cause: (a) 21/41 ffn_down_exps at Q6_K raise per-verify-row cost (+8-9%
  bytes), (b) V114's Q4_K requant (+25%/+52%) not applied per user pivot.
- MTP on this model confirmed useless; dspark the only paying drafter, copy-only.

## V120 — Expert-bytes tax CONFIRMED as the bottleneck (OPT build)
- Built OPT: official copy + 21 Q6_K ffn_down_exps -> Q4_K (requant from Q6_K),
  compacted to 20.26GB contiguous (in-place shrink alone breaks GGUF contiguity,
  V114a lesson re-confirmed: load fails until compacted).
- dspark+ngram resident bench (official file stays pristine):
    novel 26.0 (was 19.4, +34%)
    copy verbatim 51.7 med / 59.6 peak (was 33-48 peak) -> KAT-LEVEL (52-57)
    copy reproduce+continue 29.8
- VERDICT: Q6_K expert-bytes tax was THE copy-speed bottleneck. Fixing it
  lifts verbatim copy to KAT resident levels. Novel also +34%.
- CAVEAT: OPT double-quantized those 21 tensors (Q6_K->Q4_K). The clean final
  artifact = BF16 full rebuild (step 2): experts Q4_K from true BF16 source +
  control plane F16/F32, no roundtrip. BF16 verified clean (80/80 regions).
- NEXT: BF16 full rebuild (V121), then quality gate vs official.

## V121 — dspark draft-length sweep (nmax 4/6/8) on OPT build
- resident, dspark+ngram, 3 trials each:
    nmax=4: novel 26.0, copy verbatim 51.7 (baseline)
    nmax=6: novel 25.3, copy verbatim 49.0
    nmax=8: novel 21.1, copy verbatim 43.1
- Monotone decline both axes — dspark confidence decays with block position,
  longer drafts add rejected tail tokens that waste verify rows. nmax=4 stays
  optimal (matches V75/V101 width-curve shape from the KAT campaign).
- CORRECTION to earlier statement: KAT novel was ~40 sustained / 54.7 peak
  (W=3 resident soak), NOT 26. Ornith novel 26 is still BELOW KAT novel;
  only copy now matches KAT. Novel gap = draft-acceptance-bound (no ngram
  help on novel text, native MTP head too weak at 0.20-0.39).
