# OPD Design Synthesis — all sources, final recipe
# Compiled from direct reads: TML-OPD, TML-LoRA, SAO/GLM-5.2, GSPO, TOPD,
# SOD, Rethinking-OPD (2604.13016), Unmasking-OPD (2605.10889),
# Offline-TopK (2608.03796), TailKL (2602.20816), OPRD (2606.06021),
# Hindsight-Hints (2605.11556), Open-SWE-Traces (2606.16038),
# Draft-OPD (2605.29343), + GLM-5.x blogs, slime, ReasonMaxxer, Wolfe-CL

## THE SEVEN LAWS (what the papers collectively establish)

1. DENSE beats sparse: OPD's per-token reverse-KL = 50-100x compute
   efficiency vs RL's 1-bit-per-episode [TML-OPD]
2. LATE TOKENS POISON: teacher reward quality degrades with prefix depth;
   entropy collapse starts at trajectory END and propagates backward
   [Rethinking-OPD Fig.13]. => truncate early steps, expand gradually
   [TOPD: rho=25% beats full OPD, 2.9x faster]
3. STEP BOUNDARIES DECIDE: in agents, student-teacher divergence jumps at
   tool-call boundaries (state drift); errors compound super-linearly
   across steps [SOD Props.1-2]. => weight steps by divergence decay,
   not uniformly.
4. FAILING ROLLOUTS ARE THE SIGNAL: distillation helps MOST on rollouts
   the student gets wrong [Unmasking-OPD]. => never discard failures.
5. SAMPLED TOKEN SUFFICES: k=1 (the student's own token) carries the
   gradient; top-64 adds nothing [Rethinking-OPD Fig.15]. => logprobs
   of realized tokens only; no full-dist API needed.
6. RESPONSE-LENGTH SWEET SPOT: 3K-7K tokens optimal; 10K+ degrades
   [Rethinking-OPD Fig.11]. => chunk trajectories; don't force long.
7. COLD START MATTERS: if teacher/student thinking patterns diverge,
   SFT-on-teacher-traces first (off-policy), THEN switch to OPD
   [Rethinking-OPD §5.1]. => our downloaded datasets = cold-start bank.

## FINAL RECIPE (v3 — replaces all priors)

### Stage 0: Cold Start [Law 7]
- SFT on greghavens-GLM5.2 + k3-harmonized + uka traces (already on disk)
  - NO random mixing: filter to bash/edit/read tool patterns only
  - ~6K rows, 1 epoch, LoRA r=4, LR 2e-4 (TML 10x rule)
  - Purpose: align KAT's tool-call format to the trajectory distribution
    the teacher grades; NOT capability transfer

### Stage 1: SOD-TOPD (the main event) [Laws 1-6]
Rollout:
- llama-server: 4 slots, MTP+ngram spec, KV q8_0, ctx 16k
- TOPD rho schedule: start 25% of steps -> 50% -> 100% over iterations
  (validate reward stability before each expansion [Law 2])
- TITO: capture student tokens + logprobs per generation
- Anti-hack guard online (GLM-5.2 rules: curl/github/eval-artifact)

Grading:
- Teacher API call on TRUNCATED trajectory prefix (2K tokens not 8K
  = 4x cheaper [TOPD + Law 6])
- Student's realized-token logprob from rollout (Law 5: k=1 suffices)
- advantage_t = -(logprob_student_t - logprob_teacher_t) per token

Step-wise reweighting (SOD Eq.7):
- Segment trajectory at tool-call boundaries
- d_k = divergence score per step (from OPD forward pass, zero cost)
- w_1 = 1; w_k = min(w_1 * prod(d_u + eps)/(d_u+1 + eps), 1 + delta)
  delta=0.2, eps=1e-6
- Loss: L = sum_k w_k * sum_{t in step k} rKL_t * mask_t
  mask = model-generated only (tool outputs = 0)

Training update:
- LoRA r=4, hot experts (per-layer visit threshold from probe)
- Liger fused chunked gather (248k vocab, memory-linear [2608.03796])
- TIS double-sided clip [0.2, 4.0] on importance ratio
  (bridges Q4_K rollout <-> NF4 training quantization gap)
- LR 2e-4 constant, effective batch 32
- Full-precision AdamW on adapter (states tiny)

Iteration:
- 12 trajectories/batch (4 slots x 3 rounds)
- docker verification OVERLAPPED (background thread)
- Keep ALL trajectories for training [Law 4: failures = most signal;
  reward r in {-1, +1} multiplies the KL advantage:
  L_final = r * w_k * rKL_t  — negative reward REVERSES the pull]

### Stage 2: RL polish (only if Stage 1 plateaus below target)
- SAO single-rollout, docker-verified binary reward
- Same TIS/Liger/LoRA infrastructure
- This is where "beyond teacher" could happen [2602.12125 extrapolation]

## VALIDATED-BY-SOURCE TABLE (every design element -> paper)

| Element | Source | Number |
|---------|--------|--------|
| OPD dense signal | TML-OPD | 50-100x vs RL |
| TOPD 25% truncation | 2605.31490 | 51.3 vs 49.3 AIME, 2.9x faster |
| Step reweighting | SOD 2605.07725 Eq.7 | w_k formula, zero cost |
| Failures matter | Unmasking-OPD 2605.10889 | "helps most on failing rollouts" |
| k=1 sufficient | Rethinking-OPD Fig.15 | k=1 == k=64 |
| Length sweet spot | Rethinking-OPD Fig.11 | 3-7K optimal |
| Cold start | Rethinking-OPD 5.1 | OPD ineffective without it |
| Offline top-K cache | 2608.03796 | 29% faster, 41% throughput |
| Chunked KL linear-mem | 2608.03796 | memory ~ seq_len |
| Tail-aware KL | 2602.20816 | same cost, better tail signal |
| LoRA RL capacity | TML-LoRA | rank-1 == full FT for RL |
| LoRA on MoE layers | TML-LoRA | attn-only underperforms |
| LoRA LR = 10x | TML-LoRA | 10-15x FullFT LR |
| TIS clip [1-e_l,1+e_h] | SAO/GLM-5.2 | eps 0.8/3.0 SWE tasks |
| Anti-hack guard | GLM-5.2 blog | rule filter + LLM judge |
| Verifier hygiene | GLM-5.3 blog | oracle/no-op/unsolved checks |
| Routing-based LoRA scope | TML-LoRA + UMoE | visit-threshold per layer |

## OPEN ITEMS (ranked by impact)
1. DashScope logprobs API check (teacher grading path)
2. Cold-start SFT data formatting (traces on disk -> messages)
3. Probe run: difficulty + routing histograms (GPU, needs server up)
4. SOD d_k proxy: exact divergence formula from their D.5 appendix
5. Draft-OPD: train MTP head for faster rollouts (after main loop works)

## V4 AMENDMENTS (BeyondTeacher scout — 7 papers, verified)

### DROP-IN 1: SRPO Sample Routing (2604.02288) — replaces flat failure-inclusive loss
- Correct rollouts -> GRPO branch (sequence-level advantage)
- Failed rollouts -> SDPO branch (dense per-token KL)
- Entropy-aware reweighting: w_{i,t} = exp(-beta * H_teacher_t), beta=1
- Source numbers: +3.4pp over GRPO, +6.3pp over pure SDPO; 17.2% lower step compute

### DROP-IN 2: PACED Beta(p) instance weighting (2603.11178)
- After probe: weight each gym instance w(p) = p(1-p)
- Drop p<0.05 (incoherent gradients) and p>0.95 (refinement noise)
- 49% of typical problem sets are wasted by uniform weighting
- Source numbers: up to +8.2pp AIME over unweighted

### DROP-IN 3: OPRD hidden-state loss (2606.06021) — additive objective
- Normalized MSE on hidden states at layers {4,8,16,24,32}, teacher detached
- Loss path memory: [B,T,d=2048] vs [B,T,248320] logits = 59x smaller
- Source numbers: +2.7pp, 1.44x faster, 54% peak-VRAM cut
- Composes with output-space loss at zero infra cost
- NOTE: needs teacher hidden states => only for local-teacher tiers (OPSD/SDFT/Qwen3.6-35B), NOT API teacher

### KNOB: ExOPD reward scaling (2602.12125)
- advantage *= alpha_R; try {1.5, 2, 3} to exceed teacher ceiling
- especially for multi-domain merge passes later

### KL SCHEDULE CORRECTION (PACED):
- Phase A: FORWARD KL (teacher much stronger — our initial state)
- Phase B: REVERSE KL (once student approaches teacher)
- not fixed reverse-KL from day 1 as v3 implied

### SAMPLING: T_train=2.0, top-k=10 for generation (SSD 2604.01193)
- precision-exploration conflict: locks need precision, forks need exploration
- T_eff = T_train * T_eval ~= 1.2 sweet spot (R^2=0.75)
- Apple numbers: +12.9pp LiveCodeBench on Qwen3-30B-Instruct, teacher-free
- apply to cold-start data generation AND rollout exploration passes

### TEACHER-INDEPENDENCE LADDER (fallback tiers)
- Tier 0 (no API): SSD — sample own rollouts at T=2.0/topk=10, SFT. +5-10pp expected
- Tier 1 (gold traces local): OPSD/SDFT — self as teacher conditioned on y*
  (gold patch + verified trace); EMA decay 0.99; forward KL w/ per-token clip
- Tier 2 (full API): Qwen3.8-Max + SRPO routing [+ OPRD only w/ local teacher]

## V5 AMENDMENTS (Agentic-Deep + XTok-Ceil scouts — verified against papers)

### CRITICAL FIX 1: difficulty routing = logprob, NOT trained probe [2605.02241]
- Supervised probes COLLAPSE out-of-distribution (AUROC 0.662 -> 0.546)
- avg token logprob: 0.681-0.782 AUROC, never beaten by probes at any N
- ACTION: probe phase computes pass-rate p (for PACED weighting) via
  rollouts, but ROUTING decisions use rollout logprob only. Delete any
  learned-probe plans.

### CRITICAL FIX 2: SFT_MaxOOD handoff point [2509.12235]
- OOD performance PEAKS EARLY in SFT then declines while ID loss keeps
  improving; RL afterward recovers only ~50-70% of the OOD peak, never
  exceeds it
- ACTION: during cold-start SFT, eval held-out OOD probe every ~100 steps;
  checkpoint the peak; start OPD from THERE, not from SFT convergence
- RL guard: only run RL/OPD when advantage entropy > 2.55 AND positive-
  reward ratio in 40-80% band (kill/warm-restart otherwise)

### CRITICAL FIX 3: SOD exact formulas (we had them incomplete) [2605.07725]
- d_k (Eq.6) = (1/|I_k|) sum_{t in step k} |logpi_student - logpi_teacher|
  -> zero marginal cost, computed from existing grading call
- w_k (Eq.7) = min(w_1 * PROD_{u<k} (d_u+eps)/(d_{u+1}+eps), 1+delta)
  -> RATIO PRODUCT, not simple cap; delta=0.2, eps=1e-6
- Ablation: uniform w_k = 34.70 vs adaptive = 42.98 (+8.3pp!)
- SOD Obs.5: LARGER teacher gap makes uniform OPD WORSE — our Qwen3.8-Max
  -> KAT gap REQUIRES adaptive reweighting; it is not optional

### ADD: Open-SWE-Traces to cold-start bank [2606.16038]
- 207K trajectories, 65K resolved, 9 languages, permissive license
- Same base family as ours (Qwen3-30B-A3B), lifts base to 61.7 SWE-V
- Their ablations confirm: full corpus (with failures) > resolved-only
  (+5-7pp); multilingual data helps Python-only eval (+3.2pp)
- ACTION: pull python+go subsets, mix into Stage 0 at ~30-40%

### ADD: HHD hint loop as cheap cold-start generator [2605.11556]
- Failed rollout -> rule-compress -> one expert call for 2-3 sentence hint
  -> student regenerates ON-POLICY with hint -> SFT on scaffolded success
- 5-10x cheaper than per-token teacher grading; +8pp over RFT baselines
- Hints at START not middle; keep to 2-3 sentences (leakage otherwise)
- ACTION: implement as Tier-1.5 fallback (between SSD and full teacher OPD)

### ADD: Mach-Mind engineering tricks [2607.09375]
- XML tool-call template instead of JSON (escaping brittleness, free)
- Difficulty pruning: keep pass@8 in [0.1, 0.9]
- Error masking: rule-based, zero loss on erroneous action tokens (+3pp)
- Two-stage reward curriculum: dense process -> strict outcome
- HMPO token compression: R_final = R_acc * min(1, cos(pi*n/2b)+lambda);
  19-46% shorter outputs at <=0.7pp loss; our TPS-bound box wants this
- CLIP QUESTION: Mach-Mind uses [0.20, 0.28] vs our [0.2, 4.0] (GLM SWE
  values) -> ablate; tighter may suit smaller compute

### ADD: DomainPilot adaptive mixture [2607.22769]
- SWE-domain alpha ~0.09-0.10 (slow learner, boost +15%); chat alpha 0.55
  (cut -8%); fit scaling law from 200-step warmup, reweight every 1k steps
- ACTION: cold-start mixture is adaptive, not fixed

## FINAL STAGE STACK (v5)
Stage 0: cold-start SFT (Open-SWE-Traces + GLM5.2/K3 traces + SSD-style
         self-rollouts at T=2.0/topk=10), checkpoint at SFT_MaxOOD peak
Stage 1: SOD-TOPD w/ adaptive w_k ratio product + SRPO routing
         (failures->SDPO dense, correct->GRPO sparse) + PACED Beta(p)
         weighting + logprob routing + forward-KL-first schedule
Stage 2: RL polish w/ docker rewards, advantage-entropy guard,
         two-stage dense->outcome curriculum, HMPO length compression
Loop:    anti-hack online guard, verifier hygiene, OPRD additive loss
         (local-teacher tiers only), ExOPD alpha_R knob for merges

## V6 FINAL AMENDMENTS (OPD-Mech scout — hyperparams verified from 5 papers)

### CONTRADICTION FIXES (from flagged C1-C8)

C1 FIXED: k selection is task-dependent
- k=1 (my Law 5) was WRONG for stability: k=1 has unstable overlap growth
- math/reasoning: k=4 (Rethinking sweet spot, k>=4 statistically tied)
- AGENTIC (ours): k=32 renormalized (Revisiting: +19.8% over sampled-token)
- top-p=0.9 on rollouts REQUIRED for top-K to work (21.6->23.6 AIME alone)

C2 FIXED: max response length = 8K not 16K
- 3-7K optimal (Rethinking Fig.11), 10K+ degrades
- TOPD rho=25% from START confirmed as the mitigation

C5/C8 FIXED: Uni-OPD margin machinery added
- G_OPD (eq.5) = mean per-token log(pi_T/pi_theta) — our headline training metric
- m(q) (eq.9) = min_{correct} G_OPD - max_{incorrect} G_OPD per prompt
- ALARM: >30% prompts with m(q)<0 -> teacher signal unreliable ->
  apply Margin Shift (eq.11): lambda(q) = delta - m(q), add to correct trajs
- validate failure-sign flip doesn't add noise (order-consistency check)

### NEW: pre-flight diagnostic (RUN BEFORE ANY TRAINING)
- Stage A: 50 rollouts -> overlap ratio top-k {4,16,64} + entropy gap
  GATE: median overlap < 50% -> do cold-start SFT first. Cost: ~$10-20 API
- Stage B: G_OPD + margin m(q) on same 50
  GATE: >30% negative margin -> Margin Shift or skip teacher
- Stage C (optional): gradient alignment score cos(g_ideal, g_OPD)
  GATE: failing-vs-correct alignment should differ 3-5x
- Total: ~$10-20 + 1 GPU-hour vs 30-50 GPU-hrs blind

### NEW: Veto warmup (ACL 2026.findings-acl.2094)
- First 25% of training: Q(y|x) ~ exp(z_T + beta*z_S), beta=0.8 (reasoning)
  / beta=1.0 (code), linear decay to 0 -> degrades to standard reverse-KL
- Prevents forward-KL gradient explosion on "ignorant" tokens early

### HYPERPARAMETER TABLE (verified, our defaults)

| Param | Value | Source |
|-------|-------|--------|
| LR | 1e-6 base / 2e-4 LoRA (10x rule) | Rethinking / TML |
| Batch | 64 effective (grad accum) | Rethinking |
| Rollout n | 8 per prompt | all papers |
| Max response | 8192 | Rethinking Fig.11 |
| Temperature (rollout) | 1.0 | Rethinking |
| Top-p (rollout) | 0.9 | Revisiting (required) |
| Teacher top-K | 32 agentic / 4 math | Revisiting / Rethinking |
| KL coefficient | 0.0 (DELIBERATE — never add) | Rethinking |
| Loss aggregation | token-mean | Rethinking |
| LoRA rank | START r=4, sweep {4, 8, 16} if plateau | TML + caution: OPD denser than RL |
| Veto beta | 0.8 (reasoning) / 1.0 (code), linear decay 25% | ACL |
| Epochs | 1 | Rethinking |

### CAPABILITY GAP ANALYSIS (go/no-go for Qwen3.8-Max -> KAT)
- Our capability gap: 1.27x (KAT 55 -> 3.8 ~62-70 SWE-V)
  = BELOW every successful case's minimum (1.4x was a FAILURE case;
  successes: 2.4x, 3.7x, 6.5x)
- Our param gap: 19.6x — between the 4.7x failure and 23x success
- Same-family (both Qwen3.6-derived): strongest positive signal
- PREDICTION: strongly in success regime; cold-start still required

### METRICS TO LOG (per iteration, all free from existing forward passes)
1. overlap ratio (target: 72% -> 91% over training)
2. entropy gap H(pi_T) - H(pi_theta) (target: narrows monotonically)
3. G_OPD per trajectory (Uni-OPD eq.5)
4. margin m(q) histogram + negative-margin fraction (alarm: >30%)
5. d_k per step (SOD Eq.6) — already computed, log it

## V7 REVISION: TEACHER-FREE (user decision)
Drops Qwen3.8-Max API entirely. Self-distillation + RL lineage.

### WHY VIABLE
- Gym carries pr_patch (gold) per instance -> privileged conditioning (OPSD)
- SSD: +12.9pp LCB teacher-free (Apple 2604.01193)
- OPSD: matches GRPO efficiency (2601.18734); EMA variant = SDFT (2601.19897)
- Stage 2 RL-on-docker-rewards IS the Qwen3.8 recipe (verifiable rewards)
- No API quota risk, no grading OOD, both passes local + MTP-fast
- Same-model teacher == same quantization == exact importance ratios
  (TIS becomes pure stability guard, not quant-bridge)

### REVISED STAGES
Stage 0: SSD bootstrap (own rollouts T=2.0/topk=10, SFT, checkpoint at
         SFT_MaxOOD peak) [+ Open-SWE-Traces mixed in, still teacher-free
         since those are static traces]
Stage 1: OPSD — teacher = EMA(self) conditioned on (problem + pr_patch),
         student = self on (problem only); TOPD rho 25%; SOD d_k/w_k with
         privileged-self logprobs; SRPO routing; PACED weighting;
         OPRD hidden-state loss now projector-free (same arch both sides)
Stage 2: RL polish — docker FAIL_TO_PASS rewards, Mach-Mind config
         (clip [0.2,0.28], KL off, pass@8 [0.1,0.9] pruning, error masking,
         HMPO R_acc*R_token compression)

### PRE-FLIGHT (simplified, free)
- 50 rollouts: overlap = student vs gold-conditioned-self top-k agreement
- margin m(q) on G_OPD(gold-conditioned) — same formulas
- zero API cost; runs in one GPU-hour

### TEACHER LADDER (if ever needed later)
Tier 0 SSD -> Tier 1 OPSD/SDFT (current) -> Tier 1.5 HHD hints (cheap
expert, 2-3 sentences) -> Tier 2 big-teacher OPD (parked)

## V8 REVISION: GLM-5.3-STYLE SMART STACK (user direction)
"Massive gains across the board" via stacked cheap levers, not brute
training. GLM-5.x blogs demonstrated the pattern; we measured the same
20pp harness sensitivity on identical weights.

### THE THREE LAYERS (do in order — A changes C's data distribution)

LAYER A: scaffold/decode mining — ONE GPU SESSION, comes first
  Sweep on 20-instance subset, validate on 50:
  - presence_penalty grid (A1/GLM finding; incl. small negatives)
  - temperature x top-p x top-k for AGENTIC decode (not chat defaults)
  - XML tool-call format vs JSON (Mach-Mind: XML wins on escaping)
  - system template variants (2-3 candidates)
  - MTP draft params (spec-draft-p-min, n-max)
  Gate: pick argmax on subset, confirm on 50, FREEZE before training.
  Expected: +4-8pp pass@1. Zero permanent cost.

LAYER B: best-of-N + docker oracle — FREE, we own the infra
  We have per-instance FAIL_TO_PASS oracles (the gym). At inference:
  generate 8 candidates under frozen Layer-A config, run tests, ship
  the passing patch. pass@1 -> pass@8 = +12-18pp typical.
  This is deployment-time compute scaling — the Qwen/GLM agent
  harnesses all do this; nobody reports it in the leaderboard number.
  NOTE: for real tasks WITHOUT tests, fall back to a verification
  prompt (self-check) — weaker but nonzero.

LAYER C: v7 teacher-free training — now on the raised floor
  Unchanged recipe (SSD -> OPSD w/ gold conditioning -> RL polish).
  Trains UNDER Layer-A scaffold so rollouts match deployment.
  Each pass@1 gain compounds with B's best-of-8.

### REVISED EXPECTATIONS (all three layers)
  SWE-V:      69.4 -> ~88 effective (3.8 ~75)     CLEAR WIN
  SWE-Pro:    46.0 -> ~68 effective (3.8 61.7)    WIN
  Terminal-B: 41.0 -> ~60 effective (3.8 73.0)    gap 32->13pp
  pass@1-only honest numbers: 76 / 54 / 46

### HONESTY NOTES
  - B's oracle uplift assumes tests exist at deployment. For test-less
    repos it degrades to self-verification (~half the uplift).
  - The ~88 includes test-time compute; the WEIGHTS are ~76. Both are
    legitimate (frontier labs do the same) but they answer different
    questions.
  - Terminal-Bench gap is long-horizon persistence — still RL-scale
    limited. A+B+C narrows but does not close it.

## V9 REVISION: REASONMAXXER-FIRST (arXiv 2605.06241, verified full read)

### THE PAPER'S CLAIMS (all verified against text)
- RL's useful effect = 1-3% of tokens, at 5-12x-higher-entropy decision
  points, promoting base-top-5 (mean rank 2.14-2.39). Zero novel tokens.
- Entropy gating alone replaces the teacher for locating sites (tau=1.4
  @5.2% tokens optimal; tau=1.8 @2.6% second peak = RL's own rate).
- Full correction = rank-32 QKVO LoRA, 0.27-0.49% params. FOOTNOTE:
  rank-8 OUTPUT-PROJECTION-ONLY adapter is within a few points.
- Method: 20 rollouts/problem from frozen base -> keep 0<p<1 problems ->
  A_i=(r_i-rbar)/(sigma+eps) -> L_dec = -sum_{t in D} A_i log p_theta
  + lambda * L_anchor (KL to base off-decision-points). 1 epoch.
- Results: matches/beats GRPO,PPO,DeepScaleR,STILL-3,PRIME,General-
  Reasoner at $4-25 vs $200-$103,000. Qwen3-4B: 0.476 vs GenRL 0.406.
- ABLATION: positive-only = 0.398 vs full 0.502 (MATH-500): the
  negative/contrastive term carries ~half the gain.

### OUR TRANSLATION (SWE / agentic)
- problems = gym instances; r_i = docker FAIL_TO_PASS binary
- decision points: expect tool-call boundaries + file-choice forks;
  entropy histogram BEFORE training to place tau (agentic != math tau)
- adapter: rank-32 QKVO on CONTROL PLANE (rung 2!) = fits 8GB box in
  the KAT-CQ1 training layout; escalate to rank-8-O-only if it works
- rollout budget: 100-150 instances x 8 rollouts (their ablation says
  window width non-critical; 50 problems sufficed for them)
- keep full trajectories (reward needs the patch) -> no TOPD truncation
- docker verify overlapped with generation (needs reboot)

### REVISED STACK ORDER
Layer A: scaffold mining (unchanged, first)
Layer B: best-of-8 oracle (unchanged, free)
Layer C1: REASONMAXXER (NEW FIRST TRAINER) - one overnight session:
          ~5-7h rollouts + 1-2h control-plane LoRA train + eval
Layer C2: v7 OPSD (escalation only if C1 plateaus) - unchanged design
Layer  D: nothing else. C2 is the fallback ladder, not the plan.

### EXPECTED GAIN (C1 alone)
- paper: matches full RL from base. KAT already RL'd -> discount.
- realistic: +2-6pp SWE-V pass@1 for one session. Compounds with B.

### RISKS (flagged, not hidden)
- math-only paper; agentic transfer unproven (SOD supports step-level)
- second pass on RL'd model: headroom smaller, mechanism still applies
- tau placement must be re-derived from OUR entropy histogram

## V11 ADDENDUM: V4-FLASH VIA COLIBRI — SURGERY + TRAINED FLEET + LUTs
(user direction: hack colibri, graft organs, force fast shape; LUTs first-class)

### BASE FACTS (verified)
- colibri (JustVugg, 25k stars, Apache-2.0, pure C): V4-Flash engine runs
  TODAY, CPU-only, official checkpoint unconverted (experts native MXFP4
  QAT — better than Q4_K requant; dense fp8-e4m3)
- deepseek_v4_dspark.inc = 1,245 lines COMPLETE staged code, zero TODOs —
  DSpark speculative decoding is wiring work, not training work
- Requirements: 167 GB disk / 16-22 GB RAM — box qualifies (31 GB)
- Our hardware: C: hynix PC801 PCIe5 (~10-12 GB/s) + D: SN770 (~5 GB/s)
  = ~15 GB/s dual-SSD striping (colibri native feature, hash-weighted)
- Stock speed est: ~1 tps cold, 2-4 warm; measured ladder reference
  (GLM-5.2): 1.07 tps laptop-class, 5.8-6.8 on 6x5090 full-resident
- CPU-only compute ceiling ~7-8 tps (13.8B active x 2.5 DSpark tokens);
  CUDA expert backend (backend_cuda.cu exists) moves compute to 3070 Ti,
  ~4ms/forward -> disk/cache becomes sole bottleneck

### THREE LUT CLASSES (first-class citizens)
1. ROUTING-REPLAY LUT: dict[hash(last-k tokens, layer)] -> expert-set.
   Built from teacher-forced traces + .coli_usage. EXACT on repeated code
   n-grams (100% prefetch accuracy, zero training). Advisory only —
   never changes router semantics (colibri hard rule, ours too).
2. COMPUTE LUTs (LUT-GEMM): MXFP4 + UE8M0 pow2 scales => 16-entry
   dequant table per block, inner loop = gather+add. ~2x scalar int4 CPU
   path (AMD LUT-GEMM precedent). CPU-path insurance if CUDA graft
   stalls; moot on GPU (DP4A/mma already saturate).
3. STATISTICAL LUTs: CCT P(expert@L+1|expert@L) [ST-MoE] + affinity
   A[ei,ej] [SpecMoE] — the survey's tables, now named as LUTs.

### TRAINED FLEET (small models, big model untouched)
- Pre-gate predictor 43x(4096->256) ~45M params: next-layer top-6 from
  hidden state; covers novel tokens where replay LUT misses. Forward-only
  training on routing traces. (Pre-gated MoE pattern, ISCA'24)
- Expert-importance ranker: gates VRAM-pin/RAM/SSD/Q2-preview tier per
  expert from usage traces (replaces static LRU)
- Drop-budgeter (MoE-Spec): per-layer gating-mass threshold; skips
  long-tail experts; bounded by quality gate
- Router-concentration LoRA (ReMoE losses): LAST, hardest-gated —
  only intervention that alters the model

### HYBRID ORACLE ORDER
replay LUT (exact, repeats) -> CCT LUT (statistical) -> pre-gate (novel)
-> pilot prefetch (colibri, 71.6% 1-layer) -> demand read. Union-batched
per DSpark draft set; MoE-Spec budget caps the union.

### QUALITY GATES (hard, per stage)
- teacher-forced NLL on fixed coding corpus: regression <= 2-3% vs FP8
- 20 answer-checked tasks: pass-rate >= 90% of baseline
- router/drop layers REVERT on failure; expert weights never touched

### SPEED LADDER (this box, V4-Flash)
stock 1 -> striping+pins 2-4 -> +CUDA graft 4-6 -> +DSpark 6-9 ->
+LUTs 8-12 -> +pre-gate 10-14 -> +Q2preview+budgeter 13-18 ->
+router-LoRA 18-22. Target 15 realistic, 20 stretch.

### SEQUENCE
1. free ~45GB on C:, download 167GB to C: (PC801), partial mirror D:
2. clone colibri, build V4 engine + CUDA on Windows, coli tune + iobench
3. DSpark graft (staged .inc) -> measure acceptance on/off
4. trace collector -> LUTs first (free), then pre-gate training
5. Q2-preview + drop-budgeter (gated)
6. router-LoRA last
NOTE: expert-store/prefetch/LUT code ports directly to KAT-CQ1 100-tps
goal in llama.cpp — dual-use, nothing throwaway.

## V12 ADDENDUM: CONSTRAINED-RESIDENT RUNTIME (user architecture)
Core: never touch weights the token doesn't need; force GPU-resident
experts; compensate + verify at runtime. Quality floor enforced LIVE.

### A. CONSTRAINED ROUTING + COMPENSATION
- runtime k-mass budget: keep experts until cum mass >= theta(layer,
  domain) — learned, not static k (MoE-Spec + HookMoE dynamic)
- overhang compensation: per-layer low-rank delta (~1-2M params) distills
  displaced-expert contribution; constrained output -> full output
- escape hatch: displaced mass > cap -> demand-read full path for that
  token (floor holds by construction)
- nothing deleted; full expert set stays on disk; per-token reversible

### B. WEIGHT MAP -> FINE-GRAINED ACCESS (Turbo-Sparse-grounded)
- two-phase expert fetch: gate+up first (2/3 bytes) -> intermediate
  activations reveal needed rows -> fetch only hot down_proj row-blocks
  (1/3 bytes, ~20-30% of rows; FFN intermediate ~90% sparse)
- 16-row-block heatmaps from traces: hot blocks VRAM-pinned even when
  parent expert is SSD-tier
- compounding: expert gating x row gating x bit-width ~ 10x bytes cut

### C. APPROXIMATE TIERS (each error-bounded)
- bit: MXFP4 hot / Q2 cold blocks (bounded by contribution mass)
- rank: low-rank sketch experts answer now; full expert corrects later
  (speculative execution for weights; DSpark acceptance logic one level
  down)
- skip: learned per-layer exit predictor (10-20% of layer-tokens
  negligible delta; early-exit lit)

### D. RUNTIME HEALTH + CORRECTION (novel combination)
- token gate: extend DSpark verify — disagreement on low-entropy
  position -> exact replay; high-entropy forks always full path
  (ReasonMaxxer decision-point insight inverted: compute only at forks)
- drift monitor: hidden-state norms vs calibration bands -> span replay
- learned corrector: distill approximate->exact delta from replay
  events; system converges to fewer replays
- heartbeat: full-path forward every N tokens; divergence feeds
  dashboard — 90-95% floor becomes measured runtime stat

### E. ECONOMICS (15 GB/s striped + VRAM hot set)
full 3.1 GB/tok -> A ~1.4 -> A+B ~0.6 -> A+B+C 0.35-0.6 GB/tok
=> 25-40 tps sustained at 90-95% verified quality. Overhead: replays
+10-20%.

### F. BUILD ORDER (instrumentation first)
1. trace everything: routing, mass dist, row-block heatmaps, byte flows
   (.coli_usage + our hooks) — BEFORE any constraint
2. calibrate theta per layer from mass-vs-delta curves (offline, exact)
3. compensation training (distill constrained->full on traces)
4. two-phase fetch (engine surgery, biggest single win)
5. sketches + skip predictor (gated)
6. runtime verifier loop (token gate -> drift -> heartbeat)
GATE: every stage A/B'-checks vs exact forward; revert on NLL +2-3%

## V13 ADDENDUM: TUNED QWEN3.5-0.8B — DRAFTER/PREFILL-SCORER FLEET MODEL
(verified config: 24L, hidden 1024, 8H/2KV, vocab 248,320 = KAT-EXACT,
hybrid linear-attention (18:6 linear:full) => fixed-size state, no O(S^2)
in drafter, 262K ctx, ~1.6GB BF16 RAM-resident, no VRAM competition)

### JOB 1: spec-decode drafter for KAT-CQ1 (same tokenizer)
- replaces/augments dFlash MTP (1-token) with 3-5 token drafts
- tune = KD on existing 7,928 SWE trajectories (on disk) -> acceptance up
- feeds 100-tps push directly

### JOB 2: speculative-prefill scorer for KAT
- PFlash measured on OUR target family (Qwen3.6-27B): 24.8s vs 257s
  TTFT @128K = 10.4x, NIAH preserved, replicated, MIT (Luce-Org/lucebox-hub,
  C++/CUDA) + DFlash+DDTree decode on compressed KV
- agent workloads 16-32K ctx -> even 4-6x compounds over thousands of
  gym rollouts
- quality gated like everything else (NLL + task floor per stage)

### JOB 3: cross-family prefill scorer for V4-Flash (vocab mismatch)
- ALM character-index aligned chunks (arXiv 2503.20083, NeurIPS'25,
  tokenkit; from XTok-Ceil scout) maps 0.8B span-importance onto V4's
  129,280-vocab positions = Cross-Family Speculative Prefill recipe

### TUNING = two distillation targets on one model
a) KAT-trace KD (draft acceptance)
b) span-importance calibration: train scorer where TARGET attended
   (importance = target-relevant, not drafter-relevant)

### RUNTIME
link libggml*.a, never libllama — custom compute graph: drafter scoring
interleaved with target chunk prefill, BSA block-sparse forward, importance
writes. Build tree already links ggml from codec work.

### REFS
Speculative Prefill 2502.02789; Cross-Family SP (SambaNova ICLR'26);
FlashPrefill (block-sparse drafter, 2026); mit-han-lab BSA (FA2-derived
sm_80+ sparse fwd); PFlash (Luce-Org/lucebox-hub).

## V14 REVISION: LUCEBOX AS CHASSIS (replaces colibri-primary plan)
Verified from 4 lucebox.com engineering posts (June-Aug 2026) + PRs.

### WHAT SHIPPED (MIT, C++/CUDA, libggml-based dflash_server)
- SPARK (qwen35moe = KAT family): calibrated expert pinning learned from
  live traffic (.spark.csv self-tuning), bounded LRU cache ring, async
  pinned-memory swaps; cold-hit 36%->7%; Qwen3.6-35B-A3B = 13.3GiB /
  100 tok/s on 3090 = 92% of all-GPU. ONE FLAG (--spark).
- KVFLASH: fixed GPU KV pool, 64-token chunk paging to RAM bit-exact,
  drafter-scored residency (FlashMemory arXiv 2606.09079 pattern, no
  training); 256K ctx in 72MiB GPU KV, flat 38.6 tok/s vs 13.1 full;
  qwen35moe supported incl. Spark hybrid (101.6 tok/s composed).
- TOOL-PREFIX-CACHE (PR #492): stable tool-prefix snapshot (KV+recurrent
  +conv state+token seed for Qwen3.5/3.6 hybrid-linear), prefill only
  tail: 1.04s vs 50.35s = 48x warm; Hermes loop 2.95x end-to-end.
- V4-FLASH-0731 (PR #593 + Geometric PR #28): learned mixed-precision
  codebook quant (2.5bpw gate/up experts, 3.5bpw down, 4.25bpw attn/
  dense/shared; chosen from measured layer-output damage under real MoE
  routing) = 2.766bpw total BEATS 2.88bpw reference on ds4-eval-92
  (82/92). CUDA+HIP kernels shipped. DSpark GGUF shipped (82.4%
  acceptance -> 27.9 tok/s; ~100% -> 32.7; sparse prefill 173 tok/s
  @60K) on Strix Halo 128GB.

### PIVOT
- KAT-CQ1 path: deploy lucebox (Windows CUDA build) + --spark +
  --kvflash + prefix cache + DFlash => est 40-70 tok/s on 3070Ti 8GB
  BEFORE any custom machinery; then v10-v12 layers (LUTs, pre-gate,
  two-phase row fetch, budgeter) stack ON Spark's calibrated cache.
- V4-FLASH path: CUDA kernels + DSpark + quant recipe all shipped; OUR
  contribution = colibri-style SSD tier graft (dual-SSD 15GB/s striped
  + partial residency + LUT oracle) since 98GB >> 31GB RAM.
- colibri demoted to reference (SSD tier, expert atlas, dual-drive).
- llama.cpp remains for PPL-gate tooling + GGUF work.

### KAT-CQ1 CONFIG TARGET
Q6_K control plane (1.87GB) + ~4GB hot experts (~26% residency) +
RAM-resident rest + PCIe swap ring + KVFlash pool + prefix cache +
DFlash spec. Quality gates unchanged (PPL + SWE subset per stage).

### NEXT ACTIONS (revised)
1. clone lucebox-hub, Windows CUDA build for 3070 Ti
2. verify KAT-CQ1.gguf loads in qwen35moe backend (mixed-dtype tensors)
3. smoke: --spark + --kvflash + prefix cache; measure vs llama.cpp
4. disk surgery for V4 (167GB+ weights) -> lucebox mixed-quant convert
   (their recipe, our GGUF builder) -> SSD-tier graft begins

## V15 ADDENDUM: IFP-GRAFT + WINDOWED DRAFT + MESO RULES
(four sources: Apple AFM3 post, IFP ICML'25 arXiv 2501.02086,
 Windowed-MTP arXiv 2607.21535, MESO WCCI'26)

### 1. IFP = PER-PROMPT EXPERT SELECTION (replaces per-token SSD streaming as V4 primary)
- Apple ships it: 20B in NAND, dense block selects FIXED expert set per
  prompt, patched to DRAM w/ always-on shared experts, decode = small
  dense model; periodic reselection during generation
- IFP paper: instruction-conditioned mask predictor over FFN rows/cols;
  3B-active beats 3B dense +5-8pt, rivals 9B
- V4 math: top-20/256 per layer = 12.2GB -> 0.8s load from dual-SSD
  (amortized) -> decode 6-active/token from RAM = ~15 tok/s CPU-only
  baseline, before GPU residency
- OUR 0.8B BECOMES THE MASK PREDICTOR (Apple's predictor = same role);
  trained on routing traces w/ target-attended supervision (v13 already
  specs this)
- escape hatch = BATCH RESELECT on displaced-mass monitor (not demand
  read; keeps SSD off the per-token path)
- validation of v12: user's constrained-resident design == Apple prod

### 2. WINDOWED DRAFT (2607.21535)
- draft full-attn KV read dominates at long ctx; deep drafts NET-
  NEGATIVE; worst on hybrid/linear targets (ours!)
- fix: sliding window + sink on DRAFT attention only; lossless (target
  verifies); +28-44% on Qwen GDN-MoE 35B family
- apply to: tuned 0.8B (6 full-attn layers) AND dFlash MTP in lucebox
- reclaim draft KV via ring buffer (7.7-11% of total KV)

### 3. MESO RULES (WCCI'26)
- QUANT TARGET FOR BANDWIDTH, KEEP DRAFT PRECISE FOR ACCEPTANCE
  (Medium mode = best throughput AND best tok/s/GB; int4 draft collapses
  acceptance - matches colibri #8)
- draft fully offloaded; DRAFT ATTENTION ON CPU (KV never leaves RAM,
  O(d) activations cross); draft GPU footprint ~0
- gate-first loading: 32KB gate resolves top-k BEFORE expert I/O;
  per-expert I/O threads; compute-as-ready; pinned memory everywhere
- adopt memory efficiency (tok/s/GB) as our primary serving metric

### REVISED V4-FLASH PLAN (IFP-primary)
phase A: trace routing on task corpus -> per-layer mass curves
phase B: 0.8B mask predictor trained (select top-N/layer per prompt)
phase C: lucebox + IFP loader graft (per-prompt set + periodic
         reselect + displaced-mass trigger); Spark cache on top for
         VRAM hot subset; DSpark windowed
phase D: MESO rules (draft CPU-attn, pinned, gate-first)

## V16 ADDENDUM: LEARNED SCAFFOLD CONTRACT (user directive)
"Use LUTs where you can; smart learned scaffold of small helper models
to speed things up AND to stabilise and restore performance."

### THE SCAFFOLD (one system, three functions)
SPEED (helper models + LUTs):
  - 0.8B (BF16, RAM): drafter / prefill scorer / IFP mask predictor
  - pre-gate heads (per-layer, tiny): next-layer routing oracle
  - routing-replay LUT (exact on repeated code n-grams, free)
  - CCT + affinity LUTs (statistical, calibration-built)
  - k-mass thresholds theta(layer) LUT from mass-vs-delta curves

STABILISE (keep performance from drifting):
  - acceptance monitor: draft acceptance < band -> auto-switch draft
    mode (ddtree budget shrink / ngram fallback)
  - cache-hit monitor: cold-hit rate > band -> trigger reselect +
    spark.csv refresh; adaptive pool sizes
  - KVFlash drafter-scored residency (stabilises long-ctx decode)

RESTORE (recover after degradation):
  - displaced-mass trigger -> batch expert reselect (not demand-read)
  - drift monitor: hidden-state norms vs calibration bands -> replay
    span with full path
  - heartbeat: full-path forward every N tokens; divergence ->
    corrector distill event (system converges to fewer replays)

### WIRING ORDER (build Phase 3 around this)
1. LUTs first (free, no training): replay + CCT + theta tables
2. Monitors next (instrumentation only): acceptance/hit/drift bands
3. Helper models last (training): pre-gate, then compensation, then
   health-corrector distillation from replay events
Every monitor has a RESTORE action bound to it; every LUT has a
freshness check. Nothing in the scaffold touches big-model weights.

## V17 CORRECTION: REAL KAT-CQ1 GEOMETRY (from file, not assumption)
- GGUF v3, 733 tensors: arch=qwen35moe, 40 blocks x 256 experts, top-8,
  hidden 2048, moe intermediate 512, ctx 262144
- dtypes: control F16 (332) + F32 norms/ssm (281) + Q4_K experts (120)
- expert = 3 x 2048 x 512 = 3.15M params = 1.77 MB at Q4_K
- PER-TOKEN expert bytes uncached: 40x8x1.77MB = 566 MB (NOT the 2.0GB
  from my earlier wrong 94x128 assumption)
- CPU-RAM path ceiling: 566MB / 55GB/s ~ 97 tok/s UNCACHED
- PCIe GPU path: 566/12 = 21 tok/s uncached -> needs cache f>=0.8 alone,
  OR f~0.7 + DFlash spec (x2-2.5) => 120+ tok/s
- 8GB VRAM budget: control ~3GB + KVFlash pool small + experts ~4GB
  = ~2.3K/10.2K experts resident (22%) + cache ring; Qwen 16x routing
  skew + Spark calibration covers 60-75% activations at 22% resident
- 100 tps REQUIREMENT: Spark cache f~0.7 + DFlash acceptance ~2+
  => ~120 tok/s ceiling. Feasible on THIS model. PPL-gated unchanged.

## V18 ENGINEERING DIRECTIVES (user, binding)
1. Custom fused kernels; minimize kernel launches; NO CUDA GRAPHS
   (crash the GPU on this card — hard rule, matches gpu-safety skill)
2. Use PC carefully: no GPU crashes, no RAM fill; careful VRAM budgets
3. Everything GPU-able on GPU; no slow CPU training — fast GPU paths only
4. Keep most implementation in C++ (server/engine), Python only for
   orchestration/calibration tooling
5. Proper debugging, testing, profiling, MEASUREMENTS at every step
6. Do not skip complexity; engineering/research effort is free
7. Continuous research ingestion: latest 2026 MoE inference papers,
   fold into implementation (next: openreview r8YhlMRUR2)
8. NOVELTY TARGET: findings/tricks beyond the papers are a bonus goal
   (one already found: see V17 corrected-geometry insight — 566MB/token
   makes 100tps CPU-path-feasible; more below as they emerge)

## NOVEL FINDINGS LEDGER (append-only)
[N1] KAT-CQ1 real geometry (40x256 top-8, 1.77MB/expert) => 566 MB/token
     uncached expert traffic; 100 tps reachable via RAM path alone at
     ~97 tps ceiling pre-cache, cache+spec pushes past. (2026-08-17)

## V20 PREFETCH-GAP SOLUTION (Prefetch-Gap-Scout, 4 papers full-text)
THE 53% CAP IS WITHIN-LAYER ONLY. Cross-layer signals break it:

### FATE (arXiv 2502.12224) — ZERO-TRAINING, DEPLOY FIRST
- clone layer i gate input (post-attn, 8KB) to CPU; run layer i+1 ACTUAL
  W_g on CPU (524K FLOPs, 0.1-0.3ms) => 78.8% top-k recall
- OVERFETCH to 75th-pct of predicted dist => 97.15% recall
- 40 layers x [2048,256] W_g = 80MB RAM total. Fits 2-5ms layer window
- 75th-pct overfetch = 16-32 experts x 1.77MB = 28-56MB = 1.2-2.5ms PCIe
- Fate measured: 99.08% hit rate, 4.1x vs LoD, <1% quality loss
- ALSO: shallow-favoring cache (first L layers fully cached; accuracy
  lower there), ARC eviction, INT2/INT4 hybrid expert transfer

### PROMOE (2410.22134) — learned cross-layer MLP, 84.7%, 1-2h train
- chunked prefetch (3 chunks/expert), early preemption (gate hook
  enqueues exact experts HIGH prio, cancels speculative), reordered
  inference (cached experts first — needs graph split in our fused path)

### RLCG HYBRID (scout proposal, ours): routing-replay LUT (exact on
repeated code n-grams) UNION cross-layer gate fallback => est 85-95%
blended; ~128MB LUT in RAM, ~1us lookup, 0.05-0.3ms/token

### DTRR HYBRID (ours): 0.8B helper's OWN gate output + hidden ->
linear correction (0.7M/layer) -> big-model routing; est 80-92%;
0.05ms/layer; trains from existing spec-decode captures

### LEXI (2509.02753): per-layer top-k reduction via MonteCarlo
sensitivity (no data, no training) — robust layers drop to k=4 =>
halves prefetch bytes + doubles effective recall. PPL-gate mandatory.

### ORDER: Fate (free, now) -> LExI profile (one-time 320s) -> RLCG
LUT (trace-built) -> DTRR when helper deploys

## V21 PHASE-0 RESULTS (measured)
### KAT-CQ2 BUILT + VALIDATED (fixes CQ1 CUDA crash)
- CQ1 F16 control plane tripped binbcast assert (nb10 % sizeof(src1_t))
  on GDN tensors in this llama.cpp build (10331/7ba604f1c)
- CQ2: official-parity dtypes — matmuls Q8_0, norms/routers F32, experts
  byte-copied Q4_K from CQ1. Subnormal-scale quantizer bug fixed (d floor
  FLT_MIN). Verified: ssm_out Q8 rel-err median 0.68% vs transformed BF16.
- MEASURED on stock llama.cpp CUDA, ngl99 fa1 (vs official Q4_K_M):
    pp64: 59.88 -> 68.76 (+14.8%)
    tg64:  6.30 ->  6.78 (+7.6%)
- CQ2 IS THE SPEED TARGET MODEL. Stock baseline = 6.78 t/s. Goal: 100+.

## V22 BASELINE CORRECTION (user flag)
- llama-bench tg64=6.78 is AR-only kernel floor, NOT the real baseline
- USER's actual serving stack (serve-rebase.cmd): dFlash MTP draft +
  ngram-simple spec + -cmoe + fa on + KV q8_0 + cache-reuse = 30-40 t/s
  on this model — THE baseline to beat; 100 t/s = ~3x user's best config
- Action: bench stock+spec (llama-server -md dflash-q8.gguf
  --spec-type draft-dflash,ngram-simple) as honest comparison line
- lucebox ladder must stack: pipelined decode (1.84x) x spark cache x
  DFlash draft (27B-family head downloaded) x Fate prefetch

## V23 PHASE-2 MEASURED LADDER (stock llama.cpp, KAT-CQ2)
| Config | tok/s |
|---|---|
| llama-bench AR (fa1 ngl99) | 6.78 |
| -ncmoe 8 / 16 / 24 AR | 7.4 / 9.2 / 10.5 |
| -cmoe AR (all experts CPU) | 12.1 |
| -cmoe + dFlash+ngram spec (cold short) | 15.4-18.9 |
| -cmoe + spec, agent+tools warm prefix | 12.7-18.7 (prefill 92s->9s w/ cache) |
| code-gen long | 8.8 |
VERDICT: stock tapped ~19 t/s. CPU-expert path = 566MB/tok serial GEMV;
multiplier must come from (a) family MTP head 67-89% accept (NInfer-proven
on same arch), (b) batched verify over GPU hot set (lucebox), (c) leaner
expert bytes (byteshape 3.53bpw, 450MB/tok).
NEXT: bs-mtp-iq4xs.gguf (downloading) -> --spec-type draft-mtp test ->
graft MTP head into KAT-CQ2 via byte surgery -> lucebox draft attach.

## V25 STOCK AR FULLY CHARACTERIZED (every lever swept)
| Lever | Best | Measured |
|---|---|---|
| quant | bs-3.53bpw ≈ CQ2 | both ~12 t/s warm |
| expert placement | -ncmoe 45 (all CPU) | 12.0-12.1 |
| threads | 8 physical (SMT hurts) | 12.0 vs 8.8@t16 |
| fa/KV/cache | fa on, q8 KV | in all runs |
STOCK AR CEILING: ~12 t/s (sustained warm; ~13 short bursts).
User's 15-20 AR memory: remaining delta likely batch shape (-b/-ub)
or build-specific GDN CPU path. Next: -b sweep, then accept 12-15 as
stock AR floor and move the fight to lucebox pipelined + MTP3 (the
NInfer-proven combo: batched verify is where 5->100 comes from).

## V26 MTP+t8 STACKED (measured)
byteshape 3.53bpw + draft-mtp n4 + t8: 13.8 median / 17.0 peak
(acceptance 61-62%, len 3.4). Stack so far: AR 9.6->12.3 (t8) ->
MTP 13.8 (median). SPEC MATH: verify batch of ~4.5 tok over 256-expert
top-8 fragments to M~1.4 rows/expert -> no GEMM amortization; expert
bytes/round unchanged. Conclusion: reaching 100 t/s on THIS box needs
verify-batch-friendly expert residency (whole hot tiers resident) or
NInfer-style persistent batched MoE kernels — Phase 2b/3 focus.

## V27 SPEED DECOMPOSITION (measured via thread scaling, stock)
t1=4.8 t2=5.1 t4=7.1 t8=12.5 t/s -> per-token: b+c=208ms, b+c/8=80ms
=> bandwidth-bound 62ms (78%), compute 18ms (22%)
Expert bytes 566MB/62ms = ~9.1 GB/s effective vs ~55 GB/s available
=> THE BOTTLENECK IS THE GATHER PATTERN (320 scattered GEMV/token),
NOT raw bandwidth. MoE batch-1 wastes 5/6 of RAM bandwidth.
FIX RANKING (by expected yield):
1. Spec-verify batching: 4.5 tok/round amortizes each expert fetch
   (bytes/token ÷4.5) — but stock MTP only got 13.8 because verify
   fragments across 8 experts/layer (M~1.4 rows/expert)
2. Expert prefetch/pipelining (hide gather behind compute) — lucebox
3. Whole-tier residency (skip the fetch entirely) — lucebox hot tiers
This closes the loop: every speed approach we have maps to one of
these three, and the 62/18 split quantifies the ceiling of each.

## V28 PHASE-2B RESULT: PIPELINED PATH WINS (measured)
lucebox dflash_server + KAT-CQ2 + q8 draft (spec self-disabled at 6%):
  AR decode 20.2-22.0 tok/s sustained; e2e median 17.7 (n=10)
  vs stock AR 12.3 => +75%. All blockers fixed this session:
  (a) 4-arch compile -> sm_86 only; (b) 512MB metadata arenas x3 gens
  -> 32MB; (c) truncated draft re-dl; (d) heap-fragmentation malloc
  fail -> VirtualAlloc fallback w/ registry-routed free.
CRITICAL NEXT RUNG: family MTP head (61-68% accept measured) as a
STANDALONE draft GGUF for lucebox --draft. Extract blk.40 MTP block
from byteshape GGUF -> convert to dflash-draft format (5-block DFlash
head schema: embed/norm/4 blocks + fc shared/lm_head) via our GGUF
surgery tooling. Expected: 3.5 tok/round x pipelined verify batch
=> 40-60 t/s class. Then Fate A/B + tier tuning for the rest.

## V29 MTP-GRAFT VERDICT (measured)
KAT-CQ2-MTP.gguf: VALID (753 tensors, parses, coherent output,
drafting ENGAGED: acceptance 36-59% on KAT weights). Graft bugs fixed:
KV type 0->4 (UINT8 vs UINT32 trap), relative offsets, e0.tensor_nbytes
(build_gguf Q8_0 = 1/8 size bug), verbatim donor names.
STOCK RESULT: 8.9 t/s (WORSE than AR 12.3): the MTP block is itself a
full MoE layer — each draft step pays its own 566MB expert gather on
the non-amortizing CPU path. THIRD confirmation of the fragmentation
diagnosis. Stock engine: mathematically closed (~19 ceiling).
LUCEBOX: embedded-nextn consumer absent from draft path (eh_proj has
no consumer); would need DFlash-5-block draft-format conversion of the
MTP block. REMAINING PATHS TO 100:
(a) lucebox MTP draft-format converter (graft MTP block INTO the
    dflash-draft schema — small model, our surgery tooling)
(b) ninfer gdn_replay/mtp_pack port (batched CUDA verify kernels)
Both are C++ engineering on the working 22 t/s pipelined base.

## V30 DRAFT-FORMAT INVESTIGATION CLOSED (measured)
DFlash draft schema requires fc [n_capture_layers*n_embd, n_embd] fed by
ALL target layer features (40*2048=81920 -> 2048 for us). MTP eh_proj
maps [h;e]=4096 -> 2048. NOT convertible by surgery — fc must be trained
(41MB linear, feasible w/ our tooling as a future rung) or the target
engine modified to feed fewer layers. shexp-draft.gguf loads and
validates dimensionally w/ n_target_layers=1 but crashes at runtime
matmul when the engine feeds the full 40-layer feature vector.
DECISION: stock draft path closed at 19 t/s; lucebox draft needs trained
fc (future). NEXT: Fate A/B on the 22 t/s pipelined config — the
remaining multiplier available TODAY.

## V31 CUSTOM DRAFT FINAL VERDICT (measured)
- ENGINE SIDE: all fixed. capture-ids override works (N=1, id=39 log),
  fc sliced (2048,2048), Q de-interleaved per-head, dims all validate,
  draft LOADS and RUNS on the pipelined engine, crash-free.
- DRAFT QUALITY: acceptance 6.2% — the shexp-only single-block draft is
  structurally sound but semantically mismatched: DFlash drafts are
  FEATURE-CONDITIONED (attention over target feature stream via the
  dflash protocol), not token-embedding transformers. Correct drafting
  requires either (a) training the fc+draft jointly on traces (41MB
  trainable, our tooling), or (b) ninfer-style MTP-native kernels.
- RESULT TODAY: with the weak draft auto-disabled: 22.4 median /
  24.3 peak tok/s (n=10) — BEST measured config on this box.

## V32 CQ3 + gbuzhf RECIPE — SPEC DECODE WORKS (measured)
KAT-CQ3-MTP: CQ2 trunk + pristine BF16 MTP head (gbuzhf donor =
Qwen3.6 original) requantized OUR way (Q8_0 matmuls / Q4_K RTN
experts / F32 norms). Build bugs fixed: sf keys lack .weight on
expert tensors; n_kv off-by-one (56 vs 55).
RECIPE (gbuzhf coordinate-ascent, 79 configs, same GPU class):
  draft-mtp,ngram-mod TOGETHER; n-max 1 (not 2+); p-min 0.75;
  ngram-mod 8/24/48. MTP alone = net loss (we measured 8.9 — same).
RESULTS (ours):
  mixed coding bench: 18.1 med, peak 48.6, acceptance 79-82%
  copy-heavy: 46.5 MEDIAN / 48.6 sustained (vs AR 12.3 = 3.8x)
  vs gbuzhf's same-recipe numbers (71 copy / 33 agentic): ours in
  range accounting for their UD-Q4_K_XL tier vs our CQ2 map.
LADDER NOW: stock AR 12.3 -> lucebox AR 22.4 -> CQ3 spec 46.5 copy.
NEXT: same recipe on the lucebox pipelined engine (22.4 base x spec
multiplier) — the engine accepts embedded nextn; CQ3 has it.

## V33 ENGINE-STACK FINAL MAPPING (measured)
lucebox loads CQ3 (nextn validated, excluded from target) but its
qwen35moe draft path is DFlash-format only — embedded MTP not wired
(no eh_proj consumer on that path; verified twice).
=> The two best measured configs are ENGINE-BOUND:
  - COPY/PATTERN workloads: stock+CQ3 recipe => 46.5/38.8 median
  - MIXED coding: lucebox pipelined => 22.4 median / 24.3 peak
CROSSOVER: ngram-mod hit-rate decides. Agent loops w/ repeated tool
schemas + code patterns = stock+spec wins. Cold novel = lucebox AR.
100+ remains gated on: trained DFlash-fc draft on lucebox (41MB) or
NInfer batched-verify kernels. Both scoped; neither is config work.

## V34 SCHOLAR SWEEP TOP-3 (10 new papers, Jan-Aug 2026)
1. 35B-on-6GB (2606.24031): OUR EXACT MODEL FAMILY (Qwen3.6-35B-A3B,
   GDN, 4-bit) on 6GB 2011 GPU. Hand-written W4A8 SSSE3 CPU GEMV decode
   + grouped Q4 expert GEMM. Negative-results map: GPU-head offload,
   hyperthreading, GPU-kernel rewrites ALL FAIL. Direct recipe + trap map.
2. AcceptMoE (2608.02989): verifier-side expert-set restriction
   conditioned on CACHE RESIDENCY (not prefetch). 2.06x under offload,
   -73-77% H2D traffic, 0.27pp loss. Composes w/ our ngram tree natively.
3. WiSP (2606.21868): MEASURED "prefetching helps little in single-
   stream PCIe-bound decode" -> VRAM ALLOCATION (MV-WSA) is the lever.
   1.95x byte-identical. Redirects our prefetch-heavy plan.
+ EVICT (2605.00342): lossless draft-tree truncation (compose).
+ EcoSpec (2607.12696): re-rank ngram candidates by expert-reuse cost.
VALIDATION: 2606.21428 (llama.cpp 8GB Jetson study): "cost tracks TOTAL
params not active ones" — matches our 9.1/55 GB/s decomposition exactly.
IMPLICATION: Fate A/B result (whatever it is) likely small; the big
levers are (a) AcceptMoE verify-restriction, (b) W4A8 CPU GEMV kernel,
(c) EcoSpec ngram re-ranking. All C++ portable, no CUDA graphs.

## V35 FATE A/B RESULT (measured, decisive)
Fate cross-layer prefetcher ON (lazy-init fix confirmed: "OK (40
layers)", placement intact at 3.05GiB core): 4.0-4.6 t/s.
OFF: 22.4 t/s. => 5x REGRESSION. Root cause: per-layer sync D2H
(ffn_post pull) + CPU gate matmul on the same 8 cores doing expert
GEMV — the prefetch overhead IS the pipeline stall.
CONFIRMS WiSP (2606.21868) on our hardware: single-stream
bandwidth-bound decode gains ~nothing from routing prediction.
PREFETCH PATH CLOSED. Remaining levers (all verify/allocation-side):
AcceptMoE residency-conditioned verify sets, EcoSpec ngram candidate
re-ranking by expert reuse, EVICT tree truncation, W4A8 CPU GEMV.

## V36 ACCEPTMOE-STYLE UNION RESTRICTION — MEASURED NEGATIVE (stock engine)
Patch: llama.cpp b9873 build_moe_ffn + KAT_UNION_K env (commitment-weighted
union mask, 2..64-token batches only). Binary verified (KAT_UNION_K string
in llama.dll). A/B same binary, same CQ3+gbuzhf recipe, n=5/workload:
  baseline (off): copy 16.3 med / pattern 20.8 med (e2e incl prefill)
  KAT_UNION_K=24: copy 14.4 / pattern 19.6  (-12% / -6%)
  acceptance: unchanged 0.75-0.79 copy (restriction not quality-bound)
INTERPRETATION: with acceptance stable, the extra mask graph ops (9 ops x
40 layers x verify step) cost more than the expert-read savings — because
the natural union across verify rows is ALREADY compact in stock
(near-duplicate rows -> highly correlated routing). AcceptMoE's union
blowup premise does not bind at our tree sizes (n-max 1, mean len 14-19).
The M~1.4 fragmentation is a LUCEBOX-verify-path property.
DECISION: keep patch in tree (off by default, env-gated); record negative;
pivot to lucebox pipelined + spec (the 22.4 -> 55-70 path).

## V37 DFLASH-FC DRAFT CAMPAIGN STATE (2026-08-18)
Infra all verified: trace dumper (3 layout bugs caught+fixed: layer-major
staging, chunk-indexed staging, 2GB ftell overflow), 136,986 verified rows,
export roundtrip (rel err 0.0054 corr 1.0), engine probes (features norm 30
= healthy; head path clean).
TRAINER LADDER (val acc = chain-relevant top-1):
  v2 fc-only, last-token QUERY geometry (wrong vs engine): 74.2%
  v3 fc-only, engine-exact mask-query block: 6.7% (full 16-row) / 17.2% (rows 1-4)
  v4 whole-draft unfrozen (fc+attn+ffn ~90MB, lr 1e-4): running
DIAGNOSIS: mask-query block-diffusion needs a net TRAINED for it (DFlash
trains 5 layers from scratch); donor MTP block was trained for sequential
eh_proj fusion. fc-only cannot bridge; whole-draft may partially.
ECONOMICS: block verify costs verify_width target rows (6-16). >100 t/s
needs ~0.7 sustained per-row acceptance. Row-0 (last_tok query) geometry
= v2's 74% — single-proposal mode (verify width 2) is a guaranteed
fallback worth ~1.5x (22.4 -> ~33) if v4 disappoints.
NEXT: bench v4; if >=40% row-1 acc -> full chain on lucebox; if not ->
single-proposal patch OR pivot remaining effort to stock-spec verify
batching (46.5 copy base).

## V38 DRAFT PATH FINAL VERDICT + PIVOT (2026-08-18)
v5 (fc-only, all-rows last-tok query block): collapsed to 6.9% — identical
queries + RoPE-only differentiation = degenerate for the frozen donor
layer. Trainer ladder complete: 74.2% (v2 geometry, single-row causal) /
6.7% / 17.2% / 16.7% (whole-draft) / 6.9% (v5). CONCLUSION: block-diffusion
drafting requires a net trained from scratch (DFlash trains 5 layers; our
1-layer donor MTP block cannot). Single-proposal caps at ~1.2x.
PRESERVED: trace infra, 74% fc artifact, DFLASH_QUERY_LAST_TOK flag,
engine probes. PIVOT: union restriction v2 — V36 measured acceptance
STABLE (0.79) under K=24; v1 loss was pure graph-op overhead. v2 =
penalty as static input tensor (1 add op, no rebuilds). Target: halve
expert reads per verify row on the 46.5 copy config => 70-93+ t/s.

## V39 USER-DUMPED PAPERS — THREE NEW LEVERS (2026-08-18)
1. DSpark (2607.05147 + HF ecosystem): DFlash iteration, block draft +
   Markov head + confidence head. KOOPAH V2 DRAFT EXISTS FOR OUR EXACT
   FAMILY (Qwen3.6-35B-A3B hybrid GDN): 3.1x solo vLLM, ~2.5x llama.cpp
   class, per-pos acceptance ~0.78 FLAT over 8 positions, works on
   Q4_K_M-tier targets (0.313). GGUFs shipped. llama.cpp branch:
   satindergrewal/llama.cpp@dspark-qwen35 (cloned to C:/src/lmdspark).
   NOTE from card: acceptance tracks TARGET quant quality (our CQ2/CQ3
   imatrix-class => good); head shares target embed+lm_head (common-mode
   quant error — no precision tax).
2. JetSpec (2606.18394): causal parallel draft head — one-forward block
   drafting WITH branch-wise causal conditioning (solves exactly the
   marginal-vs-chain dilemma that killed our donor-block attempts v3-v5).
   9.64x H100 MATH-500. Code+models public.
3. Attention Drift (2605.09992): explains v2->chain collapse (hidden-state
   magnitude grows monotonically with chain depth in pre-norm drafter
   stack). Fixes: post-norm drafter hidden + per-hidden-state RMSNorm
   after target capture (= lucebox aux_hidden_norms, already plumbed).
4. Bole (2608.01651): tree speculation for HYBRID-ATTENTION models (our
   class): closed-form tree recurrence, parallel node verify 3.4-7.7x,
   token-level state factors. SGLang-only today; blueprint for a lucebox
   hybrid_forward_batch tree-verify patch.
PLAN: DSpark v2 GGUF on dspark branch NOW (drop-in, 2.5-3x class on our
46.5 base => 100-140). JetSpec head training as follow-up using our
136k verified traces. Drift fixes if we retrain anything ourselves.

## V40 SPD (2605.30852) — PIPELINE-DRAFT SYNERGY (2026-08-18)
SPD: partition target into n pipeline stages; PDM draft aggregates
multi-depth target features, runs CONCURRENTLY with pipeline steps ->
bounded prediction difficulty (single next-token), higher acceptance,
hidden draft latency.
MAP TO US: lucebox IS a pipelined engine whose DFlash draft already
consumes 5-capture-layer features (= PDM design). Our 74% row-1 fc fits
the PDM role exactly. But lucebox runs the draft SERIALLY between steps
-> hiding draft latency behind the pipeline = scheduling change in
do_hybrid_spec_decode, no new weights.
STACK VIEW: SPD hides draft latency + Bole hides verify latency (tree
closed-form for GDN) + DSpark raises acceptance (drop-in head). All
three compose on lucebox.

## V41 DSPARK v2 ON KAT — WORKS (measured 2026-08-18)
Koopah Qwen3.6-35B-A3B-NVFP4-DSPARK-v2-Q8_0 (556MB) on KAT-CQ3-MTP via
satindergrewal/llama.cpp@dspark-qwen35 (block_size=8, n_extract=8):
  acceptance 0.466-0.506, mean accepted len 4.69-5.05
  e2e (ab_union, n=5): copy 29.5 med/38.8 peak (vs 16.3 stock-spec = +81%)
                       pattern 23.1 med (vs 20.8 = +11%)
  decode tg 18-22 (tg_3s spikes 29.9); verify = 8-9 row batches.
CROSS-MODEL TRANSFER CONFIRMED (KAT is a qwen35moe-family finetune).
Head shares target embed+lm_head; acceptance tracks target quant quality
(CQ2/CQ3 = imatrix class => good).
NEXT LEVERS: (a) --spec-draft-conf-min tuning; (b) compose ngram-mod;
(c) PORT union-v2 patch to lmdspark — DSpark verify batches (8-9 rows)
are exactly the 2..64-token gate; V36 showed acceptance STABLE under
K=24 restriction; restriction cuts per-verify expert reads (the
dominant verify cost on CPU-resident experts).

## V42 UNION-V2 + DSPARK A/B (measured 2026-08-18)
Arms (same binary, same bench, e2e medians):
  dspark alone:      copy 29.5-37.7 / pattern 20.7-23.1, peak 39.2/23.7
  dspark + K=24:     copy 32.5 (peak 44.4) / pattern broken-run (peak 30.6)
ACCEPTANCE ROSE under restriction: 0.60-0.81 vs 0.38-0.52 (restriction ->
converged expert sets -> sharper greedy path -> higher draft agreement).
CAVEAT: intermittent early-EOS derailment (gen=1 rows) — restriction
sometimes breaks generation; likely EMA cold-start + K too tight early.
NEXT: (a) warmup gate — no restriction first N steps (EMA fill); (b) try
K=48/64; (c) check gen=1 pattern (which prompts, what position).

## V43 UNION-V2 LADDER COMPLETE (measured, e2e ab_union medians)
dspark alone      copy 29.5-37.7 / patt 20.7-23.1 (peak 39.2/23.7)
dspark K=24 cold  copy 32.5 peak 44.4 / patt peak 30.6 — but early-EOS
                  derailment; acceptance spiked 0.60-0.81
dspark K=48 warm  copy 38.1 peak 38.9 / patt 20.3 (accept 0.37-0.52)
dspark K=24 warm  copy 35.7 peak 37.8 / patt 21.2 (accept 0.47-0.53)
VERDICT: restriction is SAFE with warmup (no derails) but the cold-start
spike was the interesting artifact — warm gating removed both the derails
AND the acceptance spike (same mechanism: early restriction = stronger
prior = sharper path). Net: union-v2 ~= wash at K>=24 on this workload;
expert-read savings offset by nothing dramatic. The 44.4 peak suggests
per-request adaptation (restrict hard early, relax later) as the open
thread, but not the 100tps lever by itself.
BEST CONFIG NOW: dspark v2 alone, copy 37.7 med — 3.1x over stock AR 12.3.
REMAINING 100TPS GAPS: verify cost (8-9 rows through CPU experts) and
prefill share in e2e. Decode-only rates are the target metric.

## V44 HANDOFF RECORD — UNTRIED ARMS + BOTTLENECK QUALIFICATION
STATE: best = dspark v2 alone, copy 37.7 e2e med (3.1x stock AR). The 100tps
goal is NOT met. Two ZERO-REBUILD arms untried (both are relaunch-only):

ARM A — ngram-mod COMPOSE (--spec-type draft-dspark,ngram-mod):
  Hypothesis: ngram feeds high-confidence continuation tokens into the
  same verify batch; on repetitive code (agent loops, schemas) commits/
  step grow past dspark's 4.7-5.2 mean len. Judge on a HIGH-REPETITION
  copy prompt (ab_union COPY is moderate; craft a repetition-heavy one).
  Decision: adopt if commits/step grows >15% without acceptance collapse.

ARM B — conf-min TRUNCATION SWEEP (--spec-draft-conf-min 0.2/0.35/0.5):
  Hypothesis: Koopah head ships a confidence head; gating verify rows
  cuts the 8-9-row verify cost where acceptance is doomed anyway
  (their card: no engine wires it yet; we can). Decision: adopt the
  best latency/step x commits/step product.

BOTTLENECK QUALIFICATION: "verify cost is binding" is measured (e2e vs
decode-rate gap + union null result). WHICH verify component dominates
(expert gather vs GDN recurrence vs GPU<->CPU handoff) is NOT yet
measured. Union-v2's null ELIMINATED expert-set-size as the lever at
K>=24 (safe-warmup). NEXT SESSION SHOULD START WITH a per-stage verify
profile (server timing buckets or nsys) BEFORE any Bole-scale patch —
Bole targets GDN recurrence; if the profile says expert gather or
handoff instead, Bole is the wrong patch.

## V45 COMPOSE ARM — DSPARK + NGRAM-MOD WORKS (measured)
--spec-type draft-dspark,ngram-mod (+ p-min 0.75, ngram 8/24/48):
  copy 47.4 MED / 52.2 PEAK (vs 37.7 dspark alone = +26%)
  pattern 21.7 / 26.5 (parity w/ dspark alone — expected, ngram needs
  repetition)
  mean-len spikes 7.60 on repetitive tasks (dspark-alone ceiling ~5.2)
  => ngram continuations verified in the same batch and accepted.
BEST CONFIG NOW: compose. 3.9x stock AR e2e. Union-v2 not needed here.
REMAINING GAP to 100: ~2.1x. Verify cost per step (9-16 rows through
CPU experts) still binding; prefill share matters on short gens.

## V46 WIDTH-4 DECOMPOSITION (measured)
n-max 8:  copy 47.4 med / decode tg peak 53.0, mean-len 4.7-7.6
n-max 4:  copy 46.9 med / pattern 23.8 (up from 21.7!), tg peak similar
HALVING VERIFY ROWS CHANGED ALMOST NOTHING (46.9 vs 47.4; pattern +2).
=> VERIFY WIDTH IS NOT THE BINDING COST at these acceptance levels.
The binding cost is per-STEP fixed overhead: draft forward (556MB head
through the same CPU/GPU split) + graph launch + handoffs. Draft cost
per committed token is ~fixed since both widths take 1 draft pass/step.
NEXT: profile WHERE the step time goes (draft fwd vs verify fwd vs
handoffs) before any Bole-scale patch. Also: --spec-draft-threads,
draft GPU residency flags if they exist.

## V47 NGLD-ALL NEGATIVE + STEP-COST DECOMPOSITION (measured)
-ngld all (draft fully GPU): copy 23.3 (vs 47.4 compose) = 2x REGRESSION.
Draft on GPU contends with target for VRAM+bandwidth — 3rd confirmation
of the GPU-residency-loses pattern on 8GB (V29 GPU-head, V42-era sweeps,
now draft). DRAFT STAYS CPU.
Width-4 vs 8 null (V46) + this: the per-step cost is NOT verify rows and
NOT draft residency — it's the serial draft->verify->draft round trip
itself (graph launches + CPU/GPU handoffs per step).
STATE: best = compose config, copy 47.4 med/52.2 peak e2e, decode tg
peaks 53. 100tps needs ~2x more: candidates in order:
  1. PER-STEP PROFILE (server timing buckets) — know exactly where the
     ~50ms/step goes before patching
  2. Bole-style fused tree-verify (GDN closed form) — kills the serial
     round trip, biggest possible win, biggest patch
  3. SPD-style pipeline-hidden drafting — draft overlaps verify

## V48 PER-STEP PROFILE (verbose server, measured 2026-08-18)
256-token novel-text gen, 77 spec steps, mean acc len 3.30:
  draft (dspark fwd): 16.9 ms/step  (12%)
  ngram drafting:     ~0 (table lookup)
  accept accumulate:  ~0
  => verify + fixed:  ~127 ms/step (88%)   [total decode 11.1s - 1.3s draft]
Acc/pos DECAYS on KAT: (0.71, 0.51, 0.31, 0.22, 0.17, 0.16, 0.12, 0.10)
  — vs Koopah's flat ~0.78 on their Qwen3.6 target. KAT finetune drift.
IMPLICATION 1: 100tps at acc-len 3.3 needs step ~46ms = verify 3x faster
  (Bole-class patch) OR acc-len ~7-8 (draft-quality ceiling on KAT).
IMPLICATION 2: V46 width-null is CONFOUNDED — ngram compose backfills
  the verify batch regardless of --spec-draft-n-max. Deconfound: pure
  dspark, n-max 8 vs 3, measure tg.

## V49 VERIFY COST MODEL — EXACT (measured, deconfounded)
Pure-dspark novel-text width sweep (n-max 3 vs 8), tg decode:
  w=3: 17.0 t/s | w=8: 10.6 t/s  => V46 null was ngram-backfill confound.
Two-equation solve (steps = 256/E[len]):
  step_time = 17ms (draft, fixed) + 33ms x verify_rows. Fixed ~ 0.
Optimal novel width ~2-3 (model peak ~22 t/s == lucebox AR 22.4 sanity).
Acc/pos decay (0.71,0.51,0.31,...) makes deep rows pure waste on novel.
COPY is different: ngram acceptance ~1 fills w=8 profitably (47-53 t/s).
=> ENGINE-POLICY RULE: adapt width per step from ngram-hit + conf head
   (conf-min exists; per-step ngram-aware width is a small patch).
100TPS DECODE MATH: needs 33ms/row -> ~10ms/row. That is expert-gather/
GDN-row cost through -cmoe CPU path — the Bole/fused-verify class of
patch, or batched expert kernels on the verify rows. All smaller levers
are now measured and exhausted on this engine build.
FINAL SESSION STATE: best copy 47.4/52.2 (compose), novel ~22 (either
engine), 3.1-3.9x stock AR. 100 sustained not yet reached; remaining gap
is concentrated in the 33ms/verify-row term.

## V50 TRUE-COPY PROTOCOL — NEW BEST (measured 2026-08-18)
Verbatim-reproduce-a-large-file task (the original 46.5-protocol class),
compose config, n=5, 1400-token gens:
  e2e:  56.0 MEDIAN / 59.5 PEAK t/s  (5.6x stock AR e2e, 4.6x its tg)
  decode-only: 68.5 t/s peak
  acceptance 0.951, mean accepted len 21.2-23.7/step (!)
MECHANISM: ngram-mod at ~1.0 acceptance on verbatim echo + dspark on the
novel joints -> verify batch fully utilized at w=8+ (21+ commits/step).
THIS is the agent-relevant workload class (file regen, refactor echo,
template expansion, test scaffolds).
100TPS REMAINS OPEN: decode 68.5 < 100. Cost model says verify rows now
come in blocks of ~24 (33ms/row * amortization) — the Bole-class fused
tree-verify is the remaining 1.5x, and SPD-style pipelining could hide
the 17ms draft on top. Both scoped; engine builds + artifacts preserved.

## V51 WIDTH-SWEEP FINAL — W=8 COMPOSE IS THE OPTIMUM (measured)
True-copy protocol, decode tg peaks, only width/params varied:
  w=8  (p-min .75, ngram 8/24/48): 68.5 tg | 56.0/59.5 e2e | mean-len 24
  w=12 (p-min .70, ngram max 32):  ~31.5   | mean-len 29
  w=16 (p-min .60, ngram 48/64):   ~32.8   | mean-len 33
Wider drafting raises commits/step but verify-row cost dominates
(superlinear: wider batches fragment across MORE distinct experts per
row — the M~1.4 effect compounds). Cost model V49 confirmed again.
FINAL BEST CONFIG = kat-compose.sh (w8 recipe). Campaign config space
EXHAUSTED on this engine build. Remaining path to 100 (scoped, not
built): Bole-class fused GDN tree-verify (33ms/row -> ~10) OR SPD-style
draft/verify pipelining. Both target the same measured bottleneck.

## V52 SPLIT-CHATTER DISCOVERY + PIPELINE TEST (2026-08-18)
MEASURED: -cmoe runs at 82 graph splits per bs=1 forward (sched_reserve
banner). Each split = serialized GPU<->CPU sync round trip (WDDM). At
12.3 t/s AR (81ms/step), sync overhead plausibly dominates the 33ms/row
verify term in V49 — the cost model's "per-row" component is partly
PER-SPLIT CHATTER.
TEST: KAT_PIPELINE env gate added (llama-context.cpp:365) forcing
ggml_backend_sched pipeline_parallel for tensor-override configs.
RESULT: gate works but pipelined reserve FAILS — 69GB buffer request
(n_copies=4 x unified graph incl. CPU expert nodes all placed on CUDA0).
Upstream pipeline path was built for multi-GPU layer splits, not the
-cmoe override topology. Needs: split-aware gallocr reserve (real
scheduler work, not a flag) OR op-offload=False variants.
STATE: recorded as open thread w/ exact failure signature. CQ4 build
nearing completion -> quality gate first.

## V54 NOVEL-TEXT 100 TPS — HONEST ROADMAP (user refocus 2026-08-18)
User: 100 t/s NOVEL text is the target (copy doesn't count). Current
novel: 17-22 t/s. The ladder to 100 on novel (measured constraints):
  acceptance on novel (KAT, dspark-v2 head): 0.71/0.51/0.31/0.22...
  -> mean len 3.3 commits/step. Novel t/s = commits / step_time.
  step_time today ~= 17ms draft + 33ms/row x rows.
THREE MULTIPLYING LEVERS (all needed):
  1. FEWER BYTES/ROW: CQ4 (Q3_K experts, building) -> 33->~25ms/row.
     Quality-gated (PPL <=+0.05 vs Q4K floor) or discarded.
  2. HIGHER ACCEPTANCE: dspark head TRAINED ON KAT itself (not the
     Qwen3.6 sibling). satgeze's DeepSpec gguf-capture tool validates
     cosine 0.9998 vs HF pipeline; trained-for-target heads add the
     deep-position acceptance novel needs (0.31->0.5+ at pos 3 = mean
     len 3.3->4.5+). ~1-2 day training lane, zero big-model training.
  3. OVERLAP: draft hidden behind verify (SPD-style) or split-pipeline.
     Pipeline reserve fails on our topology (69GB at bs=512 pp graph);
     NEXT: diagnostic print of per-backend pipelined-reserve sizes to
     find the real inflation (blind patch reverted).
MATH: CQ4(25ms/row) + KAT-head(len 4.5) + serial = (17+112)/4.5 = 29
  ... still short. Need row cost ~8ms (bandwidth floor at Q3: ~19ms/row
  pure read; 8ms only if expert reads batch across rows — the union-K
  idea APPLIES HERE: cap distinct experts/verify at ~16 -> 1.1GB/step).
STACK: CQ4 + KAT-head + union-K16 + draft overlap ~= (17+~45)/4.5
  ~= 100+. Each piece measured/scoped; none require big-model training.

## V55 PIPELINE 69GB MYSTERY SOLVED (KAT_ALLOC_DIAG, measured)
The pipelined reserve's 69GB CUDA request = EXPERT WEIGHT VIEWS being
treated as split-boundary copies: 40 layers x 3 exp tensors x 144MiB x
4 copies (n_copies=4). The scheduler materializes split-input tensors
on the DESTINATION backend — for real activations that's correct, for
read-only MoE weights (CPU-resident by design) it's catastrophic.
FIX SHAPE: in ggml_backend_sched_split_graph, when a split input's
source buffer is a WEIGHTS-usage host buffer, keep it host-side and use
async H2D view (the data is identical every step; per-copy duplication
is pointless). ~20 lines in one function, no cuda graphs.
IMPACT BOUND: removes the OOM -> pipelining CAN engage -> overlaps the
82 serialized GPU<->CPU syncs/forward. Novel step ~= 17ms draft +
(33ms/row x rows) where the 33 includes sync stall; pipelining
recovers the stall portion only (est 30-50% of it). STACKS with CQ4
(fewer bytes) and KAT-head (more commits/step).

## V56 PIPELINE BREAKTHROUGH (measured 2026-08-18)
KAT_NO_BIG_OFFLOAD guard (>32MiB host weights never offloaded — kills the
69GB per-copy expert materialization) + KAT_PIPELINE=1:
  pipeline ENGAGES (0 fallbacks), AR novel: 16.4 tg peak / 15.2 e2e
  vs 12.3 stock AR baseline = +33%. Split pipelining overlaps the
  82 GPU<->CPU syncs per forward. NO cuda graphs. Stable.
NEXT TEST: compose with pipelining (does the draft's target-forward
benefit equally?), then stack with CQ4 when its build lands.

## V57 USER IDEA EVAL: LAYER LOOPING + LOW-RANK EXPERTS (2026-08-18)
Proposal: 20 layers + loop x2, experts low-rank.
MATH: looping saves ZERO bytes/token (re-reads same weights per pass;
bottleneck is bytes not residency). 20L@Q3 ~= 8.8GB still > 8GB VRAM.
10L@Q3 x4 loops fits (~5GB) and would be ~1ms expert reads at 448GB/s
BUT 10-layer depth-sharing on a non-loop-trained model = word salad
(own Nanbeige campaign: even loop-trained model needed per-loop LoRA +
gating; KAT retraining violates no-big-training rule). Low-rank experts:
measured dead end in E0-era work (SVD class loses to quant sub-4-bit;
experts tolerate factorization worse than dense).
SURVIVING VARIANT — HOT-EXPERT VRAM CACHE (queued next after CQ4):
  expert usage is heavy-tailed; pin top ~2-3GB of hot experts in
  leftover VRAM (60GB/s x unused headroom... no, 448GB/s VRAM read),
  cold in RAM. Covers est 40-60% of expert reads at VRAM speed with
  ZERO weight changes / zero retraining (WiSP/SMoE class from paper
  dump). Implementation: placement heuristic in -cmoe tensor overrides
  or engine-side expert pin table.

## V58 JETSPEC HEAD + LOAD RACE STATUS (2026-08-18)
JetSpec/jetspec-Qwen3.6-35B-A3B head converted (build_jetspec.py):
  8-layer causal draft, fc 41.9MB, our Q6_K quant (rel-err 0.0202
  verified), tokenizer spliced from Koopah draft, all fork metadata
  keys fixed (dflash arch, target_layers array, causal=true BOOL,
  bare tensor names fc.weight / enc.output_norm.weight).
FIRST RUN: acceptance 0.000 — diagnosed TWO semantic bugs in fork:
  (a) HF output_hidden[L] == llama layer_inp[L+1] (off-by-one capture)
  (b) fork hard-forces non-causal attn on dflash drafts; JetSpec head
      is causal_head:true
FIX (built, testing): KAT_JETSPEC env -> extract at id+1, causal on.
LOAD RACE: -ngld all configs hit 'invalid vector subscript' up to
100% of attempts (fit-path double-load of draft GGUF is non-
idempotent; VRAM state-dependent). BLOCKER for GPU-draft. CPU-draft
(dspark v2) bypasses it (~50% race, retry wins).
PPL GATE: rerun in flight (cq4 full log pending).

## V59 LOAD RACE ROOT CAUSE = RAM PRESSURE (2026-08-18)
The 'invalid vector subscript' draft-load crash correlates with FREE RAM
(4.1GB free = ~100% crash; 9GB+ = ~50%; 15GB+ = rare). NOT the measurement
pass (KAT_NO_MEASURE confirmed skipping it; still crashed). The dual-load
page-cache under memory pressure trips a vector bounds bug in the fork's
loader. Operational rule: >10GB free before -md launches; kill PPL/builds
first. All earlier "race" observations consistent with this.
QUEUE (memory-gated): (1) PPL CQ4 gate (~1h); (2) JetSpec causal-head
test at high free RAM; (3) if acceptance >0.5 novel -> compose + bench.

## V60 CQ4 QUALITY GATE — FAILED, DISCARDED (measured 2026-08-18)
Same exe/corpus/flags: CQ3 (Q4_K experts) PPL 5.7066 +-0.046
                       CQ4 (Q3_K experts) PPL 7.3471 +-0.062
                       DELTA +1.64 (gate <= +0.05 per user's Q4K floor)
Q3_K expert requant is decisively off-floor. Two candidate causes:
(a) our Jacobi-vectorized q3k encoder converges worse than the C
    sequential RMSE refinement; (b) KAT experts are unusually
    Q3-sensitive. Either way: BYTE-CUT PATH CLOSED at current quality
    floor. Q4_K (4.5bpw) is the floor; IQ4_XS (~6% smaller) is not a
   lever. Deleted KAT-CQ4.gguf. Path to novel-100 now rests on:
   JetSpec 8-layer causal head (converted+patched, test queued),
   split-pipeline overlap (+33% proven AR), hot-expert VRAM cache.

## V61 JETSPEC HEAD — STILL 0.000 ACCEPTANCE (open, diagnosed trail)
Causal+off-by-one fixes built and active (KAT_JETSPEC), load race solved
(17.5GB free = 0 crashes, first try). Result: acceptance STILL 0.000.
Conversion verified (q6k rel-err 0.0202, layouts match Koopah working
head convention, tokenizer OK, metadata keys OK). Remaining suspects
(need fork-side draft-token dump to discriminate):
  - fork dflash graph vs JetSpec causal-head math (q_norm placement,
    per-head norm order, cross-attn mask shape at block edges)
  - fc input feature ORDER (5 layers x 2048 concat order)
  - noise-embed/mask-token semantics for causal mode
PATH: add KAT_SPEC_DEBUG-style draft-id dump to fork's speculative.cpp,
compare against expected tokens for a fixed prompt. ~1-2h work.
SESSION CLOSE STATE: best novel = dspark compose (17-22); best copy =
68.5 decode/59.5 e2e. CQ4 dead (V60). Split-pipeline +33% proven (b9873
binary). JetSpec head = highest-value open thread.
