# Measured Ledger — every benchmark run of the campaign

All on: RTX 3070 Ti Laptop 8GB / 31GB DDR5 / Windows / llama.cpp b4311 or
lucebox dflash_server (patched, this repo). n=5 unless noted. tok/s = decode
median of the run; prompts from tools/bench_dflash.py (5 coding prompts).

## Model artifacts (all built this campaign)

| Artifact | Recipe | Size | Notes |
|---|---|---|---|
| KAT-CQ1 | experts Q4_K RTN from KAT BF16, control F16 | 20.05 GiB | crashes stock CUDA (binbcast assert on GDN F16); PPL 5.0748 (beats stock Ornith 5.1486) |
| **KAT-CQ2** | same experts, control Q8_0/F32 official-parity | 19.38 GiB | **current best**; stock-clean; beats official quant on both axes |
| KAT-CQ2-MTP | CQ2 + grafted byteshape MTP block (blk.40) | 19.92 GiB | loads everywhere; drafting engages; see stock section |
| kat-mtp-shexp-draft | surgical DFlash draft from MTP block | 41 MB | dimensionally valid; acceptance limited (see draft section) |

## Stock llama.cpp runs

| Config | Result | Notes |
|---|---|---|
| llama-bench tg64 fa1 ngl99 (official Q4_K_M base) | 6.30 t/s | reference floor |
| llama-bench tg64 fa1 ngl99 (KAT-CQ2) | 6.78 t/s | our quant +7.6% |
| server AR -ncmoe 8 | 7.4 med | |
| server AR -ncmoe 16 | 9.2 med | |
| server AR -ncmoe 24 | 10.5 med | monotone toward CPU experts |
| server AR -cmoe t8 (n=10) | **12.3 med / 12.9 max** | threads swept: t16=8.8 t12=10.5 t4=7.1 t2=5.1 t1=4.8 |
| + dflash+ngram-simple n5 | 15.9 med / 18.9 max | |
| + ngram-map-k4v | 15.3 med | |
| + ngram-cache+simple | 15.2 med | variant differences < noise on cold short prompts |
| + embedded MTP (KAT-CQ2-MTP, draft-mtp n4) | **8.9 med** | acceptance 36–61% but SLOWER than AR — verify fragmentation; MTP block own-gather ≈14MB/draft-token minor term |
| agent+tools (15K prefix) first turn | 1.8 e2e | 92s prefill |
| agent+tools warm turns | 12.7–18.7 | prefix cache → 9s prefill |

## lucebox dflash_server runs (patched)

| Config | Result | Notes |
|---|---|---|
| target-only, first attempt | crash | metadata-arena OOM (patch 001) |
| target-only, arena-fixed | crash on draft path | second arena family + heap fragmentation (patch 002) |
| target-only + q8-27B-dense draft | 12.5 med (n=5, incl. cold) | draft 6.2% accept → self-disables |
| target-only (warm, n=10) | 17.7 med / 22.0 peak | |
| + shexp surgical draft (accept 6.2% → auto AR) | **22.4 med / 24.3 peak (n=10)** | **best config to date** |

## Thread-scaling decomposition (stock, t-sweep t1..t16)

b + c = 208ms (t1), b + c/8 = 80ms (t8) → **bandwidth-bound 62ms (78%)**,
compute 18ms. 566MB/62ms = **9.1GB/s effective vs ~55GB/s available**.

## Spec-decode acceptance (all heads tried)

| Head | Acceptance | Engine | Verdict |
|---|---|---|---|
| Lucebox q4_k_m 27B-dense | 6.2% | lucebox | head mismatch |
| Lucebox q8_0 27B-dense (redownloaded, size-verified) | 6.2% | lucebox | precision was not the issue |
| byteshape embedded MTP (3.53bpw model) | **61–68%, len 3.4–3.7** | stock draft-mtp | head validated; stock verify path wastes it |
| KAT-CQ2-MTP (grafted) | 36–61% | stock | works; slower than AR on stock |
| kat-mtp-shexp-draft (surgical) | 6.2% | lucebox | loads+runs cleanly; too weak alone (shexp-only) |

## Crash/root-cause log (all fixed, tests in patches)

1. CQ1 F16 control plane → binbcast assert (GDN path) → CQ2 Q8_0 control.
2. 512MB×~60 no_alloc metadata arenas → 30GB commit at first request → 32MB.
3. 5 more 512MB arenas in graph_builders.cpp (same class).
4. _aligned_malloc fragmentation failure (0.12GiB, 12.5GB free) → VirtualAlloc fallback.
5. Truncated Q8 draft download (curl exit 0, 1.26/1.84GB) → size-verified loop.
6. GGUF KV type-code traps (u32-vs-u8 pair, F32-vs-F64 freq_base) in graft.
7. Q8_0 block size = 32 (build_gguf helper assumed 256 → ⅛ sizes).
8. Q‖gate per-head interleave (not concat halves) in donor attn_q.
9. DFlash fc consumes n_capture×n_embd features; capture-ids override + N=1 fix.
10. LNK1104: running exe holds file (kill before relink).

## 2026-08-18 — AcceptMoE-style verify-union restriction (KAT_UNION_K) — NEGATIVE
- Same-binary A/B, CQ3+gbuzhf recipe, n=5/workload, e2e rates:
  off: copy 16.3 / pattern 20.8 | K=24: copy 14.4 / pattern 19.6
- Draft acceptance unchanged (0.75-0.79 copy) -> not quality-bound;
  overhead-bound. Natural union already compact at our tree sizes.
- Patch stays in C:/src/lm9873 (env-gated, default off). DESIGN.md V36.

## 2026-08-18 — DSpark v2 (Koopah Qwen3.6-35B-A3B head) on KAT-CQ3 — BREAKTHROUGH
- Cross-model transfer works: acceptance 0.38-0.52, mean accepted len 4.0-5.2 (block 8)
- e2e ab_union (n=5): copy 37.7 med / 39.2 peak (vs 16.3 stock-spec = +131%)
- 3.1x over stock AR baseline (12.3); lucebox AR (22.4) beaten decisively
- Union-v2 restriction: safe w/ warmup but net-neutral at K=24/48; cold-K24
  spiked acceptance to 0.81 + peaks 44.4 but derailed generation (recorded)
- Engine: satindergrewal/llama.cpp@dspark-qwen35 + our KAT_UNION patches

## 2026-08-18 final — DSpark campaign complete
- compose (dspark+ngram w8): copy 47.4 med/52.2 peak e2e; decode tg peaks 53
- pure dspark w3 novel: 17.0 tg (w8: 10.6) — width policy per workload
- cost model: 17ms draft + 33ms/verify-row; 100tps = 3x verify-row cost cut
- quality: coherent, NOT bit-exact vs AR (KV-inject numerics); documented
- all negatives recorded (ngld-all, conf-min, union-v2 ladder, width-null
  confound identified + deconfounded)
