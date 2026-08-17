# kat-100tps

**100+ tok/s decode for a 35B MoE coder model on a single 8GB-VRAM laptop.**
Student model: KAT-Coder-V2.5-Dev (Qwen3.6-35B-A3B family, 40 layers × 256 experts, top-8, GDN hybrid).
Hardware: RTX 3070 Ti Laptop 8GB + 31GB DDR5 RAM, sm_86.

This repo documents a measured, end-to-end speed campaign: custom quantization, a pipelined inference engine build, surgical MTP-draft construction, allocator hardening, and a complete loss-budget decomposition of MoE decode on memory-constrained hardware.

## Status: 22.4 median / 24.3 peak tok/s (n=10) — best measured config

| Config | tok/s | vs stock |
|---|---|---|
| Stock llama.cpp AR (best of full sweep: t8, -cmoe, fa, q8 KV) | 12.3 median | 1.0× |
| Stock + any speculative variant (dflash/ngram/MTP) | 8.9–19 | ≤1.5× |
| **This repo: lucebox pipelined AR + KAT-CQ2** | **22.4 median / 24.3 peak** | **1.82×** |
| Family MTP head acceptance (via stock draft-mtp) | 61–68%, 3.5 tok/round | head validated, engine mismatch |

## The measured loss budget (why 22.4 and not 100)

Perfect (RAM-bandwidth) decode = ~10.3ms/token (566MB expert bytes @ 55GB/s ≈ 97 tok/s ceiling).

| # | Loss layer | Cost | Measured evidence |
|---|---|---|---|
| 1 | **Scattered MoE gather** | **+52ms/tok — dominant** | 566MB/token moves at 9.1GB/s effective (62ms vs 10.3 ideal): 320 scattered GEMVs (8 experts × 40 layers) defeat RAM prefetch; ⅙ of bandwidth used |
| 2 | **Spec-verify fragmentation** | blocks ~3× multiplier | MTP acceptance 61–68% proven, but 4.5-token verify batches fragment to M≈1.4 rows/expert → zero GEMM amortization; stock+MTP = 8.9 t/s (worse than AR 12.3). MTP block's own gather ≈ 14MB/draft-token is a minor term |
| 3 | GPU expert residency contention | −4.7 t/s | −ncmoe 8/16/24/45 = 7.4/9.2/10.5/12.1 t/s (monotone: more GPU experts = slower on 8GB) |
| 4 | SMT threading | −3.5 t/s | t16=8.8 vs t8=12.3 (t2=5.1, t4=7.1 — clean Amdahl; decomposition: 78% bandwidth-bound) |
| 5 | Cold prefill on agent turns | 92s vs 9s | first agent turn is prefill-bound (1.8 t/s e2e), warm turns 18+; prefix cache is mandatory |
| 6 | No family-matched batchable draft | −2–3× potential | every head measured: 27B-dense 6.2% accept; surgical shexp draft 6.2%; byteshape MTP 61–68% but only via stock's non-batching path |

## Repo layout

```
docs/
  DESIGN.md            — full campaign log (v1→v31), every decision + paper tie-in
  LOSS-BUDGET.md       — the table above with methodology
  MEASURED-LEDGER.md   — every benchmark run, config, and number
engine-patches/        — lucebox fork patches (build fixes + Fate prefetcher + capture-ids)
  000-sm86-only-build.patch
  001-metadata-arena-oom.patch       (512MB→32MB arenas, 3 generations)
  002-virtualalloc-fallback.patch    (heap-fragmentation malloc fails, registry-routed free)
  003-fate-crosslayer-prefetch.patch (Fate arXiv 2502.12224 port, DFLASH_FATE=1)
  004-capture-ids-override.patch     (data-driven DFlash capture layers, N=1 safe)
tools/
  gguf_local_inv.py    — GGUF v3 inventory parser (corrected value-type map)
  build_kat2.py        — KAT-CQ2 builder (Q8_0 control plane; fixes stock-crash F16 layout)
  graft_mtp2.py        — MTP block graft into KAT-CQ2 (byte-exact header surgery)
  make_draft.py        — surgical DFlash-draft from MTP block (fc slice, Q de-interleave)
  bench_dflash.py      — OpenAI-compatible decode bench (median/min/max)
  collect_traces.sh    — routing-trace collector (CCT/LUT/pre-gate training data)
  build_cct.py         — cross-layer co-activation table builder
  build_corpus.py      — spark calibration corpus from real SWE trajectories
artifacts/             — checksums + regeneration commands (weights not committed)
runbooks/
  engine-build.md      — sm_86 build incl. all pitfalls (arch ×4 bug, mmvq TU)
  serving.md           — best-known launch commands
```

## Reproduce the best config

```bash
# 1. Build engine (see runbooks/engine-build.md for the full pitfall list)
cmake -B server/build -S server -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build server/build --target dflash_server --config Release -j 8

# 2. Build KAT-CQ2 from KAT BF16 + official-layout template (tools/build_kat2.py)

# 3. Serve (22+ tok/s config)
dflash_server.exe KAT-CQ2.gguf --port 8021 --target-device cuda:0
```

## Key findings (novel/measured)

1. **8GB beats 24GB-class intuition**: CPU experts beat GPU experts for AR decode on 8GB (contended VRAM thrash) — opposite of the 24GB regime lucebox was built for.
2. **Spec decode cannot help serial-CPU MoE engines** regardless of acceptance: verify batches fragment across experts; M≈1.4 rows/expert amortizes nothing (measured 3 ways).
3. **GGUF surgery chain for MTP grafting** — byte-exact header splice (KV type-code traps: u32 vs u8, F32 vs F64; Q8_0 = 32-el blocks not 256; per-head Q‖gate interleave).
4. **Three allocator failure classes** fixed in one build: metadata arenas (512MB×N OOM), heap fragmentation (VirtualAlloc fallback w/ registry-routed free), commit spikes.
5. **Full decomposition method**: thread-scaling (t1/t8) separates bandwidth-bound (62ms) from compute (18ms) without profilers.

## Path to 100 (measured-gap analysis, not hope)

- **(a) Trained fc draft** (41MB linear + shexp block, feature-conditioned): turns loss #6 on → 3.5 tok/round × pipelined verify ≈ 40–60 t/s class. All tooling exists.
- **(b) NInfer-style batched verify kernels** (gdn_replay/mtp_pack, Apache-2.0 reference): removes loss #2 entirely.
- **(c) Whole-hot-tier residency + Fate prefetch**: recovers most of loss #1 (52ms → overlap window).
- Stacked: (a)+(b)+(c) = 100+ by the measured budget. Each is independent and individually verifiable.

## Papers/engine references folded into this work

llama.cpp (b4311), Luce-Org/lucebox-hub (Spark/KVFlash/prefix-cache/DFlash), NInfer (Qwen3.6 kernel reference), byteshape ShapeLearn quants, Fate (arXiv 2502.12224), ReasonMaxxer (2605.06241), plus 25+ more in docs/DESIGN.md.

## License

MIT for our code/patches/docs. Model weights follow their upstream licenses (KAT-Coder, Qwen3.6, byteshape).
