# Loss Budget — measured decode decomposition

Method: parallel-speedup decomposition from thread scaling (t1..t16), verified
against direct all-CPU run and per-request engine telemetry. All runs n=5+
except where noted. Hardware: 3070 Ti Laptop 8GB, 2×16GB DDR5-4800, sm_86.

## The budget at 22.4 tok/s (44.7ms/token)

| Layer | ms/tok | Method | Notes |
|---|---|---|---|
| Ideal expert fetch (566MB @ 55GB/s) | 10.3 | arithmetic | ceiling ≈ 97 tok/s |
| **Scattered-gather penalty** | **+52** | t1=4.8/t8=12.5 → b=62ms fixed, of which 10 ideal | 9.1GB/s effective = ⅙ bandwidth |
| Compute (GDN+attn+FFN FLOPs) | 18 | same decomposition (c/8 term) | only 22% of budget |
| Total measured (stock AR) | ~81 | 12.3 tok/s | |
| Pipelined overlap recovery | −36 | lucebox 22.4 t/s | fetch overlapped w/ compute |

## Verbatim run ledger (key rows)

```
stock llama-bench AR (fa1 ngl99)         tg64  6.78   (official quant: 6.30)
stock server AR -ncmoe 8/16/24/45        med   7.4 / 9.2 / 10.5 / 12.1
stock AR t8 clean (n=10)                 med   12.3  (t16=8.8, t4=7.1, t2=5.1, t1=4.8)
stock +dflash+ngram (all variants)       med   15.2–15.9, peak 18.9
stock +embedded MTP (byteshape head)     med   8.9   accept 61–68%, len 3.4–3.7
agent+tools warm prefix                  med   12.7–18.7  (first turn 1.8: 92s prefill)
lucebox pipelined AR (target-only)       med   17.7  peak 22.0  (n=10)
lucebox + shexp draft (auto-disabled)    med   22.4  peak 24.3  (n=10)  ← BEST
```

## The three structural blockers (each measured, not inferred)

1. **Gather fragmentation**: 320 scattered expert GEMVs/token. Effective
   bandwidth 9.1GB/s vs 55 available. Fix: whole-tier residency (no fetch),
   Fate-style prefetch (hide fetch), or batched verify (amortize fetch).
2. **Verify fragmentation**: 4.5-token spec batch → M≈1.4 rows/expert →
   no GEMM amortization; stock+MTP regressed BELOW AR (8.9 vs 12.3) despite
   61–68% acceptance. (MTP block's own gather ≈14MB/draft-token: minor term.)
   Fix: NInfer-style persistent batched MoE kernels.
3. **No batchable family draft**: every available head measured —
   27B-dense 6.2%, surgical shexp 6.2%, byteshape MTP 61–68% (non-batching
   path only). Fix: train the 41MB fc + draft jointly on traces.

## Why the fixes stack to 100+

- (a) trained draft @ 3.5 tok/round × pipelined verify → 40–60 class
- (b) batched verify kernels remove blocker 2 → verify-cost ≈ AR-cost
- (a)+(b): 22.4 × ~2.5–3 effective → 55–65
- (c) tier residency + Fate prefetch recovers most of the 52ms →
  AR base itself → 35–45 → stacked with (a)(b) → 100+
Each step independently measurable; none assumed.
