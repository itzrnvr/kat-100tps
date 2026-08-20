# CQ-Ornith build plan (V113 target)

## Inputs (all local or streamable)
- `D:/merge/out/Ornith15-Q4KM.gguf` — official Q4_K_M (21.7GB): trunk experts
  byte-copy donor + PPL reference
- `https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/resolve/main/model-*.safetensors`
  (16 shards, 71.9GB total, NEVER downloaded whole) — control plane via
  e0.Reader range streaming
- `D:/merge/out/orn-mtp-head-bf16.bin` — native MTP head blk.40.* in GGUF
  tensor order, BF16 weights + F32 norms (verified sane: median|w| .0074,
  100% finite/nonzero — NOT the stale-norm pattern that killed KAT's donor)

## Precision map (CQ3 recipe, proven on KAT at PPL 3.6952)
- routed experts ffn_{gate,up,down}_exps: Q4_K byte-copied from official
- attention projections, shared experts: F16 from BF16
- norms, routers (ffn_gate_inp, gate_inp_shexp), dt_bias/A_log: F32 from BF16
- embeddings/output: byte-copy from official
- MTP head blk.40.*: matmuls -> Q8_0, norms F32, experts Q4_K (CQ3 map)
  — bytes already in GGUF order, only quantize matmul tensors

## Steps
1. orn_local_scan.py on official -> tensor inventory (confirm arch variant:
   hybrid DeltaNet linear-attn vs full attention, per cp_pull mapping)
2. Stream control plane (cp_load-style), quantize per map
3. Emit new GGUF: official header, patched tensor dir + block_count 41 +
   nextn_predict_layers=1, data = control-plane re-quant + experts byte-copy
   + head tensors
4. superweight_audit.py on the result
5. PPL gate vs official Q4_K_M on wiki.test.raw (same corpus as KAT gates)
6. orn_bench.py --resident on the CQ build -> V113

## Disk ledger
- official 21.7 + CQ out ~20-21 -> peak ~43GB on 45.6GB free (tight, ok)
- delete official AFTER PPL+bench if space needed

## V112c corrections (measured)
- KAT vocab is ALSO 248,320 (same qwen35 tokenizer) — vocab-size hypotheses dead.
- Type map deltas (KAT-CQ3 vs official Ornith):
  output.weight Q8_0 vs Q6_K | token_embd Q8_0 vs Q4_K | attn_output Q8_0 vs Q4_K
- After V114 requant patch (21 down_exps Q6_K->Q4_K), Ornith expert bytes
  ~= KAT's 18.6GB; residual speed gap (if any) to be split-time profiled,
  not speculated.
- Output/embd live on GPU in both configs (-ngl 99 -cmoe, ~1.8GB non-expert
  fits 8GB VRAM) — logits GEMV not a CPU-bus factor.
