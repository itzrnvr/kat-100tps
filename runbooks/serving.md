# Serving runbook — best-known configurations

## Best (22.4 med / 24.3 peak) — lucebox pipelined AR

```bash
D:/lucebox/server/build/Release/dflash_server.exe \
  D:/merge/out/KAT-CQ2.gguf \
  --port 8021 --target-device cuda:0
```

Optional safe draft (auto-disables at low acceptance, adds nothing, costs nothing):
```
  --draft C:/merge/kat-mtp-shexp-draft.gguf --draft-residency request-scoped
  # with: DFLASH_CAPTURE_IDS=39
```

Optional Fate prefetch A/B:
```
  DFLASH_FATE=1   # cross-layer gate prefetcher (see patch 003)
```

## Stock llama.cpp reference (12.3 best AR)

```bash
llama-server.exe -m D:/merge/out/KAT-CQ2.gguf \
  -ngl 99 -cmoe -fa on -ctk q8_0 -ctv q8_0 -c 16384 -b 2048 -ub 512 \
  -t 8 --cache-reuse 256 --cache-prompt --port 8031
```
Key flags measured: t8 (NOT t16), -cmoe (NOT -ncmoe N<45), fa on, q8 KV.

## Verified-bad configs (don't repeat)

- any -ncmoe N < 45: slower (GPU expert contention on 8GB)
- t16/t12: SMT hurts GEMV (−29%)
- stock + MTP draft: 8.9 t/s — verify fragmentation, worse than AR
- spark --spark-vram on 8GB: synchronous swap stalls (24GB-class feature)

## Quality gate

KAT-CQ2: ssm_out Q8_0 median rel-err 0.68% vs transformed BF16 source;
PPL lineage via CQ1 (5.0748 Moby Dick, beats stock Ornith 5.1486).
