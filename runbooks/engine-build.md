# Engine build runbook — lucebox dflash_server on Windows sm_86

Every pitfall below was hit and fixed during this campaign. Follow in order.

## 1. Clone + configure

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub
cmake -B server/build -S server -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
```

**PITFALL 1 — always set CUDA_ARCHITECTURES=86 explicitly.** Default builds
4 architectures (70;75;86;120) = 4× compile time. The mmvq.cu TU alone took
>30 min per arch (cicc 600s + ptxas 1000s each). sm_86-only cut the tail
from ~2h to ~25min.

**PITFALL 2 — CURL not found is harmless** (proxy passthrough off).

## 2. Apply engine patches (engine-patches/ in this repo)

000-sm86-only-build.patch        — CMake arch pin
001-metadata-arena-oom.patch     — 512MB no_alloc metadata arenas ×60 graphs
                                   = ~30GB → OOM on 31GB RAM at first request.
                                   32MB is 4× headroom over metadata needs.
002-virtualalloc-fallback.patch  — _aligned_malloc fails on fragmented heap
                                   (observed: 0.12GiB request failing with
                                   12.5GB free). VirtualAlloc fallback with a
                                   registry-routed free (64-entry).
003-fate-crosslayer-prefetch.patch — Fate (arXiv 2502.12224) cross-layer gate
                                   prefetcher. DFLASH_FATE=1 enables.
004-capture-ids-override.patch   — data-driven DFlash capture layers
                                   (DFLASH_CAPTURE_IDS=39). Fixes N=1 div-by-0.

## 3. Build

```bash
cmake --build server/build --target dflash_server --config Release -j 8
```

**PITFALL 3 — LNK1104 "cannot open dflash_server.exe"** = a running instance
holds the file. Kill dflash_server first.

**PITFALL 4 — psmux send-keys mangles `$?`** (prints literally). Use
`printf "EXIT_%%s\n" "$?"` inside the sent command.

## 4. Serve

```bash
dflash_server.exe KAT-CQ2.gguf --port 8021 --target-device cuda:0
```

Hot/cold split auto-fits VRAM (~2000 hot / 8200 cold experts on 8GB).
Expect: `[qwen35moe] pipelined decode path active`, ~45s cold load.

## 5. Known-good bench

```bash
python tools/bench_dflash.py --url http://127.0.0.1:8021 --n 2 --max-tokens 256
```
Expect ~22 median / 24 peak (see docs/LOSS-BUDGET.md).
