#!/usr/bin/env python
# Build a CCT (cross-layer co-activation table) for Fate prefetch from a
# routing trace collected with --collect-routing / DFLASH_COLLECT_ROUTING.
#
# PURPOSE: P(experts at layer L+1 | experts at layer L) — offline LUT that
# Fate unions with its cross-layer-gate prediction (RLCG hybrid, v20).
#
# Trace format (MoeRoutingCollector): per sample
#   int32 layer_idx, int32 K, float32[n_embd] hidden, int32[K] expert_ids
# Samples arrive per token per layer in layer order (decode loop visits
# layers 0..N-1 sequentially per token).
#
# Output (fate_prefetch.h load_cct format):
#   int32 n_layers; per layer: int32 n_entries;
#   entries: {int32 cur, int32 cnt, int32 next[cnt]}
import struct, sys, os
from collections import defaultdict

def main():
    trace, out = sys.argv[1], sys.argv[2]
    max_next = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    trans = defaultdict(lambda: defaultdict(int))  # layer -> (cur,nxt)->count
    n_layers_seen = 0
    prev_layer, prev_ids = -1, []
    n_tok = 0
    n_embd = 2048  # KAT-CQ1; assert from first sample below

    size = os.path.getsize(trace)
    with open(trace, "rb") as f:
        # sanity: first sample header tells K; we trust n_embd=2048 for KAT
        while f.tell() < size:
            hdr = f.read(8)
            if len(hdr) < 8: break
            layer_idx, K = struct.unpack("<ii", hdr)
            f.seek(f.tell() + 4 * n_embd)
            ids = struct.unpack(f"<{K}i", f.read(4 * K))
            n_layers_seen = max(n_layers_seen, layer_idx + 1)
            if layer_idx == 0 and prev_layer >= 0:
                n_tok += 1
            if prev_layer >= 0 and layer_idx == prev_layer + 1:
                for c in prev_ids:
                    for nx in ids:
                        trans[layer_idx - 1][(c, nx)] += 1
            prev_layer, prev_ids = layer_idx, list(ids)
    print(f"tokens~{n_tok}, layers={n_layers_seen}")

    with open(out, "wb") as o:
        o.write(struct.pack("<i", n_layers_seen))
        for l in range(n_layers_seen):
            by_cur = defaultdict(list)
            for (c, nx), cnt in trans[l].items():
                if cnt < 2: continue
                by_cur[c].append((nx, cnt))
            entries = 0
            blobs = []
            for c, lst in sorted(by_cur.items()):
                lst.sort(key=lambda t: -t[1])
                keep = [nx for nx, _ in lst[:max_next]]
                if not keep: continue
                entries += 1
                blobs.append(struct.pack(f"<ii{len(keep)}i", c, len(keep), *keep))
            o.write(struct.pack("<i", entries))
            for b in blobs: o.write(b)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
