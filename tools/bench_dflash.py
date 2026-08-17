#!/usr/bin/env python
# Decode-rate bench for dflash_server (OpenAI-compatible).
# PURPOSE: measure sustained decode tok/s for KAT-CQ1 across configs.
# Usage: python bench_dflash.py --url http://127.0.0.1:8000 [--n 3]
import argparse, json, time, urllib.request

def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body

PROMPTS = [
    "Write a Python function implementing binary search with type hints and a docstring, then explain its complexity.",
    "Implement a LRU cache class in Python with get and put methods, O(1) operations, using an ordered dict.",
    "Write a CUDA kernel skeleton for vector addition and explain memory coalescing considerations.",
    "Refactor this SQL to use window functions instead of correlated subqueries: SELECT u.name, (SELECT COUNT(*) FROM orders o WHERE o.user_id=u.id) FROM users u;",
    "Explain the difference between Mixture-of-Experts routing load balancing loss and router z-loss, with code sketches for both.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=1, help="repeats per prompt")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    results = []
    for i, p in enumerate(PROMPTS):
        for rep in range(args.n):
            dt, body = post(args.url, {
                "model": "dflash",
                "messages": [{"role": "user", "content": p}],
                "temperature": args.temp,
                "max_tokens": args.max_tokens,
                "stream": False,
            })
            u = body.get("usage", {})
            ct = u.get("completion_tokens", len(body["choices"][0]["message"]["content"]) // 4)
            tps = ct / dt if dt > 0 else 0
            results.append((i, rep, ct, dt, tps, json.dumps(u.get("timings", {}))))
            print(f"p{i}.{rep}: {ct} tok in {dt:.1f}s = {tps:.1f} tok/s  timings={u.get('timings',{})}")

    tps_list = [r[4] for r in results]
    tps_list.sort()
    med = tps_list[len(tps_list)//2]
    print(f"\n== median {med:.1f} tok/s | min {tps_list[0]:.1f} | max {tps_list[-1]:.1f} | n={len(tps_list)} ==")

if __name__ == "__main__":
    main()
