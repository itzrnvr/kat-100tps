#!/usr/bin/env python
"""TRUE copy-heavy protocol (the original 46.5-protocol class): the model
must reproduce a large code file VERBATIM. ngram-mod acceptance -> ~1.0.
Measures decode-only t/s from server timing (max_tokens completion).
Usage: python bench_truecopy.py [port] [n]
"""
import json, statistics, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8035"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

CODE = "\n".join(
    f"def process_batch_{i}(items, config=None):\n"
    f"    cfg = config or {{}}\n"
    f"    timeout = cfg.get('timeout', 30)\n"
    f"    retries = cfg.get('retries', 3)\n"
    f"    results = []\n"
    f"    for item in items:\n"
    f"        if item.get('skip'):\n"
    f"            continue\n"
    f"        try:\n"
    f"            res = handle(item, timeout=timeout)\n"
    f"            results.append({{'id': item['id'], 'ok': True, 'res': res}})\n"
    f"        except TransientError:\n"
    f"            for attempt in range(retries):\n"
    f"                try:\n"
    f"                    res = handle(item, timeout=timeout)\n"
    f"                    results.append({{'id': item['id'], 'ok': True, 'res': res, 'retry': attempt}})\n"
    f"                    break\n"
    f"                except TransientError:\n"
    f"                    if attempt == retries - 1:\n"
    f"                        results.append({{'id': item['id'], 'ok': False, 'error': 'retries_exhausted'}})\n"
    f"        except FatalError as e:\n"
    f"            results.append({{'id': item['id'], 'ok': False, 'error': str(e)}})\n"
    f"    return results\n"
    for i in range(30)
)

PROMPT = (f"Here is the batch processing module:\n```python\n{CODE}\n```\n"
          f"Reproduce the ENTIRE file above verbatim, character for character, "
          f"then append process_batch_30 in identical style.")

def run():
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 1400, "temperature": 0,
                       "stream": False}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return ct, dt

if __name__ == "__main__":
    run()  # warm
    rates = []
    for i in range(N):
        ct, dt = run()
        r = ct / dt
        rates.append(r)
        print(f"truecopy[{i}] gen={ct} {dt:.1f}s {r:.1f} t/s e2e")
    print(f"MEDIAN {statistics.median(rates):.1f}  PEAK {max(rates):.1f}")
