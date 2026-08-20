# PURPOSE: Parallel HTTP-range downloader, direct-offset writes (no 2x disk).
# v2 fixes: pardl.py's Unicode-subscript corruption; parts-dir 2x-disk assembly;
#            os.pwrite (absent on Windows) -> per-thread fd + seek/write.
# Resume: sidecar JSON tracks completed chunks; chunks are written into the
# preallocated output file at their final offsets.
# USAGE: python pardl2.py <url> <out_path> <total_bytes> [workers]
import os, sys, time, json, threading, urllib.request

URL = sys.argv[1]
OUT = sys.argv[2]
TOTAL = int(sys.argv[3])
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 16
CHUNK = 32 << 20  # 32MB

n_chunks = (TOTAL + CHUNK - 1) // CHUNK
side = OUT + ".pstate.json"

done = [False] * n_chunks
if os.path.exists(side):
    with open(side) as f:
        for i in json.load(f):
            done[i] = True
bytes_done = sum(min(CHUNK, TOTAL - i * CHUNK) for i in range(n_chunks) if done[i])

if not os.path.exists(OUT) or os.path.getsize(OUT) != TOTAL:
    with open(OUT, "wb") as f:
        f.truncate(TOTAL)  # preallocate

lock = threading.Lock()

def persist():
    with open(side + ".tmp", "w") as f:
        json.dump([i for i in range(n_chunks) if done[i]], f)
    os.replace(side + ".tmp", side)

def worker():
    global bytes_done
    fh = open(OUT, "r+b")  # per-thread handle; Windows has no os.pwrite
    try:
        while True:
            with lock:
                idx = next((i for i in range(n_chunks) if not done[i]), None)
                if idx is None:
                    return
                done[idx] = True  # claim
            s = idx * CHUNK
            e = min(s + CHUNK, TOTAL) - 1
            for attempt in range(6):
                try:
                    req = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{e}"})
                    data = urllib.request.urlopen(req, timeout=180).read()
                    if len(data) != e - s + 1:
                        raise IOError(f"short read {len(data)}")
                    fh.seek(s)
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())  # V115a: sidecar must never claim
                    # a chunk whose bytes are not physically on disk
                    with lock:
                        bytes_done += len(data)
                        persist()
                    break
                except Exception as ex:
                    if attempt == 5:
                        print(f"chunk {idx} failed: {ex}", flush=True)
                    time.sleep(2 * (attempt + 1))
            else:
                with lock:
                    done[idx] = False  # release for a later resume pass
                return
    finally:
        fh.close()

threads = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
t0 = time.time()
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    time.sleep(20)
    el = time.time() - t0
    print(f"{bytes_done/1e9:.2f}/{TOTAL/1e9:.2f} GB  {bytes_done/1e6/el:.1f} MB/s", flush=True)

missing = [i for i in range(n_chunks) if not done[i]]
if missing:
    print(f"MISSING {len(missing)} chunks: rerun to resume", flush=True)
    sys.exit(1)
os.remove(side)
print(f"DONE {os.path.getsize(OUT)} bytes", flush=True)
