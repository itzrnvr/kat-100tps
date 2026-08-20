# PURPOSE: First-run bench of Ornith-1.5-35B-A3B (official Q4_K_M) vs KAT
# champion numbers, identical protocol: streaming tps, 1 warmup + 3 trials,
# 3 novel prompts + 2 copy prompts, dspark head + ngram-mod spec stack, t12.
# Self-gates: waits for download completion (pstate.json gone), then RAM gate
# (default 10GB free for mmap mode; 17.5 for resident).
# USAGE: python orn_bench.py [--ram-gate 10] [--trials 3] [--resident]
import argparse, json, os, subprocess, sys, time, urllib.request

MODEL = r"D:\merge\out\Ornith15-Q4KM.gguf"
DRAFT = r"D:\merge\out\kat-dspark-v2-q8.gguf"
DRAFT_FALLBACK = r"D:\merge\kat-dspark-v2-q8.gguf"
EXE   = r"D:\src\gypsy-dragon\build\bin\Release\llama-server.exe"
PORT  = 8037

NOVEL = [
    "Write an original short story about a lighthouse keeper who discovers the light has been guiding something unexpected.",
    "Explain quantum entanglement to a curious 10-year-old, then invent a brand-new analogy of your own.",
    "Design a completely new board game with unique mechanics, then describe one full turn of play.",
]
DOC = """The Aleph Project, Section 4.
The Aleph engine processes events through three distinct queues. The ingest queue
accepts raw events and stamps them with monotonically increasing sequence numbers.
The reorder queue buffers out-of-order events for up to 500 milliseconds. The
commit queue applies events to durable storage in strict sequence order.
Failure of any single queue does not halt the others; each maintains its own
watermark and its own recovery log. When the ingest queue fails, the reorder
queue continues to drain buffered events. When the reorder queue fails, ingest
events bypass reordering and are committed with a best-effort timestamp.
When the commit queue fails, both upstream queues apply backpressure and
stop accepting new events until storage recovers.
"""
COPY = [
    "Reproduce the following document exactly, then continue it in the same style:\n\n" + DOC,
    "Copy this document verbatim:\n\n" + DOC,
]

def free_ram_gb():
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMORYSTATUSEX(); st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 1e9

def stream_bench(prompt, n=256):
    body = json.dumps({"messages":[{"role":"user","content":prompt}],
                       "n_predict":n,"temperature":0.0,"stream":True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type":"application/json"})
    t_first = t_last = None; ntok = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: "): continue
            d = line[6:].strip()
            if d == b"[DONE]": break
            j = json.loads(d)
            c = j["choices"][0].get("delta", {})
            c = c.get("content") or c.get("reasoning_content")
            if c:
                now = time.time()
                if t_first is None: t_first = now
                t_last = now; ntok += 1
    return ntok/(t_last-t_first) if t_last and t_last > t_first else 0.0

def stop_all():
    subprocess.run(["powershell","-NoProfile","-c",
                    "Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue"],
                   capture_output=True)

def launch(args, log_path, extra_env=None):
    env = dict(os.environ)
    if extra_env: env.update(extra_env)
    for t in range(3):
        lf = open(log_path, "ab")
        p = subprocess.Popen([EXE]+args, env=env, stdout=lf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time()-t0 < 240:
            if p.poll() is not None: break
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                if json.load(r).get("status") == "ok":
                    return p
            except Exception: pass
            time.sleep(2)
        if p.poll() is None: p.kill()
        lf.close(); time.sleep(5)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ram-gate", type=float, default=10.0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--nmax", type=int, default=4)
    a = ap.parse_args()

    # gate 1: download complete
    side = MODEL + ".pstate.json"
    if os.path.exists(side):
        print("download incomplete — refusing to bench", flush=True); sys.exit(1)
    if not os.path.exists(MODEL):
        print("model missing", flush=True); sys.exit(1)
    # gate 2: RAM
    ram = free_ram_gb()
    print(f"free RAM {ram:.1f} GB (gate {a.ram_gate})", flush=True)
    if ram < a.ram_gate:
        print("RAM below gate — not firing", flush=True); sys.exit(1)
    draft = DRAFT if os.path.exists(DRAFT) else DRAFT_FALLBACK
    if os.environ.get("ORN_SPEC") == "mtp":
        args = ["-m", MODEL,
                "--spec-type", "draft-mtp",
                "-ngl", "99", "-cmoe", "-fa", "on",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-t", "12", "-c", "8192", "--port", str(PORT)]
    else:
        args = ["-m", MODEL, "-md", draft,
                "--spec-type", "draft-dspark,ngram-mod",
                "--spec-draft-n-max", str(a.nmax),
                "--spec-draft-p-min", "0.75",
                "--spec-ngram-mod-n-min", "8",
                "--spec-ngram-mod-n-max", "24",
                "--spec-ngram-mod-n-match", "48",
                "-ngl", "99", "-cmoe", "-fa", "on",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-t", "12", "-c", "8192", "--port", str(PORT)]
    if a.resident: args += ["--load-mode", "none"]

    stop_all(); time.sleep(4)
    p = launch(args, r"D:\merge\train\orn_serve.log")
    if not p:
        print("LAUNCH FAILED", flush=True); sys.exit(1)
    print("server up", flush=True)
    try:
        # warmup (page-in), discarded
        stream_bench(NOVEL[0])
        print("warmup done", flush=True)
        for t in range(a.trials):
            row = [round(stream_bench(pr),2) for pr in NOVEL] + \
                  [round(stream_bench(pr),2) for pr in COPY]
            print(f"trial{t+1}: novel={row[:3]} copy={row[3:]}", flush=True)
    finally:
        stop_all()

if __name__ == "__main__":
    main()
