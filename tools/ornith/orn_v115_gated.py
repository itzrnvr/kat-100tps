# PURPOSE: V115 PPL gate, machine-respect edition. Waits until the machine
# is genuinely free (zero llama-server processes AND >=20GB RAM), then runs
# the two remaining PPL legs serially and prints the verdict inputs.
# Founded after the RAM-starvation incident: never launch big-model PPL legs
# while the user's llama-server is up or RAM is tight.
import ctypes, os, re, subprocess, sys, time

PPL_EXE = r"D:\src\gypsy-dragon\build\bin\Release\llama-perplexity.exe"
CORPUS = r"D:\merge\E0\wiki.test.raw"
OFFICIAL = r"D:\merge\out\Ornith15-Q4KM-REF.gguf"
CQ = r"D:\merge\out\Ornith15-Q4K-CQ.gguf"
OFF_LOG = r"D:\merge\train\ppl_ornOFFICIAL.log"
CQ_LOG = r"D:\merge\train\ppl_ornCQ.log"

def free_ram_gb():
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in
                    ("ullTotalPhys","ullAvailPhys","ullTotalPageFile","ullAvailPageFile",
                     "ullTotalVirtual","ullAvailVirtual","ullAvailExtendedVirtual")]
    st = M(); st.dwLength = ctypes.sizeof(M)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return 0.0
    return st.ullAvailPhys / 1e9

def servers_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"],
                         capture_output=True, text=True).stdout
    return "llama-server" in out

def final_ppl(log):
    txt = open(log, "rb").read().decode("utf-8", "replace")
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", txt, re.I)
    return float(m.group(1)) if m else None

def leg_done(log):
    return final_ppl(log) is not None

def run_leg(model, log, label):
    print(f"[{label}] starting", flush=True)
    with open(log, "wb") as lf:
        subprocess.run([PPL_EXE, "-m", model, "-f", CORPUS,
                        "-ngl", "99", "-cmoe", "-t", "12", "-c", "4096"],
                       stdout=lf, stderr=subprocess.STDOUT, timeout=3500)
    # sanity: first chunks must be sane
    txt = open(log, "rb").read().decode("utf-8", "replace")
    first = re.findall(r"\[\d+\]([\d.]+)", txt)[:2]
    vals = [float(v) for v in first]
    if vals and max(vals) > 100:
        print(f"[{label}] GARBAGE chunks {vals} — aborting chain", flush=True)
        sys.exit(1)
    print(f"[{label}] done: PPL = {final_ppl(log)}", flush=True)

# stage 0: wait for a free machine
print("[gate] waiting for free machine (no llama-server, RAM >= 20GB)...", flush=True)
while True:
    if not servers_running() and free_ram_gb() >= 20.0:
        break
    time.sleep(120)

# safety re-check after a settle period
time.sleep(60)
if servers_running() or free_ram_gb() < 18.0:
    print("[gate] state flapped — continuing to wait", flush=True)
    while True:
        if not servers_running() and free_ram_gb() >= 20.0:
            break
        time.sleep(120)

print(f"[gate] machine free (RAM {free_ram_gb():.1f} GB) — starting legs", flush=True)

if not leg_done(OFF_LOG):
    run_leg(OFFICIAL, OFF_LOG, "official")
else:
    print(f"[official] already done: {final_ppl(OFF_LOG)}", flush=True)

if not leg_done(CQ_LOG):
    run_leg(CQ, CQ_LOG, "cq")
else:
    print(f"[cq] already done: {final_ppl(CQ_LOG)}", flush=True)

off = final_ppl(OFF_LOG)
cq = final_ppl(CQ_LOG)
print("\n===== V115 VERDICT INPUTS =====", flush=True)
print(f"KAT-CQ3 (context):  6.9831", flush=True)
print(f"Ornith official:    {off}", flush=True)
print(f"Ornith CQ:          {cq}", flush=True)
if off and cq:
    d = cq - off
    v = "PASS" if d <= 0.05 else ("MARGINAL" if d <= 0.30 else "FAIL")
    print(f"requant delta: {d:+.4f} -> {v}", flush=True)
print("V115 CHAIN DONE", flush=True)
