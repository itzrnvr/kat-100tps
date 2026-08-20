# PURPOSE: V115 gate orchestrator — strictly serial, no disk-overlap.
# Runs after KAT-CQ3 PPL completes (waits for it): delete broken file ->
# re-download official -> official PPL -> CQ PPL -> print verdict inputs.
# Every stage waits for the previous; no parallel disk access ever.
import os, re, subprocess, sys, time

PPL_EXE = r"D:\src\gypsy-dragon\build\bin\Release\llama-perplexity.exe"
CORPUS = r"D:\merge\E0\wiki.test.raw"
KAT_LOG = r"D:\merge\train\ppl_KAT-CQ3.log"
BROKEN = r"D:\merge\out\Ornith15-Q4KM.gguf"
OFFICIAL = r"D:\merge\out\Ornith15-Q4KM-REF.gguf"
CQ = r"D:\merge\out\Ornith15-Q4K-CQ.gguf"
OFF_LOG = r"D:\merge\train\ppl_ornOFFICIAL.log"
CQ_LOG = r"D:\merge\train\ppl_ornCQ.log"
URL = "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-Q4_K_M.gguf"
SIZE = 21713462848

def ppl_done(log):
    txt = open(log, "rb").read().decode("utf-8", "replace")
    return "final estimate" in txt.lower() or "perplexity:" in txt and len(re.findall(r"\[\d+\]", txt)) >= 77

def final_ppl(log):
    txt = open(log, "rb").read().decode("utf-8", "replace")
    m = re.search(r"final estimate.*?perplexity\s*=\s*([\d.]+)", txt, re.I | re.S)
    if not m:
        m = re.search(r"perplexity:\s*final estimate.*?=\s*([\d.]+)", txt, re.I | re.S)
    return float(m.group(1)) if m else None

# stage 0: wait for KAT PPL to finish
print("[stage 0] waiting for KAT-CQ3 PPL to complete...", flush=True)
while not ppl_done(KAT_LOG):
    time.sleep(60)
kat = final_ppl(KAT_LOG)
print(f"[stage 0] KAT-CQ3 PPL = {kat}", flush=True)
time.sleep(10)

# stage 1: delete broken file
if os.path.exists(BROKEN):
    os.remove(BROKEN)
    print(f"[stage 1] deleted broken {BROKEN}", flush=True)
time.sleep(5)

# stage 2: re-download official
if not (os.path.exists(OFFICIAL) and not os.path.exists(OFFICIAL + ".pstate.json")):
    # pardl2 preallocates the full size upfront; the completion signal is
    # the .pstate.json sidecar being removed. Never trust size alone.
    if os.path.exists(OFFICIAL) and os.path.exists(OFFICIAL + ".pstate.json"):
        os.remove(OFFICIAL)  # restart cleanly (resumable chunks lost, 16 min cost)
    print("[stage 2] re-downloading official Q4_K_M...", flush=True)
    r = subprocess.run([sys.executable, r"D:\merge\train\pardl2.py", URL, OFFICIAL, str(SIZE), "16"])
    if r.returncode != 0:
        print("[stage 2] DOWNLOAD FAILED", flush=True); sys.exit(1)
else:
    print("[stage 2] official already present", flush=True)
print("[stage 3] official PPL...", flush=True)
with open(OFF_LOG, "wb") as lf:
    subprocess.run([PPL_EXE, "-m", OFFICIAL, "-f", CORPUS,
                    "-ngl", "99", "-cmoe", "-t", "12", "-c", "4096"],
                   stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
off = final_ppl(OFF_LOG)
print(f"[stage 3] Ornith-official PPL = {off}", flush=True)
time.sleep(10)

# stage 4: CQ PPL
print("[stage 4] CQ PPL...", flush=True)
with open(CQ_LOG, "wb") as lf:
    subprocess.run([PPL_EXE, "-m", CQ, "-f", CORPUS,
                    "-ngl", "99", "-cmoe", "-t", "12", "-c", "4096"],
                   stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
cq = final_ppl(CQ_LOG)
print(f"[stage 4] Ornith-CQ PPL = {cq}", flush=True)

# stage 5: verdict inputs
print("\n===== V115 VERDICT INPUTS =====", flush=True)
print(f"KAT-CQ3 (context):  {kat}", flush=True)
print(f"Ornith official:    {off}", flush=True)
print(f"Ornith CQ:          {cq}", flush=True)
if off is not None and cq is not None:
    d = cq - off
    verdict = "PASS" if d <= 0.05 else ("MARGINAL" if d <= 0.30 else "FAIL")
    print(f"requant delta: {d:+.4f} -> {verdict}", flush=True)
print("ORCHESTRATION DONE", flush=True)
