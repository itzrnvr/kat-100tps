# PURPOSE: PPL gate for Ornith vs KAT on identical corpus + flags (V115).
# Re-measures BOTH models with the same gypsy-dragon llama-perplexity binary
# so the comparison is internally consistent (KAT's historical 3.6952 used a
# different build). Corpus: E0/wiki.test.raw (same file the KAT gates used).
# Self-gates on RAM (>=12GB) and runs serially.
import ctypes, os, subprocess, sys, time

EXE = r"D:\src\gypsy-dragon\build\bin\Release\llama-perplexity.exe"
CORPUS = r"D:\merge\E0\wiki.test.raw"
MODELS = [
    ("KAT-CQ3",  r"C:\merge\KAT-CQ3-MTP.gguf"),
    ("Ornith-Q4KM-req", r"D:\merge\out\Ornith15-Q4KM.gguf"),
]

def free_ram_gb():
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                 ("dwMemoryLoad","ullTotalPhys","ullAvailPhys","ullTotalPageFile",
                  "ullAvailPageFile","ullTotalVirtual","ullAvailVirtual","ullAvailExtendedVirtual")]
    st = M(); st.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 1e9

ram = free_ram_gb()
print(f"free RAM {ram:.1f} GB", flush=True)
if ram < 12.0:
    print("below 12GB gate — not firing", flush=True); sys.exit(1)

for label, model in MODELS:
    if not os.path.exists(model):
        print(f"{label}: missing, skip", flush=True); continue
    log = rf"D:\merge\train\ppl_{label}.log"
    with open(log, "wb") as lf:
        p = subprocess.run([EXE, "-m", model, "-f", CORPUS,
                            "-ngl", "99", "-cmoe", "-t", "12", "-c", "4096"],
                           stdout=lf, stderr=subprocess.STDOUT, timeout=2400)
    txt = open(log, encoding="utf-8", errors="replace").read()
    last = ""
    for line in txt.split("\n"):
        if "perplexity" in line.lower() or "llama_perf" in line:
            last = line.strip()
    print(f"{label}: rc={p.returncode} | {last[:160]}", flush=True)
    time.sleep(5)
print("PPL GATE DONE", flush=True)
