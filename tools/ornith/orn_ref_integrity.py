# PURPOSE: Post-download integrity check for a pardl2-downloaded GGUF.
# Samples N random 1MB regions, range-fetches the same regions from HF, and
# byte-compares. Catches zero-holes from interrupted/killed downloads that
# size checks cannot (pardl2 preallocates the full size upfront).
# V115a rule: a pardl2 target is complete ONLY when sidecar absent, no
# pardl2 process running, AND this check passes.
import os, random, sys, urllib.request

PATH = r"D:\merge\out\Ornith15-Q4KM-REF.gguf"
URL = "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-Q4_K_M.gguf"
SIZE = 21713462848
N_SAMPLES = 10
REGION = 1 << 20  # 1MB

side = PATH + ".pstate.json"
if os.path.exists(side):
    print("FAIL: sidecar still present (download incomplete)")
    sys.exit(1)
if os.path.getsize(PATH) != SIZE:
    print(f"FAIL: size {os.path.getsize(PATH)} != {SIZE}")
    sys.exit(1)

random.seed(20260820)
f = open(PATH, "rb")
bad = 0
for i in range(N_SAMPLES):
    start = random.randrange(SIZE - REGION)
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start+REGION-1}"})
    remote = urllib.request.urlopen(req, timeout=120).read()
    f.seek(start)
    local = f.read(REGION)
    ok = remote == local
    if not ok:
        bad += 1
        # zero-hole signature?
        zeros = sum(1 for j in range(0, REGION, 4096) if local[j:j+64] == b"\0"*64)
        print(f"  region @{start}: MISMATCH (zero-pages~{zeros}/{REGION//4096})")
    else:
        print(f"  region @{start}: ok")
f.close()
print("INTEGRITY:", "PASS" if bad == 0 else f"FAIL ({bad}/{N_SAMPLES} bad)")
sys.exit(0 if bad == 0 else 1)
