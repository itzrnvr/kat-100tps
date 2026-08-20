# one-shot: find + kill any process whose command line references pardl2 or
# holds Ornith REF; report llama-server command line too.
import subprocess
out = subprocess.run(
    ["powershell", "-NoProfile", "-Command",
     "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json"],
    capture_output=True, text=True).stdout
import json, os
try:
    data = json.loads(out)
except Exception:
    data = []
if isinstance(data, dict):
    data = [data]
killed = []
for p in data:
    pid = p.get("ProcessId")
    cmd = p.get("CommandLine") or ""
    if "pardl2" in cmd:
        print(f"[pardl2] pid={pid} killing")
        os.popen(f"taskkill /F /PID {pid}")
        killed.append(pid)
    elif "llama-server" in cmd or "llama-perplexity" in cmd:
        print(f"[llama] pid={pid} {cmd[:130]}")
print("killed:", killed)
