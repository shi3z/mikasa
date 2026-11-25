#!/usr/bin/env python3
"""Drive a COPY of the web console (webui.py --engine mikasa --port 8771) through the API: start the mikasa preset, run a tiny agent task, stop the engine.
usage: console_smoke.py http://127.0.0.1:8771 TOKEN"""
import sys, json, time, urllib.request, os
base, token = sys.argv[1].rstrip("/"), sys.argv[2]
def api(path, body=None):
    r = urllib.request.Request(base + path + ("?" if "?" not in path else "&") + "token=" + token, None if body is None else json.dumps(body).encode(), {"Cookie": "yh_token=" + token, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=60) as f: return json.load(f)
print("engines:", json.dumps(api("/api/engines")))
print("start:", api("/api/engine/start", {"name": "inkling"}))
t0 = time.time()
while not api("/api/engines")["inkling"]["up"]:
    if time.time() - t0 > 600: sys.exit("engine did not come up")
    time.sleep(3)
print("engine up after %.0f s: %s" % (time.time() - t0, api("/api/engines")["inkling"]))
task = sys.argv[3] if len(sys.argv) > 3 else "Create a file hello.py that prints 'hello from mikasa', then run it with python3 and tell me its output."
j = api("/api/task", dict(mode="agent", steps=12, supervisor="off", ctx_total=64000, effort="low", prompt=task, max_tokens=4000, wall=300))
jid = j["id"]; print("job", jid); t0 = time.time()
while True:
    v = api("/api/job/" + jid)
    if v["status"] == "done": break
    if time.time() - t0 > 900: print("TIMEOUT"); break
    time.sleep(2)
print("job finished after %.0f s: phase=%r steps=%s ncmd=%s" % (time.time() - t0, v.get("phase"), v.get("step"), v.get("ncmd")))
print("workdir", v.get("workdir"))
for f in os.listdir(v["workdir"]) if v.get("workdir") and os.path.isdir(v["workdir"]) else []: print("  file:", f)
print("props:", json.dumps(json.load(urllib.request.urlopen("http://127.0.0.1:8767/props"))["engine"]))
print("stop:", api("/api/engine/stop", {"name": "inkling"}))
