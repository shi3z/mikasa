#!/usr/bin/env python3
"""3-4 agent tasks at once through the console copy (engine already started by console_smoke.py or the UI). usage: console_parallel.py URL TOKEN [N]"""
import sys, json, time, urllib.request
base, token = sys.argv[1].rstrip("/"), sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
def api(path, body=None):
    r = urllib.request.Request(base + path + ("?" if "?" not in path else "&") + "token=" + token, None if body is None else json.dumps(body).encode(), {"Cookie": "yh_token=" + token, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=60) as f: return json.load(f)
tasks = ["Write fib.py with a function fib(n) (iterative), then run python3 -c 'from fib import fib; print(fib(30))' and report the number.",
         "Create primes.py that prints all primes below 50 and run it. Report the output.",
         "Write wc_lines.py that counts the lines of its own file and prints the count, then run it.",
         "Create a file squares.txt containing the squares of 1..10, one per line, using a python one-liner, then show it with cat."]
ids = [api("/api/task", dict(mode="agent", steps=15, supervisor="off", ctx_total=64000, effort="low", prompt=t, max_tokens=6000, wall=300))["id"] for t in tasks[:N]]
t0 = time.time(); done = {}
while len(done) < N and time.time() - t0 < 900:
    for i in ids:
        v = api("/api/job/" + i)
        if v["status"] == "done" and i not in done: done[i] = (time.time() - t0, v.get("phase"), v.get("step"), v.get("ncmd"))
    time.sleep(2)
print("all done after %.0f s" % (time.time() - t0)); [print(" ", k, v) for k, v in done.items()]
print(json.dumps(json.load(urllib.request.urlopen("http://127.0.0.1:8767/props"))["engine"]))
