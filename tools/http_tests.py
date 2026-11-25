#!/usr/bin/env python3
"""End-to-end tests through the OpenAI-compatible server (mikasa_server.py + daemon). usage: http_tests.py http://127.0.0.1:PORT section...
sections: basic tools stream cancel reject speed"""
import sys, os, json, time, threading, http.client, urllib.parse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BASE = urllib.parse.urlparse(sys.argv[1])

def conn(): return http.client.HTTPConnection(BASE.hostname, BASE.port, timeout=600)
def get(path):
    c = conn(); c.request("GET", path); r = c.getresponse(); return r.status, json.loads(r.read())
def post(body, stream=False):
    c = conn(); c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"}); return c, c.getresponse()
def sse(r):
    for raw in r:
        line = raw.decode().strip()
        if line.startswith("data:"):
            d = line[5:].strip()
            if d == "[DONE]": return
            yield json.loads(d)
def collect(body):
    body = dict(body, stream=True, stream_options={"include_usage": True}); t0 = time.time()
    c, r = post(body); assert r.status == 200, (r.status, r.read())
    reasoning = content = ""; calls = {}; fin = None; usage = None; tfirst = None; nchunks = 0
    for j in sse(r):
        if j.get("usage"): usage = j["usage"]
        for ch in j.get("choices", []):
            d = ch.get("delta") or {}
            if (d.get("reasoning_content") or d.get("content") or d.get("tool_calls")) and tfirst is None: tfirst = time.time() - t0
            reasoning += d.get("reasoning_content") or ""; content += d.get("content") or ""; nchunks += 1
            for tc in d.get("tool_calls") or []:
                e = calls.setdefault(tc["index"], dict(name="", args="")); e["name"] += (tc.get("function") or {}).get("name") or ""; e["args"] += (tc.get("function") or {}).get("arguments") or ""
            fin = ch.get("finish_reason") or fin
    return dict(reasoning=reasoning, content=content, calls=[calls[k] for k in sorted(calls)], finish=fin, usage=usage, secs=time.time() - t0, ttft=tfirst)

BASH = {"type": "function", "function": {"name": "bash", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
SAMP = dict(temperature=1.0, top_p=1.0, top_k=0)

def sec_basic():
    print("/health", get("/health")); s, p = get("/props"); print("/props n_ctx", p["default_generation_settings"]["n_ctx"], "slots", p["total_slots"], "build", p["build_info"])
    c, r = post(dict(model="inkling", messages=[{"role": "user", "content": "Say hello in five words."}], max_tokens=300, **SAMP)); j = json.loads(r.read())
    m = j["choices"][0]["message"]; print("non-stream:", j["choices"][0]["finish_reason"], "reasoning %d chars" % len(m.get("reasoning_content") or ""), "content:", repr(m.get("content"))[:120], j["usage"])

def sec_tools():
    msgs = [{"role": "user", "content": "List the files in the current directory with the bash tool."}]
    for stream in (False, True):
        body = dict(model="inkling", messages=msgs, tools=[BASH], max_tokens=1500, **SAMP)
        if stream: r = collect(body); print("stream    :", r["finish"], "reasoning %d chars" % len(r["reasoning"]), "tool_calls", r["calls"], "content", repr(r["content"])[:60], "usage", r["usage"])
        else:
            c, rr = post(body); j = json.loads(rr.read()); m = j["choices"][0]["message"]
            print("non-stream:", j["choices"][0]["finish_reason"], "reasoning %d chars" % len(m.get("reasoning_content") or ""), "tool_calls", [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []])

def sec_stream():
    r = collect(dict(model="inkling", messages=[{"role": "user", "content": "What is 17 * 23? Think, then answer in one line."}], max_tokens=800, **SAMP))
    print("stream: finish", r["finish"], "ttft %.2fs total %.2fs" % (r["ttft"], r["secs"]), "reasoning %d chars, content %r" % (len(r["reasoning"]), r["content"][:100]), r["usage"])

def sec_cancel():
    c, r = post(dict(model="inkling", messages=[{"role": "user", "content": "Write a very long story about a dragon."}], max_tokens=3000, stream=True, **SAMP))
    n = 0
    for j in sse(r):
        n += 1
        if n == 30: break
    r.close(); c.close(); t0 = time.time(); act = None
    while time.time() - t0 < 5:
        s, p = get("/props"); act = p["engine"]["active"]
        if act == 0: break
        time.sleep(0.05)
    print("client disconnect after 30 chunks: engine active slots = %s after %.2f s" % (act, time.time() - t0))

def sec_reject():
    for extra in (dict(top_p=0.9), dict(top_k=40), dict(stop=["x"]), dict(presence_penalty=0.5), dict(min_p=0.1)):
        body = dict(model="inkling", messages=[{"role": "user", "content": "hi"}], max_tokens=5, temperature=1.0); body.update(extra)
        c, r = post(body); print("  %-22s -> HTTP %d %s" % (extra, r.status, json.loads(r.read())["error"]["message"][:80]))
    body = dict(model="inkling", messages=[{"role": "user", "content": "x " * 70000}], max_tokens=5, temperature=1.0)
    c, r = post(body); print("  70000-word prompt     -> HTTP %d %s" % (r.status, json.loads(r.read())["error"]["message"][:80]))

def edit_prompt():
    src = open(os.path.join(ROOT, "server", "chatfmt.py")).read()
    return [{"role": "user", "content": "Here is a Python file.\n\n```python\n" + src + "\n```\n\nOutput the complete file again, with the function `_fix_messages` renamed to `_normalize_messages` everywhere. Output only the code."}]
def chat_prompt():
    return [{"role": "user", "content": "Write a Python module implementing a thread-safe LRU cache with TTL, with type hints and docstrings."}]

def sec_speed():
    for name, msgs, mt in (("chat (new code)", chat_prompt(), 600), ("edit (copy-heavy)", edit_prompt(), 1500)):
        for rep in range(2):
            r = collect(dict(model="inkling", messages=msgs, max_tokens=mt, **SAMP)); u = r["usage"]; tm = u.get("timings", {})
            print("  %-18s run %d: prompt %d (cached %d), %d tokens, ttft %.2fs, decode %.1f tok/s (daemon), wall %.1f tok/s incl ttft; finish %s" % (name, rep, u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"], u["completion_tokens"], r["ttft"], tm.get("decode_tps", 0), u["completion_tokens"] / r["secs"], r["finish"]))
    print("  4 concurrent streams (2 chat + 2 edit), temperature 1.0:")
    res = [None] * 4; mm = [chat_prompt(), edit_prompt(), chat_prompt(), edit_prompt()]
    def w(i): res[i] = collect(dict(model="inkling", messages=mm[i], max_tokens=800, **SAMP))
    th = [threading.Thread(target=w, args=(i,)) for i in range(4)]; t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0; tot = sum(r["usage"]["completion_tokens"] for r in res)
    print("   %d tokens in %.1f s wall = %.1f tok/s aggregate (incl. prefill); per stream daemon decode tok/s: %s; multi_steps %s" % (tot, wall, tot / wall, [round(r["usage"]["timings"]["decode_tps"], 1) for r in res], [r["usage"]["timings"]["multi_steps"] for r in res]))
    # steady state: stagger-free, long outputs
    res = [None] * 4
    def w2(i): res[i] = collect(dict(model="inkling", messages=chat_prompt(), max_tokens=700, **SAMP))
    th = [threading.Thread(target=w2, args=(i,)) for i in range(4)]; t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0; tot = sum(r["usage"]["completion_tokens"] for r in res)
    print("  4 x identical chat prompt (same prefix, different sampling): %d tokens in %.1f s = %.1f tok/s aggregate; per stream decode: %s" % (tot, wall, tot / wall, [round(r["usage"]["timings"]["decode_tps"], 1) for r in res]))

if __name__ == "__main__":
    for s in sys.argv[2:]: print("==", s); globals()["sec_" + s]()
