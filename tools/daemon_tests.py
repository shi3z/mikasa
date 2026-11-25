#!/usr/bin/env python3
"""Tests of the mikasa daemon (needs a running `mikasa_daemon --serve`; run with ~/rrr/venv-conv/bin/python).
usage: daemon_tests.py SOCK REFDIR section [section ...]   sections: exact cancel conc cache branch sample limit
REFDIR holds reference token files NAME.txt made by the old plain greedy loop (tools/run_daemon_tests.sh makes them) for the prompts spec/data/NAME.txt."""
import os, sys, time, threading, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "server"))
import chatfmt as C
from daemon_engine import DaemonEngine, DaemonError

REFS = ["edit_whole_py_direct", "editcopy_jsonl_direct", "shell_script_direct", "gen_py_lru_direct", "reason_math"]
GREEDY = dict(temperature=0.0)

def toks(name): return [int(x) for x in open(os.path.join(ROOT, "spec", "data", name + ".txt")).read().split()]
def ref(refdir, name): return [int(x) for x in open(os.path.join(refdir, name + ".txt")).read().split()]
def run(E, ids, n, samp=GREEDY, **kw):
    t = time.time(); g = E.generate(ids, n, samp, None, **kw); out = list(g); return out, g.final, time.time() - t
def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return None if len(a) == len(b) else min(len(a), len(b))
def cmp(tag, a, b):
    d = first_diff(a, b)
    print("  %-34s %s (len %d vs %d%s)" % (tag, "IDENTICAL" if d is None else "DIFFERS at token %d" % d, len(a), len(b), "" if d is None else ", tokens %s vs %s" % (a[d:d+1], b[d:d+1])), flush=True)
    return d is None
def tm(f): t = f["timings"]; return "prefill %.0f ms, ttft %.0f ms, %.1f tok/s, steps %d (multi %d) drafted %d accepted %d, cache=%s" % (t["prefill_ms"], t["ttft_ms"], t["decode_tps"], t["steps"], t["multi_steps"], t["drafted"], t["accepted"], t["cache"])

def sec_exact(E, refdir):
    print("== exact: daemon greedy vs old plain greedy loop (500 tokens), speculation on (default) and off (spec_k=0)")
    for w in REFS:
        r = ref(refdir, w); ids = toks(w)
        o, f, _ = run(E, ids, 500); cmp(w + " spec", o, r); print("      " + tm(f))
        o, f, _ = run(E, ids, 500, spec_k=0); cmp(w + " plain(spec_k=0)", o, r); print("      " + tm(f))

def sec_cancel(E, refdir):
    print("== cancel")
    ids = toks("gen_py_lru_direct")
    g = E.generate(ids, 3000, GREEDY); n = 0
    for t in g:
        n += 1
        if n == 40: t0 = time.time(); g.close(); break
    time.sleep(0.3)
    st = E.stats(); print("  cancel during decode: active slots after 0.3 s = %d" % st["active"])
    # measure the time to the final message: use a manual cancel
    g = E.generate(ids, 3000, GREEDY); it = iter(g)
    for _ in range(40): next(it)
    t0 = time.time(); E._send(dict(op="cancel", id=g.rid))
    for _ in it: pass
    print("  cancel -> final message after %.1f ms (finish=%s, %d tokens)" % ((time.time() - t0) * 1000, g.final["finish"], g.final["n_gen"]))
    big = toks("edit_whole_py_ctx24k_direct")
    g = E.generate(big, 100, GREEDY, no_cache=True)
    time.sleep(1.0); t0 = time.time(); E._send(dict(op="cancel", id=g.rid))
    out = list(g); print("  cancel during prefill of %d tokens: final after %.1f ms (finish=%s, n_gen=%d)" % (len(big), (time.time() - t0) * 1000, g.final["finish"], g.final["n_gen"]))
    o, f, _ = run(E, ids, 50); print("  next request after cancel ok: %d tokens, finish=%s; active=%d" % (len(o), f["finish"], E.stats()["active"]))

def sec_conc(E, refdir):
    print("== 4 concurrent greedy requests vs the plain single-sequence reference")
    names = REFS[:4]; res = {}
    def w(n):
        t = time.time(); g = E.generate(toks(n), 500, GREEDY); out = list(g); res[n] = (out, g.final, time.time() - t)
    th = [threading.Thread(target=w, args=(n,)) for n in names]; t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0; tot = sum(len(res[n][0]) for n in names)
    for n in names: cmp(n, res[n][0], ref(refdir, n)); print("      " + tm(res[n][1]))
    print("  total %d tokens in %.2f s wall (includes prefill) = %.1f tok/s aggregate" % (tot, wall, tot / wall))
    # pure decode aggregate: 4 requests with long output and short prompts
    names = ["gen_py_lru_direct", "gen_c_vec_direct", "gen_py_logstats_direct", "reason_math"]
    res.clear(); th = [threading.Thread(target=w, args=(n,)) for n in names]; t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0; tot = sum(len(res[n][0]) for n in names)
    print("  4 x short prompt, up to 500 tokens: %d tokens in %.2f s = %.1f tok/s aggregate; per request decode tok/s: %s" % (tot, wall, tot / wall, " ".join("%.1f" % res[n][1]["timings"]["decode_tps"] for n in names)))

def sample_file():
    return open(os.path.join(ROOT, "src", "batch_host.inc")).read().split("\n")

def conv_turns(E):
    """3-turn conversation: big file in the first user message; assistant replies come from the engine itself (greedy)."""
    code = "\n".join(sample_file()[100:330])
    msgs = [{"role": "system", "content": "You are a careful code reviewer. Answer briefly."},
            {"role": "user", "content": "Here is a CUDA host source file.\n\n```cpp\n" + code + "\n```\n\nWhat does forward_batch do with multi == 1? Answer in two sentences."}]
    follow = ["Now explain the role of kb_rollback in two sentences.", "Which arrays does the rollback table cover? Answer in one sentence."]
    return msgs, follow

def reply_text(out):
    p = C.StreamParser(); txt = ""
    for t in out:
        for ev in p.feed(t):
            if ev[0] == "content": txt += ev[1]
    return txt

def sec_cache(E, refdir):
    print("== prefix cache: 3-turn conversation (render_prompt), cache vs no_cache (greedy, 120 tokens each)")
    msgs, follow = conv_turns(E)
    allok = True
    for turn in range(3):
        ids = C.render_prompt(msgs)
        oc, fc, _ = run(E, ids, 120)
        on, fn, _ = run(E, ids, 120, no_cache=True)
        print("  turn %d: prompt %d tokens | cached run: n_cached %d (%s), prefill %.0f ms, ttft %.0f ms | no_cache run: n_cached %d, prefill %.0f ms, ttft %.0f ms"
              % (turn + 1, len(ids), fc["n_cached"], fc["timings"]["cache"], fc["timings"]["prefill_ms"], fc["timings"]["ttft_ms"], fn["n_cached"], fn["timings"]["prefill_ms"], fn["timings"]["ttft_ms"]))
        allok &= cmp("turn %d generated tokens" % (turn + 1), oc, on)
        if turn < 2:
            msgs = msgs + [{"role": "assistant", "content": reply_text(oc)}, {"role": "user", "content": follow[turn]}]
    # snapshot restore: a branch off turn 1
    print("== branch: same first turn, different follow-up (restore from a snapshot taken during turn 1)")
    m0, _ = conv_turns(E)
    o1, f1, _ = run(E, C.render_prompt(m0), 120)
    base = m0 + [{"role": "assistant", "content": reply_text(o1)}]
    for i, q in enumerate(["Name one risk of the lazy rollback.", "Summarise multi mode in five words."]):
        ids = C.render_prompt(base + [{"role": "user", "content": q}])
        oc, fc, _ = run(E, ids, 100); on, fn, _ = run(E, ids, 100, no_cache=True)
        print("  branch %d: prompt %d, n_cached %d (%s), prefill %.0f ms vs %.0f ms uncached" % (i, len(ids), fc["n_cached"], fc["timings"]["cache"], fc["timings"]["prefill_ms"], fn["timings"]["prefill_ms"]))
        allok &= cmp("branch %d generated tokens" % i, oc, on)
    # resident continuation (raw tokens): prompt = previous prompt + previous output + new tokens
    print("== resident continuation: prompt = previous prompt + its 60-token output (resume at an unaligned position, 1 token left to prefill)")
    ids = toks("edit_whole_py_direct"); o, f, _ = run(E, ids, 60, no_cache=True)
    ids2 = ids + o
    oc, fc, _ = run(E, ids2, 100); on, fn, _ = run(E, ids2, 100, no_cache=True)
    print("  prompt %d, n_cached %d (%s), prefill %.0f ms vs %.0f ms uncached" % (len(ids2), fc["n_cached"], fc["timings"]["cache"], fc["timings"]["prefill_ms"], fn["timings"]["prefill_ms"]))
    cmp("resident continuation tokens", oc, on)
    print("  cache results all identical: %s" % allok)

def sec_sample(E, refdir):
    print("== sampling (temperature 1.0)")
    ids = toks("gen_py_lru_direct")
    a, fa, _ = run(E, ids, 300, dict(temperature=1.0, seed=7)); b, fb, _ = run(E, ids, 300, dict(temperature=1.0, seed=7)); c, fc, _ = run(E, ids, 300, dict(temperature=1.0, seed=8))
    cmp("same seed twice", a, b); print("  other seed differs from first at token", first_diff(a, c), "; tok/s %.1f" % fa["timings"]["decode_tps"])
    print("  text:", C.tokenizer().decode(a)[:200].replace("\n", " "))
    res = {}
    def w(i):
        g = E.generate(ids, 200, dict(temperature=1.0, seed=100 + i)); res[i] = (list(g), g.final)
    th = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    for t in th: t.start()
    for t in th: t.join()
    print("  4 concurrent sampled requests: lengths", [len(res[i][0]) for i in range(4)], "finish", [res[i][1]["finish"] for i in range(4)], "multi_steps", [res[i][1]["timings"]["multi_steps"] for i in range(4)])
    try: E.generate(ids, 10, dict(temperature=1.0, top_p=0.9))
    except ValueError as e: print("  top_p=0.9 rejected:", e)
    try: E.generate(ids, 10, dict(temperature=1.0, top_k=40))
    except ValueError as e: print("  top_k=40 rejected:", e)

def sec_limit(E, refdir):
    print("== context limit (n_ctx %d)" % E.n_ctx)
    base = toks("edit_whole_py_ctx24k_direct")
    for n in (E.n_ctx - 1, E.n_ctx + 5000):
        ids = (base * (n // len(base) + 1))[:n]
        try: o, f, _ = run(E, ids, 10); print("  prompt %d: finish %s" % (n, f["finish"]))
        except DaemonError as e: print("  prompt %d rejected: %s" % (n, e))
    ids = (base * 4)[:65000]
    o, f, dt = run(E, ids, 2000)
    print("  prompt 65000, max_new 2000: generated %d tokens, finish=%s, prefill %.1f s (%.0f tok/s), decode %.1f tok/s" % (len(o), f["finish"], f["timings"]["prefill_ms"] / 1000, 65000 / (f["timings"]["prefill_ms"] / 1000), f["timings"]["decode_tps"]))

if __name__ == "__main__":
    E = DaemonEngine(sock=sys.argv[1]); refdir = sys.argv[2]
    print("daemon:", {k: E.daemon_info[k] for k in ("n_ctx", "nslots", "spec_k", "spec_copy", "snaps_max")}, flush=True)
    for s in sys.argv[3:]: globals()["sec_" + s](E, refdir)
    print("final stats:", E.stats())
