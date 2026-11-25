#!/usr/bin/env python3
"""Build token files for the long-context tests (needs the venv ~/rrr/venv-conv/bin/python).
  mkctx.py ctx OUT_PREFIX N...   -> OUT_PREFIX<N>.txt : first N tokens of [probe prompts seq1..3, then real documentation text]
  mkctx.py needle OUT TARGET_TOKENS DEPTH [effort]  -> chat-template prompt (chatfmt.render_prompt) of ~TARGET tokens with the needle at DEPTH (0..1)"""
import sys, os, glob, random
sys.path.insert(0, os.path.expanduser("~/rrr/mikasa/server"))
import chatfmt as C
tk = C.tokenizer()
def docs():
    fs = sorted(glob.glob(os.path.expanduser("~/rrr/llama.cpp-inkling/docs/**/*.md"), recursive=True)) + \
         sorted(glob.glob(os.path.expanduser("~/rrr/llama.cpp-inkling/tools/**/README.md"), recursive=True))
    for f in fs:
        try: t = open(f, errors="ignore").read()
        except Exception: continue
        yield t
def wr(path, ids):
    open(path, "w").write("\n".join(map(str, ids)) + "\n")
if sys.argv[1] == "ctx":
    pre = os.environ.get("PROBE", "probe")
    ids = []
    for k in (1, 2, 3): ids += [int(x) for x in open(f"{pre}/seq{k}.txt").read().replace(",", " ").split()]
    need = max(int(a) for a in sys.argv[3:])
    for t in docs():
        if len(ids) >= need: break
        ids += tk.encode(t, add_special_tokens=False).ids
    rnd = random.Random(1)
    while len(ids) < need: ids.append(rnd.randrange(1000, 150000))
    for a in sys.argv[3:]:
        wr(f"{sys.argv[2]}{a}.txt", ids[:int(a)]); print(a, "tokens")
else:
    out, target, depth = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    effort = sys.argv[5] if len(sys.argv) > 5 else None
    needle = "The secret code is 7391."
    q = "\n\nQuestion: What is the secret code mentioned in the document above? Answer with the number only."
    paras = []
    for t in docs():
        paras += [p for p in t.split("\n\n") if len(p.strip()) > 40]
    random.Random(2).shuffle(paras)
    def build(n):
        body = []; tot = 0
        for p in paras:
            body.append(p); tot += len(tk.encode(p, add_special_tokens=False).ids) + 2
            if tot >= n: break
        k = int(len(body) * depth)
        body.insert(k, needle)
        return "Here is a long document.\n\n" + "\n\n".join(body) + q
    n = target
    for _ in range(4):
        ids = C.render_prompt([{"role": "user", "content": build(n)}], reasoning_effort=effort)
        if abs(len(ids) - target) < 300: break
        n += target - len(ids)
    wr(out, ids); print(out, len(ids), "tokens")
