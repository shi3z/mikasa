#!/usr/bin/env python3
"""Summarise logs/spec/*.log into markdown tables.  usage: spec_table.py logdir "wl1 wl2 ..." """
import sys, re, os
d, wls = sys.argv[1], sys.argv[2].split()
def rd(w, c):
    f = f"{d}/{w}.{c}.log"
    if not os.path.exists(f): return None
    t = open(f).read(); r = {}
    m = re.search(r"generated (\d+) tokens in ([\d.]+) s \(", t)
    if m: r["n"] = int(m[1]); r["tps"] = int(m[1]) / float(m[2]); return r
    m = re.search(r"generated (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s.*?steps (\d+), tokens/step ([\d.]+)", t)
    if not m: return None
    r.update(n=int(m[1]), tps=float(m[3]), steps=int(m[4]), tps_step=float(m[5]))
    m = re.search(r"drafted (\d+), accepted (\d+) \(acceptance ([\d.]+)%\), mean accepted per step ([\d.]+), mean accepted per drafting step ([\d.]+), drafting steps (\d+) of (\d+)", t)
    if m: r.update(drafted=int(m[1]), acc=float(m[3]), mean_acc=float(m[4]), mean_acc_dr=float(m[5]), dsteps=int(m[6]), steps=int(m[7]))
    return r
exact = {}
if os.path.exists(f"{d}/exact.txt"):
    for l in open(f"{d}/exact.txt"):
        m = re.match(r"(\S+)\s+ref\s+(\d+) tokens, test\s+(\d+) tokens: (.*)", l)
        if m: exact[m[1]] = m[4].replace("first divergence at token ", "diverges @")
print("| workload | tokens | old tok/s | K=0 tok/s | cfg | tok/s | speedup vs old | tokens/step | drafts/step (frac. drafting steps) | acceptance of drafted | mean accepted per drafting step | identical to old |")
print("|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---|")
for w in wls:
    o, p = rd(w, "old"), rd(w, "p0")
    if not o: continue
    for c, nm in (("s8", "ngram K<=8"), ("s15", "+copy K<=15"), ("w63", "wide K<=63")):
        r = rd(w, c)
        if not r: continue
        print("| %s | %d | %.1f | %.1f | %s | %.1f | %.2fx | %.2f | %.2f (%.0f%%) | %.0f%% | %.2f | %s |" % (w, o["n"], o["tps"], p["tps"] if p else 0, nm, r["tps"], r["tps"] / o["tps"], r["tps_step"],
              r["drafted"] / max(1, r["steps"]), 100.0 * r["dsteps"] / max(1, r["steps"]), r["acc"], r["mean_acc_dr"], exact.get(f"{w}.{c}", "?")))
