#!/usr/bin/env python3
# usage: make_fast_tables.py BASE_TAG:FAST_TAG [BASE_TAG:FAST_TAG ...]  -> markdown table of tok/s (plain, spec) and token-file identity
import re, sys, os
def load(tag):
    d = {}
    for l in open("logs/fast/%s/summary.txt" % tag):
        m = re.match(r"(\S+) (plain|spec) .*generated (\d+) tokens in [\d.]+ s = ([\d.]+) tok/s.*steps (\d+), tokens/step ([\d.]+)\s+md5=(\w+)", l)
        if m: d[(m.group(1), m.group(2))] = (float(m.group(4)), int(m.group(5)), float(m.group(6)), m.group(7), int(m.group(3)))
    return d
print("| workload | tokens | plain before | plain after | spec before | spec after | spec x (after/before) | spec x vs plain-before | tokens/step (spec) | token files identical |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for pair in sys.argv[1:]:
    bt, ft = pair.split(":"); B = load(bt); F = load(ft)
    for w in sorted({k[0] for k in F}, key=lambda x: list(dict.fromkeys(k[0] for k in F)).index(x)):
        pb, pf, sb, sf = B[(w, "plain")], F[(w, "plain")], B[(w, "spec")], F[(w, "spec")]
        same = all(open("logs/fast/%s/%s.%s.txt" % (bt, w, c)).read() == open("logs/fast/%s/%s.%s.txt" % (ft, w, c)).read() for c in ("plain", "spec"))
        print("| %s | %d | %.1f | %.1f | %.1f | %.1f | %.2fx | %.2fx | %.2f | %s |" % (w, pf[4], pb[0], pf[0], sb[0], sf[0], sf[0] / sb[0], sf[0] / pb[0], sf[2], "yes" if same else "NO"))
