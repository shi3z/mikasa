#!/usr/bin/env python3
"""Two-sample chi-square test of two `--spec-trials` outputs (TRIAL count t1 t2 ...). usage: cmp_trials.py plain.log spec.log"""
import sys, math
def rd(f):
    d = {}
    for l in open(f):
        if l.startswith("TRIAL "):
            p = l.split(); d[tuple(p[2:])] = int(p[1])
    return d
A, B = rd(sys.argv[1]), rd(sys.argv[2])
def test(a, b, label):
    na, nb = sum(a.values()), sum(b.values())
    keys = sorted(set(a) | set(b), key=lambda k: -(a.get(k, 0) + b.get(k, 0)))
    chi = 0.0; bins = 0; ra = rb = 0
    for k in keys:
        x, y = a.get(k, 0), b.get(k, 0)
        if x + y < 10: ra += x; rb += y; continue
        chi += (x * math.sqrt(nb / na) - y * math.sqrt(na / nb)) ** 2 / (x + y); bins += 1
    if ra + rb >= 10:
        chi += (ra * math.sqrt(nb / na) - rb * math.sqrt(na / nb)) ** 2 / (ra + rb); bins += 1
    dof = max(1, bins - 1)
    z = ((chi / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    print("%-34s trials %d vs %d, distinct %d / %d, bins %d: chi2 = %.1f, dof = %d, p ~ %.3f" % (label, na, nb, len(a), len(b), bins, chi, dof, p))
def prefix(d, L):
    r = {}
    for k, v in d.items(): r[k[:L]] = r.get(k[:L], 0) + v
    return r
for L in range(1, 4): test(prefix(A, L), prefix(B, L), "first %d generated token(s)" % L)
