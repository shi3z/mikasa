#!/usr/bin/env python3
import sys
a = [int(x) for x in open(sys.argv[1]).read().split()]; b = [int(x) for x in open(sys.argv[2]).read().split()]
n = min(len(a), len(b)); d = next((i for i in range(n) if a[i] != b[i]), None)
print("%-34s ref %4d tokens, test %4d tokens: %s" % (sys.argv[3], len(a), len(b), "IDENTICAL" if d is None and len(a) == len(b) else ("first divergence at token %d (of %d)" % (d, n) if d is not None else "prefix equal, lengths differ")))
