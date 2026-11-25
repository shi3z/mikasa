# compare per-step logits of two runs (files of NVOCAB float32 per step): argmax equality, rel L2, top-1 gap of the reference
import numpy as np, sys
n = 200058
a = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, n)   # test
b = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, n)   # reference
K = min(len(a), len(b))
bad = 0; rel = []; maxabs = []
for k in range(K):
    x, y = a[k], b[k]
    rel.append(np.linalg.norm(x - y) / np.linalg.norm(y)); maxabs.append(np.abs(x - y).max())
    top = np.argsort(-y)[:2]
    gap = y[top[0]] - y[top[1]]
    ok = x.argmax() == y.argmax()
    if not ok:
        bad += 1
        print(f"  step {k}: argmax {x.argmax()} vs ref {y.argmax()}, ref gap top1-top2 {gap:.3f}, test gap between those two {x[x.argmax()] - x[y.argmax()]:.3f}")
print(f"{sys.argv[1]} vs {sys.argv[2]}: {K} steps, argmax mismatches {bad}, relL2 mean {np.mean(rel):.3e} max {np.max(rel):.3e}, max abs diff {np.max(maxabs):.3f}")
