import numpy as np, sys
def load(p): return np.fromfile(p, dtype=np.float32)
a=load(sys.argv[1]); b=load(sys.argv[2])
K=int(sys.argv[3]) if len(sys.argv)>3 else 1
n=200058
a=a.reshape(K,n); b=b.reshape(K,n)
for k in range(K):
    x,y=a[k],b[k]
    d=x-y
    top=np.argsort(-y)[:10]
    print(f"tok{k}: relL2 {np.linalg.norm(d)/np.linalg.norm(y):.3e} max|d| {np.abs(d).max():.3f} (max|y| {np.abs(y).max():.2f}) top1 {int(x.argmax())==int(y.argmax())} top10-overlap {len(set(np.argsort(-x)[:10])&set(top))}/10  mean|d| over top10 {np.abs(d[top]).mean():.3f} gap(top1-top2 ref) {y[top[0]]-y[top[1]]:.3f}")
