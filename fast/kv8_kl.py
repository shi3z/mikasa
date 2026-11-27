import numpy as np, sys
n=200058
def load(p,K): return np.fromfile(p,dtype=np.float32).reshape(K,n).astype(np.float64)
def lsm(x): m=x.max(1,keepdims=True); return x-m-np.log(np.exp(x-m).sum(1,keepdims=True))
ref=lsm(load(sys.argv[1],128)); toks=[int(t) for t in open(sys.argv[2]).read().split()][:128]
print("ref NLL of forced tokens mean %.4f"%(-ref[np.arange(128),toks].mean()))
for p in sys.argv[3:]:
    t=lsm(load(p,128)); pr=np.exp(ref)
    kl=(pr*(ref-t)).sum(1); nll=-t[np.arange(128),toks]
    print("%-24s KL(ref||test) mean %.4f max %.3f | NLL %.4f | top1 agree %d/128"%(p.split('/')[-1],kl.mean(),kl.max(),nll.mean(),(t.argmax(1)==ref.argmax(1)).sum()))
