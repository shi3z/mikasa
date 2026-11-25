import sys
a=[int(x) for x in open(sys.argv[1])]; b=[int(x) for x in open(sys.argv[2])]
n=min(len(a),len(b)); lo=int(sys.argv[3]) if len(sys.argv)>3 else 0
eq=sum(1 for i in range(n) if a[i]==b[i]); 
eq2=sum(1 for i in range(max(lo,0),n) if a[i]==b[i])
print(f"argmax agreement {sys.argv[1]} vs {sys.argv[2]}: all {eq}/{n} = {100*eq/n:.2f}%  positions>={lo}: {eq2}/{n-lo} = {100*eq2/max(1,n-lo):.2f}%")
