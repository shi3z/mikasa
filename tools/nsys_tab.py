#!/usr/bin/env python3
# usage: nsys_tab.py file.csv launches  -> per-launch kernel table (us per launch, calls per launch)
import csv,sys
rows=list(csv.reader(open(sys.argv[1]))) 
i=[k for k,r in enumerate(rows) if r and r[0].startswith('Time (%)')][0]
h=rows[i]; n=float(sys.argv[2]); tot=0; out=[]
for r in rows[i+1:]:
    if len(r)<9: continue
    d=dict(zip(h,r)); t=float(d['Total Time (ns)'])/n/1000; c=int(d['Instances'])/n
    out.append((t,c,d['Name'][:110])); tot+=t
print("total kernel us/launch %.0f, calls %.0f"%(tot,sum(c for _,c,_ in out)))
for t,c,nm in out[:int(sys.argv[3]) if len(sys.argv)>3 else 25]: print("%8.0f %5.1f%% %6.1f  %s"%(t,100*t/tot,c,nm))
