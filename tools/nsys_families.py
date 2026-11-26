#!/usr/bin/env python3
# usage: nsys_families.py before.csv after.csv launches -> per-family kernel time per launch (us) and calls
import csv, sys
FAM = [("Q8 projections / wo / head GEMV", ("gemvb2_q8",)), ("NVFP4 expert GEMV", ("nvfp4_",)), ("Q8_0 expert GEMV", ("q8_experts",)),
       ("attention (+KV write, clip, conv)", ("kb_attn", "kb_kv_write", "kb_clip_qr", "kb_scale_tau", "kb_conv4_kv")),
       ("block mixes", ("kb_mix_",)), ("loop control", ("gemvb_f32_row8", "kb_ctrl_gf", "kb_rms2_rows9")),
       ("router (GEMV, top-8, grouping)", ("gemvb_f32_row9", "kb_router", "kb_group")),
       ("tiny BF16 GEMVs (sinks, conv proj)", ("gemvb_bf16",))]
def load(fn, n):
    rows = list(csv.reader(open(fn))); i = [k for k, r in enumerate(rows) if r and r[0].startswith("Time (%)")][0]; h = rows[i]
    fam = {f: [0.0, 0.0] for f, _ in FAM}; fam["other small (norms, copies, conv, situ, moe sum, argmax)"] = [0.0, 0.0]
    for r in rows[i + 1:]:
        if len(r) < 9: continue
        d = dict(zip(h, r)); nm = d["Name"]
        if "k_repack_il" in nm: continue
        t = float(d["Total Time (ns)"]) / n / 1000; c = int(d["Instances"]) / n
        for f, keys in FAM:
            if any(k in nm for k in keys): fam[f][0] += t; fam[f][1] += c; break
        else:
            k = "other small (norms, copies, conv, situ, moe sum, argmax)"; fam[k][0] += t; fam[k][1] += c
    return fam
a, b = load(sys.argv[1], float(sys.argv[3])), load(sys.argv[2], float(sys.argv[3]))
print("| kernel family | before us/step | calls | after us/step | calls |\n|---|---:|---:|---:|---:|")
for f in a: print("| %s | %.0f | %.0f | %.0f | %.0f |" % (f, a[f][0], a[f][1], b[f][0], b[f][1]))
print("| **total** | **%.0f** | **%.0f** | **%.0f** | **%.0f** |" % (sum(v[0] for v in a.values()), sum(v[1] for v in a.values()), sum(v[0] for v in b.values()), sum(v[1] for v in b.values())))
