import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import layer0_ref as R
m = R.Model(R.MODEL)
T = len(R.TOKENS)
DUMP = R.DUMP
kqb = np.fromfile(os.path.join(DUMP, "kq_b-0.bin"), "<f4").reshape(24, T, 256)  # [h,t,j]
mask = np.fromfile(os.path.join(DUMP, "kq_mask_rel_pos-0.bin"), "<f2").reshape(24, T, 256)
print("kq_b-0 sample h0 t0:", kqb[0, 0, :8])
print("kq_b-0 sample h0 t4:", kqb[0, 4, :8])
print("mask   sample h0 t0:", mask[0, 0, :8])
print("mask   sample h0 t4:", mask[0, 4, :8])
# our table
x = None
Wrel = m.tensor("blk.0.attn_rel_bias.weight").reshape(R.N_SWA, R.N_REL)
print("rel_bias row0 first 4:", Wrel[0, :4], " row1:", Wrel[1, :4])
