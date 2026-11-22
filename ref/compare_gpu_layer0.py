#!/usr/bin/env python3
"""Compare GPU layer-0 intermediates (engine/out_l0) with the numpy F32 reference (layer0_ref.py)
and, optionally, with the llama.cpp CPU dumps (dump_all)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import layer0_ref as R  # noqa: E402

GPU = os.path.join(os.path.dirname(__file__), "..", "out_l0")
ORDER = ["norm-0#1", "Qcur-0", "Kcur-0", "Vcur-0", "Rcur-0", "kqv_out-0", "ffn_inp-0",
         "norm-0#2", "ffn_moe_logits-0", "ffn_moe_out-0", "ffn_out-0", "l_out-0"]


def main():
    m = R.Model(R.MODEL)
    ref = {}
    R.layer0(m, lambda n, a: ref.__setitem__(n, np.asarray(a, np.float32)))
    print(f"{'tensor':20s} {'size':>8s} {'max abs err':>12s} {'max rel err':>12s} {'first bad token':>16s}")
    first_bad = None
    for name in ORDER:
        r = ref[name]
        g = np.fromfile(os.path.join(GPU, name + ".bin"), dtype="<f4").reshape(r.shape)
        err = np.abs(g - r)
        scale = np.max(np.abs(r)) + 1e-30
        rel = err.max() / scale
        per_tok = err.reshape(r.shape[0], -1).max(axis=1) / scale
        bad = np.nonzero(per_tok > 1e-4)[0]
        tok = int(bad[0]) if bad.size else -1
        print(f"{name:20s} {r.size:8d} {err.max():12.3e} {rel:12.3e} {tok:16d}")
        if bad.size and first_bad is None:
            first_bad = name
    print("first tensor in the chain exceeding 1e-4 relative:", first_bad or "none")


if __name__ == "__main__":
    main()
