#!/usr/bin/env python3
"""Compare two prof.py --dump files array by array (the VoF step-performance harness, WO-0).

Usage:  python tests/study/vof_perf/cmp.py a.npz b.npz

One line per array (u, v, w, p, C, dts, iters, and every block colour col<id>): bitwise
equality, max|a-b| and max|a-b|/max|a|. An array present in only one file prints bitwise=False.
The last line is max over u, v, w, p, C of the relative difference -- N50_rtol when a is the
rtol 1e-10 dump and b the 1e-9 one (doc/vof_step_performance_design.md sec. 8, G-NUM item 1).
Exit status 0 only if every array is bitwise equal (G-BIT item 2), 1 otherwise.
"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
all_same = True
worst = 0.0
for k in list(a.files) + [k for k in b.files if k not in a.files]:
    if k not in a.files or k not in b.files:
        print(f"{k:4s} bitwise=False  only in {'b' if k in b.files else 'a'}")
        all_same = False
        continue
    x, y = a[k], b[k]
    same = x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()  # bytes: -0/NaN
    d = float(np.max(np.abs(x.astype(np.float64) - y))) if x.shape == y.shape else float('nan')
    sc = float(np.max(np.abs(x))) or 1.0
    print(f"{k:4s} bitwise={same}  max|diff|={d:.3e}  rel={d/sc:.3e}")
    all_same = all_same and same
    if k in ("u", "v", "w", "p", "C"):
        worst = max(worst, d / sc)
print(f"max rel over u v w p C = {worst:.3e}  all bitwise={all_same}")
sys.exit(0 if all_same else 1)
