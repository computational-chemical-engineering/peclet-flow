#!/usr/bin/env python3
"""G-D2 (b), (c) of doc/vof_projection_cost_design.md §13: the FP32 V-cycle against the FP64 one.

Reads the dense preconditioners `mg_dense_precond` writes for the same configuration with the
double operator (`M_double_*`, today's FP64 V-cycle) and with `--precision fp32` (`M_vf32_*`, the
FP32 V-cycle of §4), and checks per configuration:
  (b) no new indefiniteness: lambda_min(sym(M_fp32)) on the mean-free subspace
      >= lambda_min(sym(M_fp64)) - 1e-3 * lambda_max(sym(M_fp64));
  (c) ||M_fp32 - M_fp64||_F / ||M_fp64||_F <= 1e-4 at ratio <= 1e3, <= 1e-3 at 1e4.
The mean-free subspace is spanned by an orthonormal basis of 1^perp (every configuration here is
one connected fluid component), so the exact null direction of M does not enter lambda_min.

Usage: python mg_precond_fp32cmp.py <dir> [more dirs...]; last line G-D2 PASS|FAIL, exit 0 iff PASS.
"""
import glob
import os
import re
import sys

import numpy as np

MAGIC = 0x50434D47444E5331


def load(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int64, count=3)
        if hdr[0] != MAGIC:
            raise ValueError(f"{path}: bad magic {hdr[0]:x}")
        rows, cols = int(hdr[1]), int(hdr[2])
        a = np.fromfile(f, dtype=np.float64, count=rows * cols)
    return a.reshape(rows, cols)


def meanfree_eigs(M):
    n = M.shape[0]
    # orthonormal basis of 1^perp: the last n-1 columns of a Householder-completed basis
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), np.eye(n)[:, : n - 1]]))
    B = Q[:, 1:]
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(B.T @ S @ B)


def main(dirs):
    ok_all = True
    rows = 0
    for d in dirs:
        for p64 in sorted(glob.glob(os.path.join(d, "M_double_*.bin"))):
            p32 = p64.replace("M_double_", "M_vf32_")
            if not os.path.exists(p32):
                continue
            m = re.search(r"M_double_(\w+?)_n(\d+)_L(\d+)_r([0-9e+.-]+)\.bin", os.path.basename(p64))
            geom, n, lev, ratio = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
            M64, M32 = load(p64), load(p32)
            fro = np.linalg.norm(M32 - M64) / np.linalg.norm(M64)
            e64, e32 = meanfree_eigs(M64), meanfree_eigs(M32)
            lmin64, lmax64, lmin32 = e64[0], e64[-1], e32[0]
            tol_c = 1e-4 if ratio <= 1e3 else 1e-3
            ok_b = lmin32 >= lmin64 - 1e-3 * lmax64
            ok_c = fro <= tol_c
            ok_all = ok_all and ok_b and ok_c
            rows += 1
            print(f"{geom:8s} n {n:2d} L {lev} ratio {ratio:8.0e}  (c) |dM|/|M| {fro:.3e} "
                  f"(<= {tol_c:.0e}) {'ok' if ok_c else 'FAIL'}  (b) lmin32 {lmin32:+.6e} "
                  f"lmin64 {lmin64:+.6e} lmax64 {lmax64:.4e} {'ok' if ok_b else 'FAIL'}")
    if rows == 0:
        ok_all = False
        print("no M_double_* / M_vf32_* pairs found")
    print(f"G-D2 {'PASS' if ok_all else 'FAIL'} ({rows} configurations)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
