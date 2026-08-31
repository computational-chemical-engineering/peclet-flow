#!/usr/bin/env python3
"""WO-M — spectrum of the densely assembled multigrid preconditioner (float vs double storage).

Reads the matrices written by `mg_dense_precond` and answers, for each configuration, the ONE
question WO-M was opened on: is the V-cycle preconditioner M indefinite because the coarse operator
is a bad approximation (VOF_PLAN S3, coefficient-aware coarsening), or because the operator is
STORED in float (mac_cutcell_mg.hpp MReal) and float rounding breaks A*1 = 0 at eps_f32 x contrast?

Both builds run the identical code path on the identical problem; only MReal differs.

Method note. sym(M) is singular BY CONSTRUCTION: M's inputs are mean-removed and its output is
mean-removed at the end of the level-0 V-cycle, so the constant vector is an exact null direction.
An UNPIVOTED LDL^T on such a matrix produces one pivot at round-off whose SIGN is noise — which is
why WO-H's "1 negative pivot (-1.1e-12)" at ratio 1e2 must not be read as indefiniteness. This
script therefore reports both:
  * the raw unpivoted LDL^T pivots, for direct comparability with the WO-H table, and
  * the eigenvalues of sym(M) deflated onto the mean-free subspace (P sym(M) P, P = I - 11^T/n),
    where a negative eigenvalue that is not at round-off is unambiguous indefiniteness.

Usage: python mg_precond_analyze.py <dir-with-*.bin> [more dirs...]
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


def ldl_pivots(S):
    """Unpivoted LDL^T pivots of a symmetric matrix (the WO-H instrument, reproduced)."""
    n = S.shape[0]
    A = S.astype(np.float64).copy()
    d = np.zeros(n)
    L = np.eye(n)
    for j in range(n):
        d[j] = A[j, j] - (L[j, :j] ** 2 * d[:j]).sum()
        if abs(d[j]) < 1e-300:
            continue
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - (L[i, :j] * L[j, :j] * d[:j]).sum()) / d[j]
    return d


def analyse_operator(path):
    """kappa(A) of the FINE operator on the mean-free subspace, and the attainable-accuracy floor it
    implies. Classical result (Greenbaum; Higham NLA ch.19): a CG-type iteration cannot drive the
    TRUE relative residual below O(eps * kappa(A)) whatever the preconditioner, because round-off in
    the recurrence enters at that level. So this column says what a solve on this operator could
    ever reach in each storage precision — the ceiling any precision policy is measured against, and
    the reason a fixed rtol of 1e-8 stops being meaningful once kappa exceeds ~1e8 in fp64."""
    A = load(path)
    n = A.shape[0]
    S = 0.5 * (A + A.T)
    one = np.ones(n) / np.sqrt(n)
    e1 = np.zeros(n)
    e1[0] = 1.0
    w = one - e1
    nw = np.linalg.norm(w)
    H = np.eye(n) - 2.0 * np.outer(w, w) / (nw * nw) if nw > 1e-14 else np.eye(n)
    Q = H[:, 1:]
    ev = np.abs(np.linalg.eigvalsh(Q.T @ S @ Q))
    kappa = float(ev.max() / max(ev.min(), 1e-300))
    return dict(kappa=kappa, floor_f32=2.0 ** -24 * kappa, floor_f64=2.0 ** -53 * kappa)


def analyse(path):
    M = load(path)
    n = M.shape[0]
    S = 0.5 * (M + M.T)
    fnorm = np.linalg.norm(M)
    skew = np.linalg.norm(M - M.T) / (fnorm + 1e-300)

    d = ldl_pivots(S) if n <= 1024 else np.array([np.nan])  # the O(n^3) python LDL is 8^3-sized
    scale = np.max(np.abs(d))
    # a pivot below 1e-10 of the largest is at round-off: the constant null direction, sign is noise
    real_neg = np.sum((d < 0) & (np.abs(d) > 1e-10 * scale))
    any_neg = int(np.sum(d < 0))
    min_piv = float(d.min())

    # Restrict sym(M) to the mean-free subspace and take its true spectrum there. Build an
    # orthonormal basis Q of the complement of the constant (Householder) and diagonalise Q^T S Q —
    # NOT "deflate and drop the smallest eigenvalue", which at high contrast would discard a genuine
    # negative instead of the deflated zero.
    one = np.ones(n) / np.sqrt(n)
    e1 = np.zeros(n)
    e1[0] = 1.0
    w = one - e1
    nw = np.linalg.norm(w)
    H = np.eye(n) - 2.0 * np.outer(w, w) / (nw * nw) if nw > 1e-14 else np.eye(n)
    Q = H[:, 1:]  # columns span {1}^perp
    ev = np.linalg.eigvalsh(Q.T @ S @ Q)
    lmax = ev.max()
    neg_ev = int(np.sum(ev < -1e-10 * lmax))
    return dict(
        n=n,
        skew=skew,
        any_neg=any_neg,
        real_neg=int(real_neg),
        min_piv=min_piv,
        lmin=float(ev.min()),
        lmax=float(lmax),
        neg_ev=neg_ev,
    )


def main(dirs):
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "M_*.bin")))
    if not files:
        print("no M_*.bin found")
        return 1
    rows = []
    for p in files:
        m = re.match(r"M_(\w+)_(\w+)_n(\d+)_L(\d+)_r([0-9e+.]+)\.bin", os.path.basename(p))
        if not m:
            continue
        prec, geom, nn, lv, ratio = m.groups()
        r = analyse(p)
        ap = os.path.join(os.path.dirname(p), "A" + os.path.basename(p)[1:])
        r.update(analyse_operator(ap) if os.path.exists(ap) else
                 dict(kappa=float("nan"), floor_f32=float("nan"), floor_f64=float("nan")))
        r.update(prec=prec, geom=geom, grid=int(nn), levels=int(lv), ratio=float(ratio))
        rows.append(r)

    rows.sort(key=lambda r: (r["geom"], r["ratio"], r["prec"]))
    print()
    print(
        f"{'geom':<9}{'ratio':>8}  {'prec':<7}{'skew':>10}"
        f"{'LDL neg':>9}{'(raw)':>7}{'min pivot':>12}"
        f"{'lambda_min':>13}{'lambda_max':>12}{'neg eig':>9}"
        f"{'kappa(A)':>11}{'floor f32':>11}{'floor f64':>11}"
    )
    print("-" * 129)
    for r in rows:
        print(
            f"{r['geom']:<9}{r['ratio']:>8.0e}  {r['prec']:<7}{r['skew']:>10.2e}"
            f"{r['real_neg']:>9}{r['any_neg']:>7}{r['min_piv']:>12.2e}"
            f"{r['lmin']:>13.3e}{r['lmax']:>12.3e}{r['neg_ev']:>9}"
            f"{r['kappa']:>11.2e}{r['floor_f32']:>11.2e}{r['floor_f64']:>11.2e}"
        )
    print()
    print("LDL neg = pivots below -1e-10*max|pivot| (real indefiniteness);")
    print("(raw)   = every negative pivot including the round-off/null-direction sign noise;")
    print("neg eig = negative eigenvalues of sym(M) on the mean-free subspace (the verdict column);")
    print("kappa(A), floor = the FINE operator's condition number on the mean-free subspace and the")
    print("          O(eps*kappa) attainable relative residual it implies in each storage precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["."]))
