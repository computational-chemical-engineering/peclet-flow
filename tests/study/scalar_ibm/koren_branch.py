#!/usr/bin/env python
"""Q-N (WO-9, ruling D-WO9-1): where does G9 koren at R_o/h 32 lose bitwise agreement between two
runs (two backends, or one OpenMP build at two thread counts), and why?

  OMP_NUM_THREADS=1 PYTHONPATH=build_dev python koren_branch.py run a.npz [nsteps]
  OMP_NUM_THREADS=4 PYTHONPATH=build_dev python koren_branch.py run b.npz [nsteps]
  python koren_branch.py analyse a.npz b.npz

`run` drives the G9 gate case unchanged (tests/python/test_scalar_cutcell_gates.py g9_case) and
records c after every step. `analyse` finds the first step whose difference jumps, and for every
face evaluates, on each run's c^n, which branch of sadv::koren (src/staggered_advection.hpp) the
explicit flux takes: |den| < 1e-10 (r = 0), both |num|, |den| < 1e-10 (r = 1), or the ratio. A
koren face is one the kernel's §6.2 guard admits (upwind-upwind, upwind and downwind all unknowns).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))


def run(path, nsteps):
    import test_scalar_cutcell_gates as g
    cs, made, its, res = [], [], [], []

    class Rec(g.pf.Solver):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

        def advance_scalars(self, *a, **k):
            r = super().advance_scalars(*a, **k)
            cs.append(np.array(self.get_field("c")))
            cen = self.diagnostics.scalar_census("c")
            its.append(int(cen["krylov_iterations"]))
            res.append(float(cen["krylov_residual"]))
            return r

    g.pf.Solver = Rec
    Rh = 32
    g.g9_case(Rh, g.g9_offsets(Rh)[0], os.environ.get("SCHEME", "koren"), 0.5, nsteps=nsteps)
    s = made[-1]
    h = g.R_O / Rh
    fx, fy = g.annulus_fluxes(s, h)
    geo = s.diagnostics.scalar_geometry("c")
    np.savez_compressed(path, c=np.stack(cs), unk=np.asarray(geo["unknown"]),
                        kappa=np.asarray(geo["kappa"]), fx=fx, fy=fy, its=np.asarray(its),
                        res=np.asarray(res))
    print(f"wrote {path}: {len(cs)} steps")


def branch(c, f, a):
    """Per low face of every cell along axis a (periodic): the koren branch code on c^n for the
    sign of f: 0 ratio, 1 |den| < 1e-10 (r = 0), 2 both < 1e-10 (r = 1); and the upwind-pair /
    downwind unknown flags handled by the caller."""
    cm1, cm2, cp1 = (np.roll(c, 1, axis=a), np.roll(c, 2, axis=a), np.roll(c, -1, axis=a))
    pos = f > 0.0
    num = np.where(pos, cm1 - cm2, c - cp1)
    den = np.where(pos, c - cm1, cm1 - c)
    small_d = np.abs(den) < 1e-10
    small_n = np.abs(num) < 1e-10
    code = np.where(small_d & small_n, 2, np.where(small_d, 1, 0))
    return code, num, den


def analyse(pa, pb):
    A, B = np.load(pa), np.load(pb)
    unk = A["unk"] > 0.5
    cA, cB = A["c"], B["c"]
    n = min(len(cA), len(cB))
    d = np.array([np.max(np.abs(cA[k] - cB[k])[unk]) for k in range(n)])
    cmax = float(np.max(np.abs(cA[0])))
    jump = next((k for k in range(1, n) if d[k] > 1e3 * max(d[k - 1], 1e-30) and d[k] > 1e-14),
                None)
    print("per-step max|cA - cB| (absolute; max c ~ %.3g):" % cmax)
    for k in range(n):
        if k < 3 or (jump is not None and abs(k - jump) <= 2) or k == n - 1:
            print(f"  step {k + 1:4d}: {d[k]:.3e}")
    if jump is None:
        print("no jump found")
        return
    k = jump  # step jump+1 consumed c^n = c after step `jump`
    cnA, cnB = cA[k - 1], cB[k - 1]
    loc = np.unravel_index(np.argmax(np.where(unk, np.abs(cA[k] - cB[k]), 0.0)), unk.shape)
    print(f"jump at step {k + 1}: argmax cell {tuple(int(v) for v in loc)}, c^n there "
          f"{cnA[loc]:.3e}, |dc^n| there {abs(cnA[loc] - cnB[loc]):.3e}; "
          f"after: |dc| {abs(cA[k][loc] - cB[k][loc]):.3e}")
    for a, f in ((0, A["fx"]), (1, A["fy"])):
        u0, um1, um2, up1 = unk, np.roll(unk, 1, a), np.roll(unk, 2, a), np.roll(unk, -1, a)
        koren = (f != 0.0) & u0 & um1 & np.where(f > 0.0, um2, up1)
        bA, nA, dA = branch(cnA, f, a)
        bB, nB, dB = branch(cnB, f, a)
        diff = koren & (bA != bB)
        for idx in zip(*np.nonzero(diff)):
            dist = max(abs(int(idx[0]) - int(loc[0])), abs(int(idx[1]) - int(loc[1])))
            print(f"  axis {a} face {tuple(int(v) for v in idx)} (cheb. distance {dist} to the "
                  f"jump): branch A {bA[idx]} B {bB[idx]}; num {nA[idx]:.17e} / {nB[idx]:.17e}; "
                  f"den {dA[idx]:.17e} / {dB[idx]:.17e}; F/V {f[idx]:.3e}")
        print(f"axis {a}: koren faces {int(koren.sum())}, branch differs on {int(diff.sum())}")
    # the reach of the c^n differences near the threshold
    near = unk & (np.abs(cnA) < 1e-8) & (np.abs(cnA) > 1e-12)
    if np.any(near):
        rel = np.abs(cnA - cnB)[near]
        print(f"cells with 1e-12 < c^n < 1e-8: {int(near.sum())}, max |dc^n| {rel.max():.3e}")


def onestep(src, k, out, rtol=None):
    """Re-run step k+1 alone from run `src`'s c after step k, on a fresh solver (same setup as
    g9_case), and store c after it: two thread counts from the SAME c^n isolate what the step
    itself generates from what it amplifies."""
    import test_scalar_cutcell_gates as g
    Rh = 32
    s, h = g.annulus_case(Rh, g.g9_offsets(Rh)[0], "koren")
    geo = s.diagnostics.scalar_geometry("c")
    unk = geo["unknown"] > 0.5
    kap = geo["kappa"]
    ax, ay = geo["aperture_x"], geo["aperture_y"]
    full = unk & (kap >= 1.0)
    for a, ap in ((0, ax), (1, ay), (2, geo["aperture_z"])):
        full &= (ap >= 1.0) & (np.roll(ap, -1, axis=a) >= 1.0)
    fx, fy = g.annulus_fluxes(s, h)
    o = (np.maximum(np.roll(fx, -1, axis=0), 0) + np.maximum(-fx, 0) +
         np.maximum(np.roll(fy, -1, axis=1), 0) + np.maximum(-fy, 0))
    T = 2.0 * math.pi
    nrev = int(math.ceil(T * float(np.max(o[full])) / 0.5))
    s.set_dt(T / nrev)
    g.set_annulus_velocity(s, fx, fy)
    if rtol is not None:
        s.set_scalar_tolerance("c", rtol)
    s.set_field("c", np.asfortranarray(np.load(src)["c"][k - 1]))
    s.advance_scalars()
    cen = s.diagnostics.scalar_census("c")
    np.savez_compressed(out, c=np.array(s.get_field("c")), its=cen["krylov_iterations"],
                        res=cen["krylov_residual"])
    print(f"step {k + 1} from {src}: its {cen['krylov_iterations']} res "
          f"{cen['krylov_residual']:.17e}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 70)
    elif sys.argv[1] == "onestep":
        onestep(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        analyse(sys.argv[2], sys.argv[3])
