#!/usr/bin/env python
"""§11 G11 (WO-9): the cut-cell scalar on two Kokkos backends, CUDA against OpenMP.

Not a ctest: it needs two build trees. Run the same rows on each tree, then compare:

  OMP_NUM_THREADS=4 PYTHONPATH=build_dev  python tests/python/scalar_cutcell_backends.py dump omp.npz
                    PYTHONPATH=build_cuda python tests/python/scalar_cutcell_backends.py dump gpu.npz
  python tests/python/scalar_cutcell_backends.py compare omp.npz gpu.npz [omp_1thread.npz]

The optional third dump (the reference build at another OMP_NUM_THREADS) prints the reference
backend's own reproducibility floor beside the cross-backend difference (column 'floor').

The rows are the gate cases of tests/python/test_scalar_cutcell_gates.py, called unchanged (their
solver is captured by a recording subclass that logs every solve's BiCGStab count):

  g1_<Rh>          G1 Dirichlet sphere, steady (single phase), R/h 8 / 16 / 32
  g6c1_<Rh>        G6 case 1 (conjugate composite sphere, 30 BE steps), R/h 8 / 16
  g9_<scheme>_<Rh>[_c09] G9 annulus advection (transient, explicit/implicit faces, small cells),
                   one revolution at bulk Courant 0.5 (0.9), fou / koren, R_o/h 16 / 32
  gadva_<pe>       G-adv(a) (steady advective, A2 surrogate), R/h 16, Pe_h 1 / 10
  gadvb_<wall>_<pe> G-adv(b) closure (Neumann: mean-gradient mode, singular) and (b') Dirichlet,
                   32^3, Pe_h 1 / 10

G11 (doc/scalar_ibm_design.md §11): geometry fields <= 1e-12 relative; probe rungs identical;
solutions <= 1e-9 relative; iterations within +-1 per solve. "Relative" = the max-norm difference
over the max-norm of the reference array (solutions: over the unknowns of the reference).
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

GEOM_CELL = ("kappa", "aperture_x", "aperture_y", "aperture_z", "kappa_solid")
EXACT_CELL = ("unknown", "solid_unknown")
GEOM_FACET = ("centroid", "normal", "area")
EXACT_FACET = ("instance", "probe_rung")
SOL_FACET = ("flux", "wall_value")


def rows(g):
    """name -> thunk; each thunk runs one gate case of the gates module `g`."""
    r = {}
    for Rh in (8, 16, 32):
        r[f"g1_{Rh}"] = lambda Rh=Rh: g.g1_case(Rh, g.OFFSETS[0])
    for Rh in (8, 16):
        r[f"g6c1_{Rh}"] = lambda Rh=Rh: g.g6_case(Rh, g.OFFSETS[0], g.G6_CASES[0])
    for Rh in (16, 32):
        for sch in ("fou", "koren"):
            for cr in (0.5, 0.9):
                r[f"g9_{sch}_{Rh}" + ("" if cr == 0.5 else "_c09")] = (
                    lambda Rh=Rh, sch=sch, cr=cr: g.g9_case(Rh, g.g9_offsets(Rh)[0], sch, cr))
    for pe in (1.0, 10.0):
        r[f"gadva_{pe:g}"] = lambda pe=pe: g.gadv_a_case(16, pe)
        for wall in ("neumann", "dirichlet"):
            r[f"gadvb_{wall}_{pe:g}"] = lambda pe=pe, wall=wall: g.gadv_b_case(32, pe, wall)
    return r


def dump(path, only=None):
    try:
        import peclet.flow as pf
    except ImportError:
        print("SKIP: peclet.flow not importable")
        sys.exit(77)
    sys.path.insert(0, HERE)
    import test_scalar_cutcell_gates as g

    made, its = [], []

    class Recording(pf.Solver):
        """The gates' Solver, logging every cut-cell solve's BiCGStab count (scalar 'c')."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

        def _log(self):
            its.append(int(self.diagnostics.scalar_census("c")["krylov_iterations"]))

        def solve_scalar_steady(self, *a, **k):
            r = super().solve_scalar_steady(*a, **k)
            self._log()
            return r

        def advance_scalars(self, *a, **k):
            r = super().advance_scalars(*a, **k)
            self._log()
            return r

    g.pf.Solver = Recording  # the gates module builds every case through pf.Solver
    out = {}
    for name, run in rows(g).items():
        if only and name not in only:
            continue
        made.clear()
        its.clear()
        t0 = time.time()
        run()
        dt = time.time() - t0
        s = made[-1]
        geo = s.diagnostics.scalar_geometry("c")
        for k in GEOM_CELL + EXACT_CELL:
            if k in geo:
                out[f"{name}/geo/{k}"] = np.asarray(geo[k])
        fac = s.diagnostics.scalar_facets("c")
        for k in GEOM_FACET + EXACT_FACET + SOL_FACET:
            out[f"{name}/fac/{k}"] = np.asarray(fac[k])
        out[f"{name}/sol/c"] = np.asarray(s.get_field("c"))
        if "solid_unknown" in geo and np.any(np.asarray(geo["solid_unknown"]) > 0.5):
            out[f"{name}/sol/c_solid"] = np.asarray(s.get_field("c_solid"))
        out[f"{name}/its"] = np.asarray(its, dtype=np.int64)
        cen = s.diagnostics.scalar_census("c")
        pr = cen["probe_rungs"]
        out[f"{name}/rungs"] = np.asarray([*pr["fluid"], *pr.get("solid", ())], dtype=np.int64)
        out[f"{name}/time"] = np.asarray(dt)
        print(f"{name:20s} {dt:7.1f} s  solves {len(its):4d}  iterations {sum(its):6d}  "
              f"rungs {out[f'{name}/rungs'].tolist()}", flush=True)
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({len(out)} arrays)")


def rel(a, b, mask=None):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if mask is not None:
        a, b = a[mask], b[mask]
    fin = np.isfinite(a)
    if not np.array_equal(fin, np.isfinite(b)):
        return float("inf")
    a, b = a[fin], b[fin]
    if a.size == 0:
        return 0.0
    sc = float(np.max(np.abs(a)))
    d = float(np.max(np.abs(a - b)))
    return d / sc if sc > 0.0 else d


def compare(pa, pb, pc=None):
    A, B = np.load(pa), np.load(pb)
    C = np.load(pc) if pc else None
    names = sorted({k.split("/")[0] for k in A.files})
    bad = []
    print(f"{'row':20s} {'geom':>9s} {'exact':>5s} {'solution':>9s} {'its diff':>8s} "
          f"{'solves':>6s} {'t_ref':>7s} {'t_new':>7s}")
    for n in names:
        if f"{n}/its" not in B.files:
            print(f"{n:20s} missing in {pb}")
            bad.append(n)
            continue
        g = max([rel(A[k], B[k]) for k in A.files
                 if k.startswith(f"{n}/") and k.split("/")[-1] in GEOM_CELL + GEOM_FACET] or [0.0])
        ex = all(np.array_equal(A[k], B[k]) for k in A.files
                 if k.startswith(f"{n}/") and k.split("/")[-1] in EXACT_CELL + EXACT_FACET)
        ex = ex and np.array_equal(A[f"{n}/rungs"], B[f"{n}/rungs"])
        unk = A[f"{n}/geo/unknown"] > 0.5
        sol = rel(A[f"{n}/sol/c"], B[f"{n}/sol/c"], unk)
        if f"{n}/sol/c_solid" in A.files:
            sol = max(sol, rel(A[f"{n}/sol/c_solid"], B[f"{n}/sol/c_solid"],
                               A[f"{n}/geo/solid_unknown"] > 0.5))
        sol = max(sol, max(rel(A[f"{n}/fac/{k}"], B[f"{n}/fac/{k}"]) for k in SOL_FACET))
        floor = ""
        if C is not None and f"{n}/sol/c" in C.files:
            floor = f"  floor {rel(A[f'{n}/sol/c'], C[f'{n}/sol/c'], unk):9.2e}"
        ia, ib = A[f"{n}/its"], B[f"{n}/its"]
        di = int(np.max(np.abs(ia - ib))) if ia.shape == ib.shape else 10**9
        ok = g <= 1e-12 and ex and sol <= 1e-9 and di <= 1
        print(f"{n:20s} {g:9.2e} {'yes' if ex else 'NO':>5s} {sol:9.2e} {di:8d} {ia.size:6d} "
              f"{float(A[f'{n}/time']):7.1f} {float(B[f'{n}/time']):7.1f}  {'ok' if ok else 'FAIL'}{floor}")
        if not ok:
            bad.append(n)
    print("G11:", "PASS" if not bad else f"FAIL {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "dump":
        dump(sys.argv[2], set(sys.argv[3:]) or None)
    elif len(sys.argv) in (4, 5) and sys.argv[1] == "compare":
        sys.exit(compare(*sys.argv[2:]))
    else:
        print(__doc__)
        sys.exit(2)
