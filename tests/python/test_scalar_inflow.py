#!/usr/bin/env python
"""Gate: a transported scalar enters through a Dirichlet inflow face with the prescribed flux.

Why this test exists. advanceScalars froze c^n into cOld BEFORE filling c's ghosts, and the
explicit advection reads cOld's ghosts. set_field (and the coupling drivers, and the VoF energy
transport) write inner cells only, so the first step after such a write upwinded the inflow from
stale values -- zeros on a fresh scalar: c = 1 at a Dirichlet-1 inlet fell to 0.5 in one step
(doc/uf_outlet_fix.md section B).

Checks:
  1. a c == 1 scalar with a Dirichlet-1 inlet stays 1 in the INLET column after one step, both
     grids, first-order upwind and Koren;
  2. a 1-D plug: uniform flow U through a box periodic in y and z, c = 0 initially, c = 1 at the
     inlet. Until the front reaches the outlet the inventory is exactly U t (the inflow carries
     U * 1 per unit area), and the c = 1/2 crossing sits at x = U t to within a cell.

Run:  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \
          python tests/python/test_scalar_inflow.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

MU, DT = 0.1, 0.5


def box(pf, cls, nx, ny, nz, walls):
    s = cls(nx, ny, nz)
    s.set_rho(1.0); s.set_mu(MU); s.set_dt(DT)
    s.set_advection(True); s.set_advection_scheme("koren")
    if cls is pf.SolverColocated:
        s.set_collocated_scheme("gauge-exact")
    s.set_domain_bc("-x", "inflow", (1.0, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow")
    if walls:
        s.set_domain_bc("-y", "wall"); s.set_domain_bc("+y", "wall")
    s.set_pressure_geometry(np.asfortranarray(np.full((nx, ny, nz), 10.0)))
    s.set_pressure_pcg(True, 400, 1e-11)
    return s


def check_inlet_column(pf, cls, scheme):
    nx, ny, nz = 24, 12, 4
    s = box(pf, cls, nx, ny, nz, walls=True)
    for _ in range(5):
        s.step()
    s.add_scalar("c", diffusivity=0.0, scheme=scheme, iters=1)
    s.set_scalar_bc("c", "-x", "dirichlet", 1.0)
    s.set_scalar_bc("c", "+x", "neumann", 0.0)
    s.set_field("c", np.asfortranarray(np.ones((nx, ny, nz))))
    s.step()
    c = np.asarray(s.get_field("c"))
    err = float(np.abs(c[: nx - 1] - 1.0).max())  # the outlet column is test_open_face_plane's
    assert err <= 1e-10, f"{cls.__name__}/{scheme}: c = 1 moved by {err:.3e} upstream of the outlet"
    return err


def check_plug(pf, cls, scheme, steps=40):
    nx, ny, nz = 64, 4, 4
    s = box(pf, cls, nx, ny, nz, walls=False)
    s.add_scalar("c", diffusivity=0.0, scheme=scheme, iters=1)
    s.set_scalar_bc("c", "-x", "dirichlet", 1.0)
    s.set_scalar_bc("c", "+x", "neumann", 0.0)
    s.set_field("c", np.asfortranarray(np.zeros((nx, ny, nz))))
    for _ in range(steps):
        s.step()
    c = np.asarray(s.get_field("c")).mean(axis=(1, 2))
    t = steps * DT
    inv = float(c.sum())                       # cell units: inventory per unit cross-section
    k = int(np.argmax(c < 0.5))                # first cell below 1/2
    xf = k - 0.5 + (c[k - 1] - 0.5) / (c[k - 1] - c[k]) * 1.0 if k > 0 else 0.0
    assert abs(inv - t) <= 1e-9, f"{cls.__name__}/{scheme}: inventory {inv:.12f} != U t = {t}"
    assert abs(xf - t) <= 1.0, f"{cls.__name__}/{scheme}: c = 1/2 at x = {xf:.3f}, U t = {t}"
    return inv - t, xf - t


def main():
    try:
        import peclet.flow as pf
    except ImportError:
        print("SKIP: peclet.flow not importable (set PYTHONPATH to the build tree)")
        return 77
    try:
        for cls in (pf.SolverColocated, pf.Solver):
            for scheme in ("fou", "koren"):
                e = check_inlet_column(pf, cls, scheme)
                di, dx = check_plug(pf, cls, scheme)
                print(f"  {cls.__name__:16s} {scheme:5s} inlet |c-1| {e:.2e}  plug inventory - Ut "
                      f"{di:+.2e}  front - Ut {dx:+.3f}")
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    print("OK: the scalar enters with the prescribed flux")
    return 0


if __name__ == "__main__":
    sys.exit(main())
