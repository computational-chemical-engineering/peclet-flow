#!/usr/bin/env python
"""Diagnosis probe (branch uf-outlet-diag): does the collocated projected face field uf_ carry, on
the OUTLET face plane, the flux the projection solved for?

The outlet face (index nx along x) is a ghost index of uf_, so get_uf() cannot read it. A passive
scalar c == 1 with zero diffusivity and first-order upwind reads it exactly: in the last column
    c_new = 1 - dt * (U_out_raw - U_out_required)            (cell units, h = 1)
where U_out_required closes the last cell's discrete divergence from the inner faces get_uf/vf/wf
return. Every other cell changes by dt * div(face field), i.e. by the projection residual.

Run: OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=build_host python tests/study/uf_outlet_diag.py
"""
import sys

import numpy as np

import peclet.flow as pf

NX, NY, NZ = 48, 16, 4
U_IN, MU, DT = 1.0, 0.1, 0.5


def channel(cls, scheme=None, steps=200):
    s = cls(NX, NY, NZ)
    s.set_rho(1.0); s.set_mu(MU); s.set_dt(DT)
    s.set_advection(True); s.set_advection_scheme("koren")
    if scheme is not None:
        s.set_collocated_scheme(scheme)
    s.set_domain_bc("-x", "inflow", (U_IN, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow")
    s.set_domain_bc("-y", "wall"); s.set_domain_bc("+y", "wall")
    s.set_pressure_geometry(np.asfortranarray(np.full((NX, NY, NZ), 10.0)))
    s.set_pressure_pcg(True, 400, 1e-11)
    for _ in range(steps):
        s.step()
    return s


def required_outlet(s):
    """Outlet flux that closes the last cell's divergence, from the inner faces."""
    uf, vf, wf = (np.asarray(s.get_uf()), np.asarray(s.get_vf()), np.asarray(s.get_wf()))
    i = NX - 1
    vtop = np.zeros((NY, NZ)); vtop[:-1] = vf[i, 1:, :]       # +y wall face = 0
    wtop = np.roll(wf[i], -1, axis=1)                          # z periodic
    return uf[i] - (vtop - vf[i]) - (wtop - wf[i]), uf[0]


def probe(name, s):
    req_before, uin = required_outlet(s)
    s.add_scalar("c", diffusivity=0.0, scheme="fou", iters=1)
    s.set_scalar_bc("c", "-x", "dirichlet", 1.0)
    s.set_scalar_bc("c", "+x", "neumann", 0.0)
    s.set_field("c", np.asfortranarray(np.ones((NX, NY, NZ))))
    u_last = np.asarray(s.get_u())[NX - 1].copy()
    s.step()                       # the scalar rides the face field of THIS step's projection
    c = np.asarray(s.get_field("c"))
    req, uin = required_outlet(s)
    mism = (1.0 - c[NX - 1]) / DT  # = U_out_raw - U_out_required
    raw = req + mism
    j = slice(None)
    print(f"--- {name}")
    print(f"  inner columns: max|c-1| = {np.abs(c[:NX - 1] - 1).max():.3e}   "
          f"max_open_divergence_projected = {s.max_open_divergence_projected():.3e}")
    d = np.abs(c[:NX - 1] - 1); loc = np.unravel_index(np.argmax(d), d.shape)
    print(f"  inner max at {loc}, c there {c[loc]:.4f}; per-column max|c-1| cols 0,1,2,NX-3,NX-2: "
          + " ".join(f"{v:.1e}" for v in d.max(axis=(1, 2))[[0, 1, 2, NX - 3, NX - 2]]))
    print(f"  outlet column: min c = {c[NX-1].min():.4f}  max c = {c[NX-1].max():.4f}")
    print(f"  mass: inflow sum uf[0] = {uin.sum():.6f}  required outflow = {req.sum():.6f}  "
          f"raw outflow = {raw.sum():.6f}  (per-z-slice: in {uin.sum()/NZ:.4f}, raw {raw.sum()/NZ:.4f})")
    k = 0
    print("  y | u_in face | u_last cell | required out | raw out (from c) | raw - u_in_face")
    for y in range(NY):
        print(f"  {y:2d} | {uin[y,k]: .5f} | {u_last[y,k]: .5f} | {req[y,k]: .6f} | {raw[y,k]: .6f} |"
              f" {raw[y,k]-uin[y,k]: .3e}")
    return c, req, raw


if __name__ == "__main__":
    which = sys.argv[1:] or ["colo-default", "colo-gauge-exact", "colo-embed", "staggered"]
    steps = 200
    for w in which:
        if w == "staggered":
            s = channel(pf.Solver, steps=steps)
        elif w == "colo-default":
            s = channel(pf.SolverColocated, steps=steps)
        else:
            s = channel(pf.SolverColocated, scheme=w.split("colo-")[1], steps=steps)
        probe(w, s)
