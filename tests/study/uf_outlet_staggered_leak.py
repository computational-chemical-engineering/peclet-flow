#!/usr/bin/env python
"""Follow-up A (branch uf-outlet-diag): does advanceScalars' wrap of the staggered outlet face
(C[0].u at the outflow plane overwritten by the inlet plane) reach momentum?

Staggered inflow/wall/outflow channel with a transported scalar, N steps; prints the SHA-256 of
u/v/w/p/c and saves them to <out>.npz so two trees (with / without the fix) can be compared.
Cases: 'passive' (one-way scalar), 'boussinesq' (scalar feeds force_y), 'colo-passive'.

Run: OMP_NUM_THREADS=1 PYTHONPATH=<tree> python tests/study/uf_outlet_staggered_leak.py <out>
"""
import hashlib
import sys

import numpy as np

import peclet.flow as pf

NX, NY, NZ, STEPS = 48, 16, 4, 100


def run(case):
    cls = pf.SolverColocated if case.startswith("colo") else pf.Solver
    s = cls(NX, NY, NZ)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(0.5)
    s.set_advection(True); s.set_advection_scheme("koren")
    if cls is pf.SolverColocated:
        s.set_collocated_scheme("gauge-exact")
    s.set_domain_bc("-x", "inflow", (1.0, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow")
    s.set_domain_bc("-y", "wall"); s.set_domain_bc("+y", "wall")
    s.set_pressure_geometry(np.asfortranarray(np.full((NX, NY, NZ), 10.0)))
    s.set_pressure_pcg(True, 400, 1e-11)
    s.add_scalar("c", diffusivity=0.01, scheme="koren", iters=50)
    s.set_scalar_bc("c", "-x", "dirichlet", 1.0); s.set_scalar_bc("c", "+x", "neumann", 0.0)
    s.set_scalar_bc("c", "-y", "neumann", 0.0); s.set_scalar_bc("c", "+y", "neumann", 0.0)
    if case == "boussinesq":
        s.set_property_model("force_y", "boussinesq", "c", [1.0, 1e-3, 1.0, 0.5])
    for _ in range(STEPS):
        s.step()
    return {k: np.asarray(f).copy() for k, f in
            (("u", s.get_u()), ("v", s.get_v()), ("w", s.get_w()), ("p", s.get_p()),
             ("c", s.get_field("c")))}


if __name__ == "__main__":
    out = {}
    for case in ("passive", "boussinesq", "colo-passive"):
        d = run(case)
        hs = " ".join(f"{k}={hashlib.sha256(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest()[:12]}"
                      for k, v in d.items())
        print(f"{case:14s} {hs}")
        out.update({f"{case}_{k}": v for k, v in d.items()})
    np.savez(sys.argv[1], **out)
