#!/usr/bin/env python
"""Diagnosis probe 3: SHA-256 of u/v/w/p after 200 steps of the forward collocated channel (no
scalar) -- to show whether the outlet face plane of uf_ reaches momentum. Run at one thread."""
import hashlib
import sys

import numpy as np

sys.argv += []
import peclet.flow as pf

NX, NY, NZ = 48, 16, 4
for scheme in ("gauge-exact", "embed"):
    s = pf.SolverColocated(NX, NY, NZ)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(0.5)
    s.set_advection(True); s.set_advection_scheme("koren")
    s.set_collocated_scheme(scheme)
    s.set_domain_bc("-x", "inflow", (1.0, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow")
    s.set_domain_bc("-y", "wall"); s.set_domain_bc("+y", "wall")
    s.set_pressure_geometry(np.asfortranarray(np.full((NX, NY, NZ), 10.0)))
    s.set_pressure_pcg(True, 400, 1e-11)
    for _ in range(200):
        s.step()
    h = hashlib.sha256()
    for f in (s.get_u(), s.get_v(), s.get_w(), s.get_p()):
        h.update(np.ascontiguousarray(np.asarray(f), dtype=np.float64).tobytes())
    ob = s.diagnostics.outflow_backflow()
    print(f"{scheme:12s} uvwp sha256 {h.hexdigest()[:24]}  u_out_max {np.asarray(s.get_u())[-1].max():.15f}")
