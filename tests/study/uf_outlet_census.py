#!/usr/bin/env python
"""Diagnosis probe 2 (branch uf-outlet-diag): read the RAW outlet plane of the face field with no
scalar in the step (advanceScalars re-wraps the plane itself, on both grids).

outflow_backflow() reads the collocated uf_ (staggered C[0].u) at the outlet face index directly.
It only sees REVERSED flow, so the channel is driven backwards: the -x Dirichlet face has
u = -U and fluid enters through the +x open face. The census max_reverse is then max(-u_out); the
discrete-divergence answer is max(-U_out_required) from the inner faces (get_uf/vf/wf).

Run: OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=build_host python tests/study/uf_outlet_census.py
"""
import numpy as np

import peclet.flow as pf

NX, NY, NZ = 48, 16, 4


def required_outlet(s):
    uf, vf, wf = (np.asarray(s.get_uf()), np.asarray(s.get_vf()), np.asarray(s.get_wf()))
    i = NX - 1
    vtop = np.zeros((NY, NZ)); vtop[:-1] = vf[i, 1:, :]
    wtop = np.roll(wf[i], -1, axis=1)
    return uf[i] - (vtop - vf[i]) - (wtop - wf[i]), uf[0]


for name, cls, scheme in (("colo-gauge-exact", pf.SolverColocated, "gauge-exact"),
                          ("colo-embed", pf.SolverColocated, "embed"),
                          ("staggered", pf.Solver, None)):
    s = cls(NX, NY, NZ)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(0.5)
    s.set_advection(True); s.set_advection_scheme("koren")
    if scheme:
        s.set_collocated_scheme(scheme)
    s.set_domain_bc("-x", "inflow", (-1.0, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow")
    s.set_domain_bc("-y", "wall"); s.set_domain_bc("+y", "wall")
    s.set_pressure_geometry(np.asfortranarray(np.full((NX, NY, NZ), 10.0)))
    s.set_pressure_pcg(True, 400, 1e-11)
    print(f"--- {name}")
    for n in range(1, 201):
        s.step()
        if n in (1, 2, 10, 200):
            req, uin = required_outlet(s)
            ob = s.diagnostics.outflow_backflow()
            print(f"  step {n:3d}: census max_reverse = {ob['max_reverse']:.6f} "
                  f"reversed {ob['reversed_faces']}/{ob['outlet_faces']} | required: "
                  f"max(-u_out) = {(-req).max():.6f} min(-u_out) = {(-req).min():.6f} "
                  f"sum = {req.sum():.4f} | inlet face sum = {uin.sum():.4f} | "
                  f"max_open_div_proj = {s.max_open_divergence_projected():.2e}")
