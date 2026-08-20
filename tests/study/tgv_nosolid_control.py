#!/usr/bin/env python
"""No-solid periodic control (isolation ladder rung 0): 2D Taylor-Green on an all-fluid domain.

With no solid cells every collocated cut-cell device degenerates to its textbook form
(gpCenterGrad -> central difference, centerToFace -> plain 1/2-1/2, openness == 1), so
staggered-vs-collocated here tests ONLY the core scheme pair (exact vs ABC approximate
projection + advection).  Expected: both 2nd order in space, col-stag field difference
converging ~2nd order.  If this control FAILS the whole cut-cell diagnosis is re-scoped.

2D TGV is an exact NS solution: u = U0 sin(kx)cos(ky) F(t), v = -U0 cos(kx)sin(ky) F(t),
F = exp(-2 nu k^2 t).  Fixed Re = U0*N/nu and fixed CFL per rung; measured at t* = T_TRANSITS
tile transits.  Backward-Euler time error is O(dt) and shared by both solvers; the col-stag
difference cancels most of it, so ITS order is the clean spatial readout.

  SDFLOW_BUILD=build_omp3 python tests/study/tgv_nosolid_control.py [N ...]   # default 16 32 64
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow  # noqa: E402

U0, RE, CFL, T_TRANSITS, NZ = 1.0, 100.0, 0.2, 1.0, 4


def run(kind, N):
    nu = U0 * N / RE
    dt = CFL / U0
    steps = int(round(T_TRANSITS * N / (U0 * dt)))
    k = 2 * np.pi / N
    s = (flow.Solver if kind == "stag" else flow.SolverColocated)(N, N, NZ)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt)
    s.set_advection(True)
    s.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, max(2, int(np.log2(N)) - 2))
    s.set_pressure_pcg(True, 300, 1e-10)
    if kind != "stag":
        if hasattr(s, "set_collocated_scheme"):
            s.set_collocated_scheme("gauge-exact")
        else:
            s.set_face_interp(9)
    s.set_solid(np.full((N, N, NZ), 1e3, order="F"), cutcell_pressure=True,
                pressure_coarse="rediscretized")
    # initial condition sampled at each solver's own u/v locations
    cc = np.arange(N) + 0.5
    fc = np.arange(N) * 1.0                     # low-face coordinate of cell i
    xu, yu = (fc, cc) if kind == "stag" else (cc, cc)
    xv, yv = (cc, fc) if kind == "stag" else (cc, cc)
    u0 = np.asfortranarray(np.broadcast_to(
        (U0 * np.sin(k * xu)[:, None] * np.cos(k * yu)[None, :])[:, :, None], (N, N, NZ)).copy())
    v0 = np.asfortranarray(np.broadcast_to(
        (-U0 * np.cos(k * xv)[:, None] * np.sin(k * yv)[None, :])[:, :, None], (N, N, NZ)).copy())
    s.set_state(u0, v0, np.zeros((N, N, NZ), order="F"))
    for _ in range(steps):
        s.step()
    t = steps * dt
    F = np.exp(-2 * nu * k * k * t)
    uex = (U0 * np.sin(k * xu)[:, None] * np.cos(k * yu)[None, :]) * F
    U = np.asarray(s.get_u())[:, :, NZ // 2]
    err = float(np.sqrt(np.mean((U - uex) ** 2)) / (U0 * F))
    # center-sampled u for the cross-solver difference (staggered averaged to centers)
    Uc = 0.5 * (U + np.roll(U, -1, axis=0)) if kind == "stag" else U
    return err, Uc, F


if __name__ == "__main__":
    Ns = [int(x) for x in (sys.argv[1:] or [16, 32, 64])]
    print(f"{'N':>5} {'err_stag':>11} {'ord':>6} {'err_col9':>11} {'ord':>6} "
          f"{'|col-stag|':>11} {'ord':>6}")
    prev = None
    for N in Ns:
        es, Us, F = run("stag", N)
        ec, Uc, _ = run("gauge-exact", N)
        d = float(np.sqrt(np.mean((Uc - Us) ** 2)) / (U0 * F))
        o = ["", "", ""]
        if prev:
            lr = np.log(N / prev[0])
            o = [f"{np.log(a / b) / lr:+.2f}" if a > 0 and b > 0 else ""
                 for a, b in zip(prev[1:], (es, ec, d))]
        print(f"{N:>5} {es:>11.4e} {o[0]:>6} {ec:>11.4e} {o[1]:>6} {d:>11.4e} {o[2]:>6}",
              flush=True)
        prev = (N, es, ec, d)
