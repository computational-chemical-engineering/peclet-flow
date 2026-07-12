#!/usr/bin/env python3
"""Verification (flow, ghost projection): plane Poiseuille through an SDF channel with the
EXPERIMENTAL directional ghost-cell projection active (set_ghost_projection + cutcell_pressure).

The steady parabola u(y) = F/(2 mu)(y-ylo)(yhi-y) is exactly quadratic, so a second-order scheme
must reproduce it pointwise to solver tolerance. For this unidirectional flow the projection RHS
is identically zero (u* depends on y only, v = w = 0, and every ghost closure sees zero data), so
this gates that the ghost-projection machinery — overlay build, binary surrogate, BiCGStab — does
NOT corrupt an exact solution (classification/indexing bugs inject spurious divergence and break
the parabola). The closure-sign/theta physics is gated by tests/study/staggered_zh_ghostproj.py.

Single-rank only (ghost projection v1). Pattern: scripts/verify_poiseuille_flow.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                os.environ.get("SDFLOW_BUILD", "build_cuda2"))))
from peclet import flow  # noqa: E402


def channel_sdf(nx, ny, nz, ylo, yhi):
    gy = np.arange(ny, dtype=np.float64)
    sdf = np.empty((nx, ny, nz))
    sdf[:, :, :] = np.minimum(gy - ylo, yhi - gy)[None, :, None]
    return sdf


def run(N, ghost, rho=1.0, mu=0.1, dt=50.0, F=0.01, max_steps=400):
    nx, nz = 8, 8
    ny = N
    ylo = round(0.30 * ny) + 0.5  # non-integer walls -> cut cells
    yhi = round(0.70 * ny) + 0.5

    s = flow.Solver(nx, ny, nz)
    s.set_rho(rho)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0.0, 0.0)
    s.set_velocity_solver_params(200)
    s.set_pressure_pcg(True, 200, 1e-8)
    if ghost:
        s.set_ghost_projection(True)
    s.set_solid(channel_sdf(nx, ny, nz, ylo, yhi), cutcell_pressure=True)

    prev = 0.0
    for it in range(max_steps):
        s.step()
        u_now = float(s.get_u().max())
        if it > 5 and abs(u_now - prev) < 1e-10 * (abs(u_now) + 1e-12):
            break
        prev = u_now

    u = s.get_u()
    prof = u[nx // 2, :, nz // 2]
    gy = np.arange(ny, dtype=np.float64)
    fluid = (gy > ylo) & (gy < yhi)
    u_ana = (F / (2.0 * mu)) * (gy - ylo) * (yhi - gy)
    err = float(np.max(np.abs(prof[fluid] - u_ana[fluid])))
    return err, s.max_open_divergence()


def main():
    print("=== flow: Poiseuille + GHOST projection -- pointwise node error vs the parabola ===")
    print(f"{'mode':>9} {'Ny':>5} {'max|u - u_ana|':>16} {'max div':>12}")
    worst = 0.0
    for name, ghost in (("cutcell", False), ("ghost", True)):
        for N in (16, 32, 64):
            err, dv = run(N, ghost)
            print(f"{name:>9} {N:5d} {err:16.3e} {dv:12.3e}")
            if ghost:
                worst = max(worst, err)
    ok = worst < 1e-4
    print(f"  ghost-mode worst node error = {worst:.3e}  (exact-on-quadratic -> solver tol)")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
