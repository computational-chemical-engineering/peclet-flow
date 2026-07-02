#!/usr/bin/env python3
"""Verification (sdflow): plane Poiseuille flow through an SDF-defined channel, driven by a body force,
with Robust-Scaled cut-cell IBM no-slip at the (non-grid-aligned) walls. The steady centreline velocity
must match the analytic parabola U_max = F*H^2 / (8*mu), and the error must shrink with resolution.

Uses the canonical `sdflow` module (one GPU as plain `python`, or multi-rank under `mpirun -np N python`).
Physical units: set_rho/set_mu, body force F is a force per unit volume (= -dp/dx). Grid spacing = 1.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build_mpi"))))
from peclet import flow as sdflow  # noqa: E402


def channel_sdf(nx, ny, nz, ylo, yhi):
    """Global SDF as a 3-D array sdf[x,y,z]; negative inside the solid walls."""
    gy = np.arange(ny, dtype=np.float64)
    sdf = np.empty((nx, ny, nz))
    sdf[:, :, :] = np.minimum(gy - ylo, yhi - gy)[None, :, None]
    return sdf


def run(N, rho=1.0, mu=0.1, dt=50.0, F=0.01, max_steps=400):
    nx, nz = 8, 8
    ny = N
    ylo = round(0.30 * ny) + 0.5  # non-integer walls -> cut cells
    yhi = round(0.70 * ny) + 0.5
    H = yhi - ylo

    s = sdflow.Solver(nx, ny, nz)
    s.set_rho(rho)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0.0, 0.0)              # force per unit volume (= -dp/dx)
    s.set_velocity_solver_params(200)          # simple IBM RB-GS velocity (the default; no multigrid)
    s.set_pressure_solver_params(1)            # x-independent flow is divergence-free -> projection is a no-op
    s.set_solid(channel_sdf(nx, ny, nz, ylo, yhi), cutcell_pressure=False)  # Robust-Scaled no-slip walls

    prev = 0.0
    for it in range(max_steps):
        s.step()
        u = s.get_u()                          # collective gather: ALL ranks must call it
        converged = False
        if s.rank() == 0:
            u_now = float(u.max())
            converged = it > 5 and abs(u_now - prev) < 1e-7 * (abs(u_now) + 1e-12)
            prev = u_now
        if s.bcast_from_root(converged):       # all ranks agree on the stop -> collectives stay matched
            break

    u = s.get_u()                              # collective; 3-D array u[x,y,z] on root, empty elsewhere
    _ = s.get_p()                              # collective; exercise the pressure path (p = rho/dt * phi)
    if s.rank() != 0:
        return None
    U_sim = float(u.max())
    U_ana = F * H * H / (8.0 * mu)
    err = 100.0 * abs(U_sim - U_ana) / U_ana
    return ny, H, U_sim, U_ana, err


def main():
    # All ranks must call run() for every N (it runs collective steps/gathers); only root has results.
    results = [run(N) for N in (16, 32, 64)]
    if results[0] is None:  # non-root rank: nothing to report
        return
    print("=== sdflow: Poiseuille through an SDF channel (rho/mu units, cut-cell IBM no-slip) ===")
    print(f"{'Ny':>5} {'H':>7} {'U_max(sim)':>12} {'U_max(ana)':>12} {'err %':>8}")
    errs = []
    for ny, H, U_sim, U_ana, err in results:
        print(f"{ny:5d} {H:7.1f} {U_sim:12.5f} {U_ana:12.5f} {err:8.3f}")
        errs.append(err)
    ok = errs[-1] < 2.0 and errs[-1] <= errs[0] + 1e-9
    print(f"  result: {'PASS' if ok else 'FAIL'}  (error shrinks with resolution, <2% at N=64)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
