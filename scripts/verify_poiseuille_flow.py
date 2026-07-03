#!/usr/bin/env python3
"""Verification (flow): plane Poiseuille flow through an SDF-defined channel, driven by a body force, with
Robust-Scaled cut-cell IBM no-slip at the (non-grid-aligned) walls. The steady profile is the parabola
u(y) = F/(2 mu) (y - ylo)(yhi - y); because it is exactly quadratic and a second-order scheme is exact on
quadratics, the cut-cell solution must match it AT EVERY GRID NODE to solver tolerance -- at every
resolution and on both the staggered and collocated meshes.

We therefore verify POINTWISE: max_node |u - u_analytic|. (The old version of this script compared the
discrete u.max() to the continuum peak U_max = F H^2/(8 mu); with half-integer walls the channel centre
sits 0.5h from the nearest node, so u.max() is a fixed amount below U_max and dividing by U_max fabricated
a "converging" error -- a sampling artifact, not discretization error. The pointwise metric below actually
tests method order and would catch a genuine first-order regression.)

Uses the canonical `peclet.flow` module (one GPU as plain `python`, or multi-rank under `mpirun -np N`).
Physical units: set_rho/set_mu, body force F is a force per unit volume (= -dp/dx). Grid spacing = 1.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build_mpi"))))
from peclet import flow  # noqa: E402


def channel_sdf(nx, ny, nz, ylo, yhi):
    """Global SDF as a 3-D array sdf[x,y,z]; negative inside the solid walls."""
    gy = np.arange(ny, dtype=np.float64)
    sdf = np.empty((nx, ny, nz))
    sdf[:, :, :] = np.minimum(gy - ylo, yhi - gy)[None, :, None]
    return sdf


def run(N, Solver, rho=1.0, mu=0.1, dt=50.0, F=0.01, max_steps=400):
    """Solve one channel case at wall-to-wall resolution N; return the pointwise node error vs the parabola."""
    nx, nz = 8, 8
    ny = N
    ylo = round(0.30 * ny) + 0.5  # non-integer walls -> cut cells
    yhi = round(0.70 * ny) + 0.5
    H = yhi - ylo

    s = Solver(nx, ny, nz)
    s.set_rho(rho)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0.0, 0.0)              # force per unit volume (= -dp/dx)
    s.set_velocity_solver_params(200)          # IBM RB-GS velocity solve
    s.set_pressure_solver_params(1)            # x-independent flow is divergence-free -> projection is a no-op
    s.set_solid(channel_sdf(nx, ny, nz, ylo, yhi), cutcell_pressure=False)  # Robust-Scaled no-slip walls

    prev = 0.0
    for it in range(max_steps):
        s.step()
        u = s.get_u()                          # collective gather: ALL ranks must call it
        stop = False
        if s.rank() == 0:
            u_now = float(u.max())
            stop = it > 5 and abs(u_now - prev) < 1e-10 * (abs(u_now) + 1e-12)
            prev = u_now
        if s.bcast_from_root(stop):
            break

    u = s.get_u()
    _ = s.get_p()                              # exercise the pressure path (collective)
    if s.rank() != 0:
        return None
    prof = u[nx // 2, :, nz // 2]
    gy = np.arange(ny, dtype=np.float64)
    fluid = (gy > ylo) & (gy < yhi)
    u_ana = (F / (2.0 * mu)) * (gy - ylo) * (yhi - gy)
    node_err = float(np.max(np.abs(prof[fluid] - u_ana[fluid])))
    return ny, H, node_err


def main():
    """Run staggered + collocated at a few N, print the pointwise node error, exit non-zero on failure."""
    print("=== flow: Poiseuille through an SDF channel -- POINTWISE node error vs the exact parabola ===")
    print(f"{'mesh':>11} {'Ny':>5} {'H':>7} {'max|u - u_analytic|':>22}")
    worst = 0.0
    for name, Solver in (("staggered", flow.Solver), ("collocated", flow.SolverColocated)):
        for N in (16, 32, 64):
            r = run(N, Solver)
            if r is None:
                return                          # non-root rank
            ny, H, err = r
            print(f"{name:>11} {ny:5d} {H:7.1f} {err:22.3e}")
            worst = max(worst, err)
    # A 2nd-order scheme is exact on a quadratic: the node error is pure solver tolerance, ~1e-7 at these
    # resolutions. A first-order regression (or a broken cut-cell closure) would blow this to O(1e-2)+.
    ok = worst < 1e-4
    print(f"  worst-case node error = {worst:.3e}   (exact-on-quadratic -> solver tolerance)")
    print(f"  result: {'PASS' if ok else 'FAIL'}  (cut-cell IBM reproduces the parabola pointwise, both meshes)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
