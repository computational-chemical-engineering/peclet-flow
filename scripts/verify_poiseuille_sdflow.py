#!/usr/bin/env python3
"""Verification (distributed solver): plane Poiseuille flow through an SDF-defined channel, driven by a
body force, with Robust-Scaled IBM no-slip at the (non-grid-aligned) walls. The steady centreline
velocity must match the analytic parabola U_max = (fx / 2nu) * (H/2)^2, and the error must shrink with
resolution. Mirrors the production scripts/verify_poiseuille.py but uses the distributed `dcfd` module
(works on one GPU as plain `python`, or multi-rank under `mpirun -np N python`).

All quantities are in grid units (spacing = 1), matching the distributed solver's convention.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build_mpi")))
import dcfd  # noqa: E402


def channel_sdf(nx, ny, nz, ylo, yhi):
    """Global SDF as a 3-D array sdf[x,y,z]; negative inside the solid walls."""
    gy = np.arange(ny, dtype=np.float64)
    sdf = np.empty((nx, ny, nz))
    sdf[:, :, :] = np.minimum(gy - ylo, yhi - gy)[None, :, None]
    return sdf


def run(N, nu=0.1, dt=50.0, fx=0.01, max_steps=400):
    nx, nz = 8, 8
    ny = N
    ylo = round(0.30 * ny) + 0.5  # non-integer walls -> cut cells
    yhi = round(0.70 * ny) + 0.5
    H = yhi - ylo

    s = dcfd.Solver(nx, ny, nz, nu, dt)
    s.set_body_force(fx, 0.0, 0.0)
    s.set_ibm_solid(channel_sdf(nx, ny, nz, ylo, yhi))           # Robust-Scaled no-slip walls
    s.set_velocity_multigrid(True, levels=3, v_cycles=20)

    # advance to steady state (diffusion only; x-independent flow is divergence-free)
    prev = 0.0
    for it in range(max_steps):
        s.step(n_diff=0, n_pois=0)
        u_now = float(s.get_u().max()) if s.rank() == 0 else 0.0
        if s.rank() == 0 and it > 5 and abs(u_now - prev) < 1e-7 * (abs(u_now) + 1e-12):
            break
        prev = u_now

    if s.rank() != 0:
        return None
    U_sim = float(s.get_u().max())  # get_u() is a 3-D array u[x,y,z]
    U_ana = (fx / (2.0 * nu)) * (H / 2.0) ** 2
    err = 100.0 * abs(U_sim - U_ana) / U_ana
    return ny, H, U_sim, U_ana, err


def main():
    print("=== distributed solver: Poiseuille through an SDF channel (IBM no-slip) ===")
    print(f"{'Ny':>5} {'H':>7} {'U_max(sim)':>12} {'U_max(ana)':>12} {'err %':>8}")
    errs = []
    for N in (16, 32, 64):
        res = run(N)
        if res is None:
            return
        ny, H, U_sim, U_ana, err = res
        print(f"{ny:5d} {H:7.1f} {U_sim:12.5f} {U_ana:12.5f} {err:8.3f}")
        errs.append(err)
    ok = errs[-1] < 2.0 and errs[-1] <= errs[0] + 1e-9  # accurate at high res and not worse than low res
    print(f"  result: {'PASS' if ok else 'FAIL'}  (error shrinks with resolution, <2% at N=64)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
