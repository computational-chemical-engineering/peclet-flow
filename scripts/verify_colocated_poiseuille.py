#!/usr/bin/env python3
"""Phase-2 verification (collocated grid): plane Poiseuille flow through an SDF channel, driven by a body
force, with Robust-Scaled cut-cell IBM no-slip walls and NO pressure projection (cutcell_pressure=False).

Runs the SAME setup on both grids: sdflow.Solver (staggered) and sdflow.SolverColocated (cell-centered).
The steady centreline velocity must match the analytic parabola U_max = F*H^2/(8*mu) and the error must
shrink with resolution on the collocated grid too. The u-component's wall-normal (y) location is the cell
centre on BOTH grids, so the two solvers should agree closely. This exercises the collocated {0,0,0} IBM
offset + implicit diffusion path (advection is ~0 for unidirectional flow; it is stressed in phase 3).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build"))))
import sdflow  # noqa: E402


def channel_sdf(nx, ny, nz, ylo, yhi):
    gy = np.arange(ny, dtype=np.float64)
    sdf = np.empty((nx, ny, nz))
    sdf[:, :, :] = np.minimum(gy - ylo, yhi - gy)[None, :, None]
    return sdf


def run(SolverCls, N, rho=1.0, mu=0.1, dt=50.0, F=0.01, max_steps=400):
    nx, nz = 8, 8
    ny = N
    ylo = round(0.30 * ny) + 0.5
    yhi = round(0.70 * ny) + 0.5
    H = yhi - ylo

    s = SolverCls(nx, ny, nz)
    s.set_rho(rho)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0.0, 0.0)
    s.set_velocity_solver_params(200)
    s.set_solid(channel_sdf(nx, ny, nz, ylo, yhi), cutcell_pressure=False)

    prev = 0.0
    for it in range(max_steps):
        s.step()
        u_now = float(s.get_u().max())
        if it > 5 and abs(u_now - prev) < 1e-7 * (abs(u_now) + 1e-12):
            break
        prev = u_now

    U_sim = float(s.get_u().max())
    U_ana = F * H * H / (8.0 * mu)
    err = 100.0 * abs(U_sim - U_ana) / U_ana
    return ny, H, U_sim, U_ana, err


def main():
    print("=== sdflow phase-2: collocated vs staggered Poiseuille (cut-cell IBM, no pressure) ===")
    print(f"{'Ny':>5} {'H':>6} {'U_stag':>11} {'U_coloc':>11} {'U_ana':>11} {'err_st%':>8} {'err_co%':>8} {'|st-co|':>9}")
    errs_co = []
    ok_all = True
    for N in (16, 32, 64):
        _, H, U_st, U_an, e_st = run(sdflow.Solver, N)
        _, _, U_co, _, e_co = run(sdflow.SolverColocated, N)
        d = abs(U_st - U_co)
        print(f"{N:5d} {H:6.1f} {U_st:11.5f} {U_co:11.5f} {U_an:11.5f} {e_st:8.3f} {e_co:8.3f} {d:9.2e}")
        errs_co.append(e_co)
    ok_all = errs_co[-1] < 2.0 and errs_co[-1] <= errs_co[0] + 1e-9
    print(f"  result: {'PASS' if ok_all else 'FAIL'}  (collocated error shrinks with resolution, <2% at N=64)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
