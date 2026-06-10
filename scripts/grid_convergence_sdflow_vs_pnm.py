#!/usr/bin/env python3
"""Grid-convergence study: sdflow vs pnm_backend permeability on the SAME geometrically self-similar
sphere packing (R = 0.18*N) at N = 32, 64, 128. The packing is identical at every resolution, so the
DIMENSIONLESS permeability k/N^2 (k = mu*<u>/F, grid units) should converge to a fixed continuum value
as the cut cells resolve. The question: does the ~12% sdflow-vs-pnm gap at N=64 SHRINK with N (a
resolution/rate issue, converging to the same answer) or PERSIST (a genuine cut-cell discretisation
disagreement)? Each solver uses its accurate cut-cell mode. np=1 (pnm_backend is single-GPU).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import pnm_backend  # noqa: E402
import sdflow  # noqa: E402

NU = 0.1
FX = 1e-3
TOL = 1e-5


def packing_sdf(N):
    R = N * 0.18
    g = np.arange(N)
    cs = [(c + 0.5) * N / 2.0 for c in (0, 1)]
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    best = np.full((N, N, N), 1e30)
    for sx in cs:
        for sy in cs:
            for sz in cs:
                dx = X - sx; dx -= N * np.round(dx / N)
                dy = Y - sy; dy -= N * np.round(dy / N)
                dz = Z - sz; dz -= N * np.round(dz / N)
                best = np.minimum(best, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return best


def march(stepfn, meanfn, maxs=400):
    stepfn(); prev = 0.0; n = 0
    for it in range(maxs):
        stepfn(); n += 1
        if it % 10 == 0:
            m = meanfn()
            if it > 10 and abs(m - prev) < TOL * (abs(m) + 1e-30):
                break
            prev = m
    return n, meanfn()


def run_sdflow(N, dt):
    sdf = packing_sdf(N)
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(dt); s.set_body_force(FX, 0, 0); s.set_advection(False)
    s.set_velocity_multigrid(True, levels=4, v_cycles=12)
    s.set_pressure_pcg(True, max_iter=80, rtol=1e-6)          # accurate cut-cell pressure
    s.set_solid(sdf, cutcell_pressure=True, galerkin=True)
    n, m = march(lambda: s.step(), lambda: float(s.get_u().mean()))
    return n, m


def run_pnm(N, dt):
    sdf = packing_sdf(N)
    s = pnm_backend.CFDSolver([N, N, N], [1.0, 1.0, 1.0])
    s.initialize(np.ascontiguousarray(np.transpose(sdf, (2, 1, 0))).astype(np.float32), [0.] * 3, [1.] * 3)
    s.set_rho(1.0); s.set_mu(NU); s.set_body_force(pnm_backend.float3(FX, 0, 0)); s.set_theta_(1.0)
    s.set_pressure_solver_params(200); s.set_velocity_solver_params(80)
    s.set_pressure_multigrid_enabled(True)
    n, m = march(lambda: s.step(dt), lambda: float(np.transpose(np.asarray(s.get_u()), (2, 1, 0)).mean()))
    return n, m


def main():
    print("=== grid convergence: dimensionless permeability k/N^2 = mu*<u>/(F*N^2) ===")
    print(f"{'N':>4} | {'sdflow k/N^2':>13} {'pnm k/N^2':>11} {'rel diff':>9}")
    last = None
    for N in (32, 64, 128):
        dt = 30.0 if N <= 64 else 60.0
        _, us = run_sdflow(N, dt)
        _, up = run_pnm(N, dt)
        ks = NU * us / FX / N**2
        kp = NU * up / FX / N**2
        diff = abs(ks - kp) / abs(kp) * 100
        print(f"{N:>4} | {ks:13.5e} {kp:11.5e} {diff:8.2f}%")
        last = diff
    print(f"  => sdflow-vs-pnm relative difference at N=128: {last:.2f}%")
    print("     (shrinking with N -> resolution/rate issue, converging together;"
          " ~constant -> genuine scheme disagreement)")


if __name__ == "__main__":
    main()
