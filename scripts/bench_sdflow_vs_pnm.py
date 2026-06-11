#!/usr/bin/env python3
"""Head-to-head single-GPU timing: the canonical `sdflow` solver vs the production `pnm_backend`, on the
SAME problem (Stokes flow through a periodic sphere packing). Both march to the same steady state (mean
velocity change < tol, checked every CHK steps); we report wall-time-to-steady, steps, ms/step, and the
converged mean velocity (to confirm they reach the same physics). Same grid units (rho=1, mu=nu), same
dt, so the comparison is fair. np=1 (pnm_backend is single-GPU).
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

N = 64
NU = 0.1
FX = 1e-3
DT = 30.0
TOL = 1e-5
CHK = 10
MAXS = 600


def packing_sdf():
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


def to_pnm(f):
    return np.ascontiguousarray(np.transpose(f, (2, 1, 0)))


def from_pnm(f):
    return np.transpose(np.asarray(f, np.float64), (2, 1, 0))


def march(step_fn, mean_fn):
    prev = 0.0
    step_fn()  # one untimed warm-up step (lazy init / first-touch allocations)
    t0 = time.perf_counter()
    steps = 0
    for it in range(MAXS):
        step_fn(); steps += 1
        if it % CHK == 0:
            m = mean_fn()
            if it > CHK and abs(m - prev) < TOL * (abs(m) + 1e-30):
                break
            prev = m
    wall = time.perf_counter() - t0
    return wall, steps, mean_fn()


def run_sdflow():
    sdf = packing_sdf()
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(DT); s.set_body_force(FX, 0, 0)
    s.set_advection(False)
    s.set_velocity_multigrid(True, levels=4, v_cycles=12)
    s.set_pressure_pcg(True, max_iter=120, rtol=1e-9)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="galerkin")
    return march(lambda: s.step(), lambda: float(s.get_u().mean()))


def run_pnm():
    sdf = packing_sdf()
    s = pnm_backend.CFDSolver([N, N, N], [1.0, 1.0, 1.0])
    s.initialize(to_pnm(sdf).astype(np.float32), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    s.set_rho(1.0); s.set_mu(NU); s.set_body_force(pnm_backend.float3(FX, 0, 0)); s.set_theta_(1.0)
    s.set_pressure_solver_params(200); s.set_velocity_solver_params(80)
    s.set_pressure_multigrid_enabled(True)
    return march(lambda: s.step(DT), lambda: float(from_pnm(s.get_u()).mean()))


def main():
    print(f"=== single-GPU timing: sdflow vs pnm_backend (sphere packing Stokes, N={N}, dt={DT}) ===")
    w_s, n_s, m_s = run_sdflow()
    w_p, n_p, m_p = run_pnm()
    print(f"  sdflow      : {w_s:6.2f} s to steady in {n_s:3d} steps  ({w_s/n_s*1e3:6.1f} ms/step)  <u>={m_s:.4e}")
    print(f"  pnm_backend : {w_p:6.2f} s to steady in {n_p:3d} steps  ({w_p/n_p*1e3:6.1f} ms/step)  <u>={m_p:.4e}")
    same = abs(m_s - m_p) < 0.05 * abs(m_p)
    print(f"  same steady physics (<u> within 5%): {same}   ({abs(m_s-m_p)/abs(m_p)*100:.2f}%)")
    print(f"  => sdflow is {w_p/w_s:.2f}x {'FASTER' if w_s < w_p else 'SLOWER'} to steady "
          f"({w_p/n_p/(w_s/n_s):.2f}x per step)")


if __name__ == "__main__":
    main()
