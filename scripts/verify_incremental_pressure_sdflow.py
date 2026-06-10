#!/usr/bin/env python3
"""Validate (and profile) the incremental-rotational pressure correction in sdflow against the production
pnm_backend (which uses the same scheme). Flow around a sphere in a periodic box, full 3-D NS (Re~30):
  (1) STEADY VELOCITY is unchanged by the pressure scheme: sdflow incremental-ON vs incremental-OFF
      (classical Chorin) agree to discretisation.
  (2) PRESSURE: sdflow's incremental pressure matches pnm_backend's pressure far better than classical
      Chorin does (classical Chorin carries an O(dt) splitting error + a spurious near-wall layer).
  (3) PROFILE: per-step cost of incremental vs classical (the overhead is a Phi exchange + a gradient
      kernel per Picard iteration + one potential-update kernel per step).
np=1 (pnm_backend is single-GPU). Grid units (rho=1, mu=nu).
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

N = 32
NU = 0.1


def sphere_sdf():  # sdf[x,y,z], negative inside the sphere
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    return np.sqrt((X - N / 2.0) ** 2 + (Y - N / 2.0) ** 2 + (Z - N / 2.0) ** 2) - N * 0.3


def to_pnm(f):
    return np.ascontiguousarray(np.transpose(f, (2, 1, 0)))


def from_pnm(f):
    return np.transpose(np.asarray(f, np.float64), (2, 1, 0))


def demean(p, fluid):
    p = p.copy()
    p[~fluid] = 0.0
    p[fluid] -= p[fluid].mean()
    return p


def run_sdflow(incremental, steps=400, dt=10.0, fx=2e-4):
    sdf = sphere_sdf()
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(dt)
    s.set_body_force(fx, 0.0, 0.0)
    s.set_advection(True); s.set_outer_iterations(3)
    s.set_incremental_pressure(incremental)
    s.set_velocity_solver_params(0)
    s.set_velocity_multigrid(True, levels=3, v_cycles=16)
    s.set_pressure_pcg(True, max_iter=120, rtol=1e-9)
    s.set_solid(sdf, cutcell_pressure=True, galerkin=True)
    for _ in range(5):
        s.step()  # warm up before timing
    t0 = time.perf_counter()
    for _ in range(steps):
        s.step()
    ms = (time.perf_counter() - t0) / steps * 1e3
    return s.get_u(), s.get_p(), ms


def run_pnm(steps=800, dt=5.0, fx=2e-4):
    sdf = sphere_sdf()
    s = pnm_backend.CFDSolver([N, N, N], [1.0, 1.0, 1.0])
    s.initialize(to_pnm(sdf).astype(np.float32), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    s.set_rho(1.0); s.set_mu(NU); s.set_body_force(pnm_backend.float3(fx, 0, 0)); s.set_theta_(1.0)
    s.set_pressure_solver_params(200); s.set_velocity_solver_params(80)
    s.set_pressure_multigrid_enabled(True)
    for _ in range(steps):
        s.step(dt)
    return from_pnm(s.get_u()), from_pnm(s.get_p())


def main():
    sdf = sphere_sdf()
    fluid = sdf > 0.5  # well inside the fluid, away from cut cells

    u_off, p_off, ms_off = run_sdflow(incremental=False)
    u_on, p_on, ms_on = run_sdflow(incremental=True)
    u_pnm, p_pnm = run_pnm()

    def rl2(a, b):
        return np.sqrt(np.mean((a[fluid] - b[fluid]) ** 2)) / (np.max(np.abs(b[fluid])) + 1e-30)

    vel = rl2(u_on, u_off)  # velocity: incremental vs classical (should ~match at steady state)
    p_off_d, p_on_d, p_pnm_d = demean(p_off, fluid), demean(p_on, fluid), demean(p_pnm, fluid)
    p_classical_vs_pnm = rl2(p_off_d, p_pnm_d)
    p_increment_vs_pnm = rl2(p_on_d, p_pnm_d)

    print("=== sdflow incremental-rotational pressure: validation vs production pnm_backend (sphere, NS) ===")
    print(f"  (1) steady velocity, incremental-ON vs classical-Chorin : rel L2 = {vel*100:.2f}%  (expect small)")
    print(f"  (2) pressure field vs pnm_backend (mean-removed, fluid):")
    print(f"        classical Chorin   : rel L2 = {p_classical_vs_pnm*100:6.2f}%")
    print(f"        incremental-rot.   : rel L2 = {p_increment_vs_pnm*100:6.2f}%   (should be much smaller)")
    print(f"  (3) profile: {ms_off:.2f} ms/step (classical)  vs  {ms_on:.2f} ms/step (incremental)"
          f"   overhead {100*(ms_on/ms_off-1):+.1f}%")
    vel_ok = vel < 0.03
    press_ok = p_increment_vs_pnm < 0.5 * p_classical_vs_pnm  # incremental clearly closer to production
    ok = vel_ok and press_ok
    print(f"  result: {'PASS' if ok else 'FAIL'}  (velocity unchanged; incremental pressure matches "
          f"production better)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
