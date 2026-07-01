#!/usr/bin/env python3
"""Validation of the upwind-convective velocity-multigrid coarse operator (task #56, Phase 4 of
doc/velocity_mg_plan.md). When implicit-FOU advection is on, the velocity (momentum) solve is an
advection-dominated, non-symmetric Helmholtz system (I - nu*dt*Lap + dt*FOU(u^k)). The opt-in
upwind-convective vel-MG builds the coarse operators as anisotropic const-coeff diffusion + a coarse
first-order-upwind advection from the restricted advecting velocity, keeping every level an M-matrix.

This checks, on a sphere in a periodic box (full 3-D NS + cut-cell IBM + cut-cell pressure, one GPU):
  (1) the vel-MG path stays finite/bounded at high Re where the operator is advection-dominated;
  (2) at steady state it matches the RB-GS implicit-FOU solve (same operator, MG just sets the rate)
      -> drag-equivalent: U_max and U_mean agree to a tight tolerance.
The fine residual + smoother guarantee the exact fine (sharp-IBM) answer regardless of the coarse op,
so MG-on must converge to the same field as RB-GS.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))
from peclet import flow as sdflow  # noqa: E402

N = 32
NU = 0.1


def sphere_sdf(rfrac=0.3):  # sdf[x,y,z], negative inside the sphere
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    return np.sqrt((X - N / 2.0) ** 2 + (Y - N / 2.0) ** 2 + (Z - N / 2.0) ** 2) - N * rfrac


def run(vmg, dt, fx, n_steps, vlevels=3, vcycles=10, to_steady=False):
    sdf = sphere_sdf()
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0)
    s.set_mu(NU)
    s.set_dt(dt)
    s.set_body_force(fx, 0.0, 0.0)
    s.set_advection(True)
    s.set_implicit_advection(True)
    s.set_outer_iterations(3)
    s.set_velocity_solver_params(80)              # RB-GS sweeps when vmg off
    if vmg:
        s.set_velocity_multigrid(True, vlevels, vcycles)
    s.set_pressure_pcg(True, max_iter=120, rtol=1e-9)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="galerkin")
    prev = 0.0
    for it in range(n_steps):
        s.step()
        u = s.get_u()
        if not np.isfinite(u).all():
            return None
        um = float(u.mean())
        if to_steady and it > 8 and abs(um - prev) < 1e-7 * (abs(um) + 1e-15):
            break
        prev = um
    return s.get_u()


def main():
    print("=== upwind-convective velocity-MG: high-Re stability + RB-GS equivalence ===")

    # (1) high Re / large dt: advection-dominated; the upwind coarse op must stay an M-matrix (no NaN)
    print("  (1) high Re: dt=5, fx=0.02 (U~2 -> CFL >> 1, operator advection-dominated)")
    ur = run(vmg=False, dt=5.0, fx=0.02, n_steps=30)
    uv = run(vmg=True, dt=5.0, fx=0.02, n_steps=30)
    rbgs_ok = ur is not None and np.isfinite(ur).all()
    vmg_ok = uv is not None and np.isfinite(uv).all() and uv.max() < 1e3
    print(f"      RB-GS implicit-FOU : {'finite, U_max=%.3f' % ur.max() if rbgs_ok else 'unstable'}")
    print(f"      vel-MG implicit-FOU: {'finite, U_max=%.3f' % uv.max() if vmg_ok else 'unstable'}")

    # (2) moderate Re, steady: MG-on must converge to the SAME field as RB-GS (fine residual is identical)
    print("  (2) moderate Re: dt=5, fx=2e-4 (steady) -> RB-GS vs vel-MG agree")
    ur2 = run(vmg=False, dt=5.0, fx=2e-4, n_steps=400, to_steady=True)
    uv2 = run(vmg=True, dt=5.0, fx=2e-4, n_steps=400, to_steady=True)
    rel_max = abs(uv2.max() - ur2.max()) / ur2.max()
    rel_mean = abs(uv2.mean() - ur2.mean()) / abs(ur2.mean())
    print(f"      RB-GS  U_max={ur2.max():.6f} U_mean={ur2.mean():.6e}")
    print(f"      vel-MG U_max={uv2.max():.6f} U_mean={uv2.mean():.6e}")
    print(f"      diff: U_max {rel_max*100:.3f}%   U_mean {rel_mean*100:.3f}%")

    ok = rbgs_ok and vmg_ok and rel_max < 0.01 and rel_mean < 0.01
    print(f"  result: {'PASS' if ok else 'FAIL'}  (upwind vel-MG stable at high Re; "
          f"matches RB-GS at steady to <1%)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
