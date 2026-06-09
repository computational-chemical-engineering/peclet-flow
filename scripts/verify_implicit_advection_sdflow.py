#!/usr/bin/env python3
"""Verification of the implicit-FOU deferred-correction advection (dcfd.set_implicit_advection). The
distributed solver is a full Navier-Stokes solver (Koren TVD advection). The default advection is
EXPLICIT (Picard-lagged) and therefore CFL-limited: at high Reynolds number / large dt it goes unstable.
The implicit-FOU mode solves the first-order-upwind part of advection implicitly (diagonally dominant ->
unconditionally stable for advection) and keeps the (Koren - FOU) correction explicit, so the scheme is
still Koren TVD at convergence but is robust at high Re, matching the production solver's deferred
correction. This checks both:
  (1) high Re / large dt: explicit BLOWS UP, implicit-FOU stays finite and bounded;
  (2) moderate Re (where explicit is stable): the two agree -> same Koren scheme at convergence.
Flow around a sphere in a periodic box (full 3-D NS + cut-cell IBM + cut-cell pressure). One GPU.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import sdflow  # noqa: E402

N = 32
NU = 0.1


def sphere_sdf(rfrac=0.3):  # sdf[x,y,z], negative inside the sphere
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    return np.sqrt((X - N / 2.0) ** 2 + (Y - N / 2.0) ** 2 + (Z - N / 2.0) ** 2) - N * rfrac


def run(implicit, dt, fx, n_steps, to_steady=False):
    sdf = sphere_sdf()
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0)            # grid units: rho=1, mu=NU -> nu=NU
    s.set_mu(NU)
    s.set_dt(dt)
    s.set_body_force(fx, 0.0, 0.0)
    s.set_advection(True)
    s.set_implicit_advection(implicit)
    s.set_outer_iterations(3)
    s.set_velocity_solver_params(80)  # implicit-FOU uses RB-GS (n_diff sweeps), not vel-MG
    s.set_pressure_pcg(True, max_iter=120, rtol=1e-9)
    s.set_solid(sdf, cutcell_pressure=True, galerkin=True)
    prev = 0.0
    for it in range(n_steps):  # np=1 demo
        s.step()
        u = s.get_u()
        if not np.isfinite(u).all():
            return None  # blew up
        um = float(u.mean())
        if to_steady and it > 8 and abs(um - prev) < 1e-6 * (abs(um) + 1e-15):
            break
        prev = um
    return s.get_u()


def main():
    print("=== implicit-FOU advection: high-Re stability + moderate-Re correctness ===")

    # (1) high Re / large dt: explicit should blow up, implicit-FOU should not
    print(f"  (1) high Re: dt=5, fx=0.02 (velocity reaches U~2 -> CFL = U*dt/dx >> 1)")
    ue = run(implicit=False, dt=5.0, fx=0.02, n_steps=30)
    ui = run(implicit=True, dt=5.0, fx=0.02, n_steps=30)
    expl_blew = ue is None
    impl_ok = ui is not None and np.isfinite(ui).all() and ui.max() < 1e3
    print(f"      explicit advection : {'BLEW UP (NaN/Inf)' if expl_blew else f'finite, U_max={ue.max():.3f}'}")
    print(f"      implicit-FOU       : {'finite, U_max=%.3f' % ui.max() if impl_ok else 'unstable'}")

    # (2) moderate Re (explicit stable): the two should agree -> same Koren scheme
    print(f"  (2) moderate Re: dt=5, fx=2e-4 (steady, both stable)")
    ue2 = run(implicit=False, dt=5.0, fx=2e-4, n_steps=300, to_steady=True)
    ui2 = run(implicit=True, dt=5.0, fx=2e-4, n_steps=300, to_steady=True)
    rel = abs(ui2.max() - ue2.max()) / ue2.max()
    print(f"      explicit U_max={ue2.max():.5f}   implicit-FOU U_max={ui2.max():.5f}   diff={rel*100:.2f}%")

    ok = expl_blew and impl_ok and rel < 0.03
    print(f"  result: {'PASS' if ok else 'FAIL'}  (implicit-FOU stable at high Re where explicit fails; "
          f"agrees with explicit at moderate Re)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
