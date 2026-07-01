#!/usr/bin/env python3
"""Upwind-convective velocity-MG on the DOMAIN-BC path (lid-driven cavity, Ghia Re=100), task #56.

This exercises the implicit-FOU advection + upwind-convective velocity multigrid on a problem with native
domain boundary conditions (no immersed solid) -- the cavity. The point is the CFL >> 1 regime: with
EXPLICIT advection a large dt is unstable (advective CFL = U*dt/dx); the implicit-FOU deferred correction
solves the first-order-upwind part implicitly (every MG level an M-matrix -> unconditionally stable for
advection) and keeps the (Koren - FOU) correction explicit, so the scheme is still Koren TVD at steady.

Checks:
  (1) reference: explicit advection at a SMALL dt converges to Ghia (ground truth);
  (2) at a LARGE dt (CFL >> 1): explicit BLOWS UP, while implicit-FOU + upwind vmg stays bounded and
      converges to the SAME Ghia centreline -> the upwind coarse operator is stable + correct at high CFL.
Quasi-2D (nz=4) so semi-coarsening builds a deep velocity-MG hierarchy. One GPU.
"""
import os
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                       os.environ.get("SDFLOW_BUILD", "build_mpi")))
from peclet import flow as sdflow  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_lid_cavity_sdflow import GHIA_Y, GHIA_U  # noqa: E402


def run(N=128, Re=100.0, U=1.0, nz=4, dt=1.0, max_steps=6000, implicit=False, vmg=False,
        vlevels=8, vcycles=4, vel_iter=60, outer=2):
    nu = U * N / Re
    s = sdflow.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(True)
    s.set_implicit_advection(implicit)
    s.set_outer_iterations(outer)
    s.set_domain_bc(0, 1); s.set_domain_bc(1, 1); s.set_domain_bc(2, 1)
    s.set_domain_bc(3, 2, U, 0.0, 0.0)          # +y face = moving lid
    s.set_velocity_solver_params(vel_iter)
    if vmg:
        s.set_velocity_multigrid(True, vlevels, vcycles)
    s.set_pressure_multigrid(True, levels=8)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((N, N, nz), 1e30))
    prev = None
    steps = max_steps
    for it in range(max_steps):
        s.step()
        u = np.asarray(s.get_u()).reshape((N, N, nz), order="F")
        if not np.isfinite(u).all() or np.abs(u).max() > 1e3:
            return None  # blew up
        if it % 100 == 0:
            cur = u[:, :, nz // 2].copy()
            if prev is not None:
                d = np.max(np.abs(cur - prev)) / (np.max(np.abs(cur)) + 1e-30)
                if d < 2e-5:
                    steps = it + 1; break
            prev = cur
    u = np.asarray(s.get_u()).reshape((N, N, nz), order="F")
    uc = 0.5 * (u[N // 2 - 1, :, nz // 2] + u[N // 2, :, nz // 2])
    yc = (np.arange(N) + 0.5) / N
    u_rms = float(np.sqrt(np.mean((np.interp(GHIA_Y, yc, uc) - GHIA_U) ** 2)))
    return dict(u_rms=u_rms, umin=float(uc.min()), div=float(s.max_open_divergence()), steps=steps)


def main():
    N = int(os.environ.get("VMG_N", 64))
    Re = 100.0
    print(f"=== upwind velocity-MG on domain-BC (lid cavity Re={Re:.0f}, N={N}, quasi-2D nz=4) ===")

    # (1) ground truth: explicit advection, small dt (stable) -> Ghia. Grid units: dx=1, lid U=1, so the
    #     advective CFL is just U*dt; dt=2 (CFL=2) is comfortably stable.
    ref = run(N=N, Re=Re, dt=2.0, implicit=False, vmg=False, max_steps=12000)
    print(f"  (1) explicit dt=2 (CFL=2, stable): u_rms={ref['u_rms']:.4f} "
          f"umin={ref['umin']:.4f} (Ghia -0.2058) div={ref['div']:.1e} steps={ref['steps']}")

    # (2) high CFL: explicit blows up (threshold ~dt=20); implicit-FOU + upwind vmg stays stable + reaches
    #     the same steady state. Grid units dx=1 -> advective CFL = U*dt (here 40).
    DT = float(os.environ.get("VMG_DT", 40.0))
    print(f"  (2) high CFL: dt={DT:.0f} (advective CFL = U*dt = {DT:.0f})")
    ex = run(N=N, Re=Re, dt=DT, implicit=False, vmg=False, max_steps=4000)
    iv = run(N=N, Re=Re, dt=DT, implicit=True, vmg=True, vlevels=8, vcycles=6, max_steps=4000)
    expl_blew = ex is None
    vmg_ok = iv is not None
    print(f"      explicit advection      : {'BLEW UP' if expl_blew else 'u_rms=%.4f umin=%.4f' % (ex['u_rms'], ex['umin'])}")
    if vmg_ok:
        print(f"      implicit-FOU + upwind vmg: u_rms={iv['u_rms']:.4f} umin={iv['umin']:.4f} "
              f"div={iv['div']:.1e} steps={iv['steps']}")
    else:
        print("      implicit-FOU + upwind vmg: UNSTABLE")

    # the implicit-FOU vmg solution must match the explicit ground-truth Ghia centreline
    match = vmg_ok and abs(iv["u_rms"] - ref["u_rms"]) < 0.02 and iv["div"] < 1e-4
    ok = expl_blew and vmg_ok and match
    print(f"  result: {'PASS' if ok else 'FAIL'}  (upwind vmg stable at CFL>>1 where explicit fails; "
          f"matches the Ghia ground truth)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
