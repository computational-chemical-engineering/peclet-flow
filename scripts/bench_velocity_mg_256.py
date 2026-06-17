#!/usr/bin/env python3
"""Velocity-MG benchmark on the DOMAIN-BC paths (lid cavity 256^3, backward-facing step high-res 2D).

Unlike the IBM packed case (which has a dt ceiling from the row-scaled cut/solid coarse-coupling), the
cavity/BFS have NO immersed solid, so the velocity-MG coarse operator is geometry-exact -> it should be
both (a) FASTER than RB-GS at high resolution (MG kills the stiff-diffusion error in O(1) V-cycles vs RB-GS
O(N) sweeps), and (b) free of any dt restriction (a standard Helmholtz MG is unconditionally stable).

Reports, per case:
  - SPEEDUP: wall-clock + step count to reach steady, RB-GS vs velocity-MG (diffusion-only) at the same dt,
    and the velocity-MG run with a LARGE dt (pseudo-transient acceleration -> far fewer steps).
  - NO-DT-RESTRICTION: implicit-FOU + upwind velocity-MG at a sweep of growing dt; must stay finite and keep
    converging (CFL = U*dt up to O(1000)), where explicit advection and the IBM packed case would diverge.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import sdflow  # noqa: E402


def cavity(N, nz, Re, dt, mode, U=1.0, nsteps=200, vcyc=4, vel_iter=60, outer=2, pit=40):
    """Lid-driven cavity (N x N x nz), run a FIXED nsteps; return wall-clock + the converged-so-far centreline
    umin. mode: 'rbgs' | 'vmg' (diffusion-only) | 'vmg_fou' (implicit-FOU + upwind)."""
    nu = U * N / Re
    s = sdflow.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(True)
    if mode == "vmg_fou":
        s.set_implicit_advection(True); s.set_outer_iterations(outer)
    for f in (0, 1, 2):
        s.set_domain_bc(f, 1)
    if nz > 1:
        s.set_domain_bc(4, 1); s.set_domain_bc(5, 1)
    s.set_domain_bc(3, 2, U, 0.0, 0.0)               # +y lid
    s.set_velocity_solver_params(vel_iter)
    if mode in ("vmg", "vmg_fou"):
        s.set_velocity_multigrid(True, 8, vcyc)
    s.set_pressure_multigrid(True, levels=8); s.set_pressure_solver_params(pit)
    s.set_pressure_geometry(np.full((N, N, nz), 1e30))
    t0 = time.time()
    for it in range(nsteps):
        s.step()
        if not np.isfinite(s.get_u()).all() or np.abs(s.get_u()).max() > 1e3:
            return dict(blew=True, wall=time.time() - t0, sps=(time.time() - t0) / (it + 1), umin=np.nan)
    wall = time.time() - t0
    u = np.asarray(s.get_u()).reshape((N, N, nz), order="F")
    uc = 0.5 * (u[N // 2 - 1, :, nz // 2] + u[N // 2, :, nz // 2])
    return dict(blew=False, wall=wall, sps=wall / nsteps, umin=float(uc.min()))


def main():
    N = int(os.environ.get("VMG_N", 256))
    nz = int(os.environ.get("VMG_NZ", N))             # full 3D by default; set VMG_NZ=4 for quasi-2D
    Re = float(os.environ.get("VMG_RE", 100.0))
    ns = int(os.environ.get("VMG_NSTEPS", 200))
    dts = float(os.environ.get("VMG_DT_SPEEDUP", 2.0))  # near the explicit-advection CFL limit -> stiff diffusion
    print(f"=== lid cavity {N}x{N}x{nz}  Re={Re:.0f}  ({ns} steps, dt={dts:g}) ===")

    # Reference steady centreline: vel-MG, many V-cycles -> the converged velocity solve each step.
    ref = cavity(N, nz, Re, dt=dts, mode="vmg", vcyc=10, nsteps=ns)["umin"]
    print(f"--- SPEEDUP (cost/step + convergence vs reference umin={ref:.4f}) ---")
    for label, kw in [("RB-GS  60 sweeps", dict(mode="rbgs", vel_iter=60)),
                      ("RB-GS 200 sweeps", dict(mode="rbgs", vel_iter=200)),
                      ("RB-GS 400 sweeps", dict(mode="rbgs", vel_iter=400)),
                      ("vel-MG  4 vcyc  ", dict(mode="vmg", vcyc=4))]:
        r = cavity(N, nz, Re, dt=dts, nsteps=ns, **kw)
        print(f"  {label}: {r['sps']*1000:6.0f} ms/step  wall={r['wall']:6.1f}s  umin={r['umin']:.4f} "
              f"(err {abs(r['umin']-ref):.4f})")

    print("--- NO DT RESTRICTION: implicit-FOU + upwind vel-MG, growing dt (CFL=U*dt) ---")
    for dt in [30.0, 100.0, 300.0, 1000.0]:
        r = cavity(N, nz, Re, dt=dt, mode="vmg_fou", vcyc=6, nsteps=120)
        tag = "BLEW UP" if r["blew"] else f"finite, umin={r['umin']:.4f}, {r['sps']*1000:.0f} ms/step"
        print(f"  dt={dt:7.0f} (CFL~{dt:.0f}): {tag}")


if __name__ == "__main__":
    main()
