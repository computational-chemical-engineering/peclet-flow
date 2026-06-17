#!/usr/bin/env python3
"""Velocity-MG benchmark on the backward-facing step (high-resolution 2D; BFS is a channel, not a cube).

Same two questions as the cavity bench: does the velocity-MG (a) speed up the march to steady at high
resolution, and (b) stay free of any dt restriction (no immersed solid -> geometry-exact coarse operator)?
Re_S = U_in*S/nu; the step is the inlet condition (set_domain_bc_profile), +x outflow, -y/+y no-slip walls.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import sdflow  # noqa: E402


def inlet_profile(H, S, nz, U_in):
    prof = np.zeros((H, nz, 3))
    yc = np.arange(H) + 0.5
    eta = (yc - S) / S
    up = yc > S
    prof[up, :, 0] = (6.0 * U_in * eta * (1.0 - eta))[up, None]
    return prof


def reattach(u_bottom):
    rev = False
    for i in range(1, len(u_bottom)):
        if u_bottom[i] < 0.0:
            rev = True
        elif rev and u_bottom[i] >= 0.0:
            return (i - 1) + u_bottom[i - 1] / (u_bottom[i - 1] - u_bottom[i])
    return 0.0


def bfs(S, Re, dt, mode, Lr=12, U_in=1.0, nz=4, max_steps=6000, tol=2e-5, vcyc=4, vel_iter=60, outer=2):
    H, L, nu = 2 * S, Lr * S, U_in * S / Re
    s = sdflow.Solver(L, H, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(True)
    if mode == "vmg_fou":
        s.set_implicit_advection(True); s.set_outer_iterations(outer)
    s.set_domain_bc_profile(0, inlet_profile(H, S, nz, U_in))
    s.set_domain_bc(1, 3); s.set_domain_bc(2, 1); s.set_domain_bc(3, 1)
    s.set_velocity_solver_params(vel_iter)
    if mode in ("vmg", "vmg_fou"):
        s.set_velocity_multigrid(True, 8, vcyc)
    s.set_pressure_multigrid(True, levels=8); s.set_pressure_solver_params(80)
    prev = None; steps = max_steps; t0 = time.time()
    for it in range(max_steps):
        s.step()
        u = s.get_u()
        if not np.isfinite(u).all() or np.abs(u).max() > 1e3:
            return dict(blew=True, steps=it, wall=time.time() - t0, xr=np.nan)
        if it % 25 == 24:
            uu = np.asarray(u).reshape((L, H, nz), order="F")
            cur = uu[:, :, nz // 2].copy()
            if prev is not None and np.max(np.abs(cur - prev)) / (np.max(np.abs(cur)) + 1e-30) < tol:
                steps = it + 1; break
            prev = cur
    uu = np.asarray(s.get_u()).reshape((L, H, nz), order="F")
    xr = reattach(uu[:, 0, nz // 2]) / S
    return dict(blew=False, steps=steps, wall=time.time() - t0, xr=xr)


def main():
    S = int(os.environ.get("VMG_S", 128))           # H=2S=256 channel height; L=12S
    Re = float(os.environ.get("VMG_RE", 200.0))
    ms = int(os.environ.get("VMG_MAXSTEPS", 6000))
    print(f"=== backward-facing step  {12*S}x{2*S}x4  Re_S={Re:.0f} ===")
    dts = float(os.environ.get("VMG_DT_SPEEDUP", 0.4))
    print(f"--- SPEEDUP: time to steady (explicit-advection dt={dts:g}) ---")
    rb = bfs(S, Re, dt=dts, mode="rbgs", vel_iter=60, max_steps=ms)
    print(f"  RB-GS  60 sweeps : steps={rb['steps']:5d}  wall={rb['wall']:7.1f}s  x_r/S={rb['xr']:.2f}")
    rb2 = bfs(S, Re, dt=dts, mode="rbgs", vel_iter=200, max_steps=ms)
    print(f"  RB-GS 200 sweeps : steps={rb2['steps']:5d}  wall={rb2['wall']:7.1f}s  x_r/S={rb2['xr']:.2f}")
    vg = bfs(S, Re, dt=dts, mode="vmg", vcyc=4, max_steps=ms)
    print(f"  vel-MG  4 vcyc   : steps={vg['steps']:5d}  wall={vg['wall']:7.1f}s  x_r/S={vg['xr']:.2f}"
          f"   speedup x{rb2['wall'] / max(vg['wall'], 1e-9):.2f} (vs RB-GS 200)")
    print("--- NO DT RESTRICTION: implicit-FOU + upwind vel-MG, growing dt (CFL=U*dt) ---")
    for dt in [2.0, 8.0, 32.0, 128.0]:
        r = bfs(S, Re, dt=dt, mode="vmg_fou", vcyc=6, max_steps=600)
        tag = "BLEW UP" if r["blew"] else f"finite, x_r/S={r['xr']:.2f} (steps {r['steps']})"
        print(f"  dt={dt:7.1f} (CFL~{dt:.0f}): {tag}")


if __name__ == "__main__":
    main()
