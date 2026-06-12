#!/usr/bin/env python3
"""Verification (sdflow): the backward-facing step (BFS) -- the canonical separated-flow benchmark, and
the showcase for the per-face INLET PROFILE. Geometry is the Gartling (1990) expansion-ratio-2 step: a
channel of full height H = 2S (S = step height) with no-slip walls top/bottom; flow enters ONLY the upper
half (y in [S, 2S]) with the developed parabolic channel profile and is zero over the lower half (the step
face); it leaves through the outflow. Behind the step a recirculation bubble forms on the bottom wall; the
reattachment length x_r/S grows with Re. Quasi-2D (periodic z).

The step is realized purely as the inlet condition via set_domain_bc_profile (no immersed solid): the
parabola vanishes at y=S (the step lip) and y=2S (top wall), exactly the developed upper-channel profile.

We de-risk in the laminar regime (Re_S = U_in*S/nu = 100, 200) -- a bubble must form, x_r/S must grow with
Re, mass is conserved and the flow stays divergence-free -- then push toward Gartling Re=800 and report the
reattachment for comparison (Gartling lower-wall reattachment ~6.1 H = ~12 S). Re_S = U_in * S / nu, with
U_in the mean inlet velocity over the open upper half.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import sdflow  # noqa: E402


def inlet_profile(H, S, nz, U_in):
    """Developed parabola over the open upper half y in [S, 2S], zero over the step face y in [0, S]."""
    prof = np.zeros((H, nz, 3))
    yc = np.arange(H) + 0.5
    eta = (yc - S) / S                       # 0 at the step lip, 1 at the top wall
    up = (yc > S)
    prof[up, :, 0] = (6.0 * U_in * eta * (1.0 - eta))[up, None]
    return prof


def reattachment(u_bottom):
    """First sign change (- -> +) of the near-bottom-wall streamwise velocity = end of the primary bubble."""
    for i in range(2, len(u_bottom)):
        if u_bottom[i - 1] < 0.0 <= u_bottom[i]:
            return (i - 1) + u_bottom[i - 1] / (u_bottom[i - 1] - u_bottom[i])
    return 0.0


def run(Re, S=16, Lr=12, U_in=1.0, nz=4, dt=0.4, max_steps=12000):
    H = 2 * S
    L = Lr * S
    nu = U_in * S / Re
    s = sdflow.Solver(L, H, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(True)
    s.set_domain_bc_profile(0, inlet_profile(H, S, nz, U_in))  # -x partial parabolic inlet (-> inflow)
    s.set_domain_bc(1, 3)                                      # +x outflow
    s.set_domain_bc(2, 1); s.set_domain_bc(3, 1)              # -y, +y no-slip walls
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=1)                  # RB-GS pressure (set BEFORE geometry)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((L, H, nz), 1e30))        # all-fluid + BC pressure faces

    t0 = time.time()
    prev = 0.0
    steps = max_steps
    for it in range(max_steps):
        s.step()
        if it % 100 == 99:
            u = s.get_u()
            xr = reattachment(u[:, 0, nz // 2]) if s.rank() == 0 else 0.0
            done = it > 2000 and abs(xr - prev) < 1e-3 * S
            prev = xr
            if s.bcast_from_root(done):
                steps = it + 1
                break
    u = s.get_u(); div = s.max_open_divergence()
    if s.rank() != 0:
        return None
    xr = reattachment(u[:, 0, nz // 2])
    flux_in = float(u[2, :, nz // 2].sum())
    flux_out = float(u[L - 3, :, nz // 2].sum())
    mass_err = abs(flux_out - flux_in) / (abs(flux_in) + 1e-30)
    has_bubble = bool((u[1:S, 0, nz // 2] < 0).any())  # reverse flow just behind the step
    return dict(Re=Re, S=S, H=H, L=L, steps=steps, secs=time.time() - t0, xr_S=xr / S, xr_H=xr / H,
                div=div, mass_err=mass_err, bubble=has_bubble)


def main():
    print("=== sdflow: backward-facing step (Gartling expansion ratio 2) ===")
    # laminar de-risk: a bubble must form, x_r/S must grow with Re, flow stays mass-conserving & div-free
    laminar = [run(100), run(200)]
    ok = True
    xr_prev = -1.0
    for r in laminar:
        if r is None:
            return
        print(f"  Re_S={r['Re']:<4d} S={r['S']} L={r['L']}  x_r/S={r['xr_S']:.2f}  bubble={r['bubble']}  "
              f"mass_err={r['mass_err']:.1e}  div={r['div']:.1e}  ({r['steps']} steps, {r['secs']:.0f}s)")
        ok &= (r["bubble"] and r["mass_err"] < 1e-3 and r["div"] < 1e-6 and 0.5 < r["xr_S"] < 12.0
               and r["xr_S"] > xr_prev)
        xr_prev = r["xr_S"]
    # push toward the Gartling Re=800 benchmark (long; opt in with SDFLOW_BFS_RE800=1). Informational
    # comparison: Gartling's lower-wall reattachment ~6.1 H (~12 S), S=32, L=30S.
    if os.environ.get("SDFLOW_BFS_RE800") == "1":
        r = run(800, S=32, Lr=30, dt=0.3, max_steps=24000)
        if r is not None:
            print(f"  Re_S={r['Re']:<4d} S={r['S']} L={r['L']}  x_r/S={r['xr_S']:.2f}  x_r/H={r['xr_H']:.2f}"
                  f"  (Gartling ~6.1 H)  mass_err={r['mass_err']:.1e}  div={r['div']:.1e}  "
                  f"({r['steps']} steps, {r['secs']:.0f}s)")
            ok &= (r["bubble"] and r["mass_err"] < 1e-3 and r["div"] < 1e-6 and r["xr_S"] > xr_prev)
    print(f"  result: {'PASS' if ok else 'FAIL'}  (recirculation grows with Re, mass-conserving, div-free)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
