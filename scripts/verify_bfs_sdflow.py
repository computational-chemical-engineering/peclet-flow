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
    """End of the primary bubble = the first - -> + crossing of the near-bottom-wall streamwise velocity
    AFTER the flow has actually reversed behind the step (skip any thin positive sliver at the step lip)."""
    reversed_yet = False
    for i in range(1, len(u_bottom)):
        if u_bottom[i] < 0.0:
            reversed_yet = True
        elif reversed_yet and u_bottom[i] >= 0.0:
            return (i - 1) + u_bottom[i - 1] / (u_bottom[i - 1] - u_bottom[i])
    return 0.0


def run(Re, S=16, Lr=12, U_in=1.0, nz=4, dt=0.4, max_steps=12000):
    """Run the backward-facing step at Reynolds number `Re` (Re_S = U_in*S/nu) and return its diagnostics.

    Step height S, full channel height 2S, channel length Lr*S; a partial parabolic inlet feeds the upper
    half, the lower half is the step face. Marches to steady state and returns a dict with the lower-wall
    reattachment length (x_r/S, x_r/H), bubble presence, mass-conservation error and max divergence.
    Returns None off root.
    """
    H = 2 * S
    L = Lr * S
    nu = U_in * S / Re
    s = sdflow.Solver(L, H, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(True)
    s.set_domain_bc_profile(0, inlet_profile(H, S, nz, U_in))  # -x partial parabolic inlet (-> inflow)
    s.set_domain_bc(1, 3)                                      # +x outflow
    s.set_domain_bc(2, 1); s.set_domain_bc(3, 1)              # -y, +y no-slip walls
    s.set_velocity_solver_params(60)
    if os.environ.get("SDFLOW_BFS_VMG") == "1":               # opt-in: velocity-MG (diffusion-only) on the BFS
        s.set_velocity_multigrid(True, 8, 4)                  #   -- exercises the outflow -beta fold + inflow
    s.set_pressure_multigrid(True, levels=8)                  # semi-coarsening MG (z frozen, x/y deep; capped)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((L, H, nz), 1e30))        # all-fluid + BC pressure faces

    verbose = os.environ.get("SDFLOW_BFS_VERBOSE") == "1"
    t0 = time.time()
    prev = 0.0
    steps = max_steps
    for it in range(max_steps):
        s.step()
        if it % 100 == 99:
            u = s.get_u()
            xr = reattachment(u[:, 0, nz // 2]) if s.rank() == 0 else 0.0
            # converge only once a real bubble has formed (x_r > S) AND its length has stopped drifting --
            # early in the transient x_r sits near 0 ("stable" but undeveloped), which must NOT trigger.
            done = it > 3000 and xr > S and abs(xr - prev) < 1e-3 * S
            if verbose and s.rank() == 0:
                print(f"    [Re={Re} it={it+1} x_r/S={xr/S:.2f} t={time.time()-t0:.0f}s]", flush=True)
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
    """Run the laminar de-risk (Re_S 100, 200) and optional Re=800 push; print results and set exit code."""
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
    # push toward Re=800 (long: ~30 min, opt in with SDFLOW_BFS_RE800=1). NB our Re_S = U_in*S/nu, whereas
    # Gartling's Re=800 uses the full height H=2S (~ our Re_S=400, where x_r/H ~ 6 would match his ~6.1).
    # At Re_S=800 the bubble is large and slow: measured x_r/H ~ 7.3, still creeping at 36k steps -- a
    # textbook slow open-flow transient near the steady/unsteady boundary. The laminar points above are the
    # quantitative validation; this is a stress/illustration run (PASS gate = monotone growth + invariants).
    if os.environ.get("SDFLOW_BFS_RE800") == "1":
        r = run(800, S=32, Lr=22, dt=0.3, max_steps=36000)
        if r is not None:
            print(f"  Re_S={r['Re']:<4d} S={r['S']} L={r['L']}  x_r/S={r['xr_S']:.2f}  x_r/H={r['xr_H']:.2f}"
                  f"  (Gartling ~6.1 H)  mass_err={r['mass_err']:.1e}  div={r['div']:.1e}  "
                  f"({r['steps']} steps, {r['secs']:.0f}s)")
            ok &= (r["bubble"] and r["mass_err"] < 1e-3 and r["div"] < 1e-6 and r["xr_S"] > xr_prev)
    print(f"  result: {'PASS' if ok else 'FAIL'}  (recirculation grows with Re, mass-conserving, div-free)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
