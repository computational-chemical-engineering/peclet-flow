#!/usr/bin/env python3
"""Phase-5b verification (collocated grid): the backward-facing step (BFS) -- the separated-flow benchmark
and the showcase for the collocated per-face INLET PROFILE. Gartling (1990) expansion-ratio-2 step: a
channel of full height H=2S with no-slip walls top/bottom; flow enters ONLY the upper half (y in [S,2S])
with the developed parabola and is zero over the lower half (the step face); it leaves through the outflow.
The step is realized purely as the inlet condition (set_domain_bc_profile -> the collocated profile inlet
ghost, bcVelocityColocated with a per-position value), no immersed solid.

Laminar de-risk (Re_S = U_in*S/nu = 100, 200): a recirculation bubble must form behind the step, the
reattachment length x_r/S must grow with Re, mass is conserved, and the flow stays incompressible (the open
outflow leaves the approximate projection's O(h^2) residual, looser than the staggered 1e-6). The staggered
solver reaches x_r/S 5.3 (Re=100) -> 8.3 (Re=200) on the Armaly/Biswas curve.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow as sdflow  # noqa: E402


def inlet_profile(H, S, nz, U_in):
    prof = np.zeros((H, nz, 3))
    yc = np.arange(H) + 0.5
    eta = (yc - S) / S
    up = yc > S
    prof[up, :, 0] = (6.0 * U_in * eta * (1.0 - eta))[up, None]
    return prof


def reattachment(u_bottom):
    reversed_yet = False
    for i in range(1, len(u_bottom)):
        if u_bottom[i] < 0.0:
            reversed_yet = True
        elif reversed_yet and u_bottom[i] >= 0.0:
            return (i - 1) + u_bottom[i - 1] / (u_bottom[i - 1] - u_bottom[i])
    return 0.0


def run(Re, S=16, Lr=12, U_in=1.0, nz=4, dt=0.4, max_steps=8000):
    H, L = 2 * S, Lr * S
    nu = U_in * S / Re
    s = sdflow.SolverColocated(L, H, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(True)
    s.set_domain_bc_profile(0, inlet_profile(H, S, nz, U_in))  # -x partial parabolic inlet (-> inflow)
    s.set_domain_bc(1, 3)                                      # +x outflow
    s.set_domain_bc(2, 1); s.set_domain_bc(3, 1)              # -y, +y no-slip walls
    s.set_velocity_solver_params(60)
    s.set_pressure_pcg(True, 400, 1e-9)
    s.set_pressure_geometry(np.asfortranarray(np.full((L, H, nz), 1e30)))

    t0 = time.time()
    prev = 0.0
    steps = max_steps
    for it in range(max_steps):
        s.step()
        if it % 100 == 99:
            xr = reattachment(s.get_u()[:, 0, nz // 2])
            if it > 3000 and xr > S and abs(xr - prev) < 1e-3 * S:
                steps = it + 1
                break
            prev = xr
    u = s.get_u()
    xr = reattachment(u[:, 0, nz // 2])
    flux_in = float(u[2, :, nz // 2].sum())
    flux_out = float(u[L - 3, :, nz // 2].sum())
    mass_err = abs(flux_out - flux_in) / (abs(flux_in) + 1e-30)
    bubble = bool((u[1:S, 0, nz // 2] < 0).any())
    return dict(Re=Re, S=S, L=L, steps=steps, secs=time.time() - t0, xr_S=xr / S,
                div=float(s.max_open_divergence()), mass_err=mass_err, bubble=bubble)


def main():
    print("=== sdflow phase-5b: collocated backward-facing step (Gartling expansion ratio 2) ===")
    laminar = [run(100, max_steps=5000), run(200, max_steps=8000)]
    ok = True
    xr_prev = -1.0
    for r in laminar:
        print(f"  Re_S={r['Re']:<4d} S={r['S']} L={r['L']}  x_r/S={r['xr_S']:.2f}  bubble={r['bubble']}  "
              f"mass_err={r['mass_err']:.1e}  div={r['div']:.1e}  ({r['steps']} steps, {r['secs']:.0f}s)")
        ok &= (r["bubble"] and r["mass_err"] < 1e-3 and r["div"] < 2e-3 and 0.5 < r["xr_S"] < 12.0
               and r["xr_S"] > xr_prev)
        xr_prev = r["xr_S"]
    print(f"  (staggered reference: x_r/S 5.3 -> 8.3 on the Armaly/Biswas curve)")
    print(f"  result: {'PASS' if ok else 'FAIL'}  (recirculation grows with Re, mass-conserving)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
