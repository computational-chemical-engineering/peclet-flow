#!/usr/bin/env python3
"""Phase-5a verification (collocated grid): the lid-driven cavity vs Ghia, Ghia & Shin (1982), Re=100 --
the canonical native-domain-BC benchmark. Three no-slip walls (-x,+x,-y) + a lid (+y) moving in +x,
quasi-2D (periodic z), no immersed solid. This exercises the collocated domain-BC machinery: cell-centered
reflection velocity ghosts (every component reflects about the boundary face), the explicit-reflection
diffusion smoother, and the Neumann phi wall ghost in the approximate projection.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow as sdflow  # noqa: E402

GHIA_Y = np.array([0, .0547, .0625, .0703, .1016, .1719, .2813, .4531, .5, .6172, .7344, .8516, .9531,
                   .9609, .9688, .9766, 1])
GHIA_U = np.array([0, -.03717, -.04192, -.04775, -.06434, -.10150, -.15662, -.21090, -.20581, -.13641,
                   .00332, .23151, .68717, .73722, .78871, .84123, 1])
GHIA_X = np.array([0, .0625, .0703, .0781, .0938, .1563, .2266, .2344, .5, .8047, .8594, .9063, .9453,
                   .9531, .9609, .9688, 1])
GHIA_V = np.array([0, .09233, .10091, .10890, .12317, .16077, .17507, .17527, .05454, -.24533, -.22445,
                   -.16914, -.10313, -.08864, -.07391, -.05906, 0])


def run(N=128, Re=100.0, U=1.0, nz=4, max_steps=9000):
    nu = U * N / Re
    s = sdflow.SolverColocated(N, N, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(1.0); s.set_advection(True)
    s.set_domain_bc(0, 1); s.set_domain_bc(1, 1); s.set_domain_bc(2, 1)   # -x, +x, -y no-slip
    s.set_domain_bc(3, 2, U, 0.0, 0.0)                                    # +y lid moving in +x
    s.set_velocity_solver_params(60)
    s.set_pressure_pcg(True, 400, 1e-9)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, nz), 1e30)))  # all-fluid + Neumann walls

    prevp = None
    steps = max_steps
    for it in range(max_steps):
        s.step()
        if it % 100 == 99:
            uc = s.get_u()[N // 2, :, nz // 2]
            if prevp is not None and np.max(np.abs(uc - prevp)) < 1e-5:
                steps = it + 1
                break
            prevp = uc.copy()

    u = s.get_u(); v = s.get_v(); div = float(s.max_open_divergence())
    yc = (np.arange(N) + 0.5) / N
    uc = u[N // 2, :, nz // 2] / U
    vc = v[:, N // 2, nz // 2] / U
    u_rms = float(np.sqrt(np.mean((np.interp(GHIA_Y, yc, uc) - GHIA_U) ** 2)))
    v_rms = float(np.sqrt(np.mean((np.interp(GHIA_X, yc, vc) - GHIA_V) ** 2)))
    return N, Re, u_rms, v_rms, float(uc.min()), div, steps


def main():
    N, Re, u_rms, v_rms, umin, div, steps = run()
    print("=== sdflow phase-5a: collocated lid-driven cavity vs Ghia, Ghia & Shin (1982) ===")
    print(f"  Re={Re:.0f}  N={N}  ({steps} steps)")
    print(f"  u(y) rms vs Ghia = {u_rms:.4f}   v(x) rms vs Ghia = {v_rms:.4f}")
    print(f"  min centreline u = {umin:.4f} (Ghia -0.2058)   max flux divergence = {div:.1e}")
    ok = u_rms < 0.02 and v_rms < 0.02 and div < 1e-6
    print(f"  result: {'PASS' if ok else 'FAIL'}  (centreline profiles match Ghia, incompressible)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
