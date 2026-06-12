#!/usr/bin/env python3
"""Verification (sdflow): the lid-driven cavity -- the canonical NATIVE-domain-BC benchmark. Three
no-slip walls (-x,+x,-y) + a lid (+y) moving in +x; quasi-2D (periodic z). Driven only by the lid, so
this exercises the whole BC framework: Dirichlet/no-slip velocity ghosts (mac_bc.cuh), a non-periodic
halo on x and y, and Neumann pressure on the walls (boundary-face openness zeroed). We compare the
centreline profiles to the tabulated Ghia, Ghia & Shin (1982) data at Re=100.

Uses the canonical `sdflow` module. NO immersed solid -- the cavity is set up with set_domain_bc +
set_pressure_geometry(all-fluid). Physical units: set_rho/set_mu; Re = U_lid * L / nu, L = N (grid units).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import sdflow  # noqa: E402

# Ghia, Ghia & Shin (1982), Re=100 -- u along the vertical centreline, v along the horizontal centreline.
GHIA_Y = np.array([0, .0547, .0625, .0703, .1016, .1719, .2813, .4531, .5, .6172, .7344, .8516, .9531,
                   .9609, .9688, .9766, 1])
GHIA_U = np.array([0, -.03717, -.04192, -.04775, -.06434, -.10150, -.15662, -.21090, -.20581, -.13641,
                   .00332, .23151, .68717, .73722, .78871, .84123, 1])
GHIA_X = np.array([0, .0625, .0703, .0781, .0938, .1563, .2266, .2344, .5, .8047, .8594, .9063, .9453,
                   .9531, .9609, .9688, 1])
GHIA_V = np.array([0, .09233, .10091, .10890, .12317, .16077, .17507, .17527, .05454, -.24533, -.22445,
                   -.16914, -.10313, -.08864, -.07391, -.05906, 0])


def run(N=128, Re=100.0, U=1.0, nz=4, max_steps=5000):
    nu = U * N / Re
    s = sdflow.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(1.0); s.set_advection(True)
    s.set_domain_bc(0, 1); s.set_domain_bc(1, 1); s.set_domain_bc(2, 1)  # -x, +x, -y no-slip
    s.set_domain_bc(3, 2, U, 0.0, 0.0)                                   # +y lid moving in +x
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=4)         # multilevel MG (auto-clamped to 2 for nz=4; pre-geom)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((N, N, nz), 1e30))                  # all-fluid + Neumann walls

    prev = 0.0
    steps = max_steps
    for it in range(max_steps):
        s.step()
        if it % 50 == 49:
            u = s.get_u()
            m = float(u[:, :, nz // 2].mean()) if s.rank() == 0 else 0.0
            done = it > 300 and abs(m - prev) < 1e-5 * (abs(m) + 1e-30)
            prev = m
            if s.bcast_from_root(done):
                steps = it + 1
                break
    u = s.get_u(); v = s.get_v(); div = s.max_open_divergence()
    if s.rank() != 0:
        return None
    yc = (np.arange(N) + 0.5) / N
    uc = u[N // 2, :, nz // 2] / U          # vertical-centreline u(y)
    vc = v[:, N // 2, nz // 2] / U          # horizontal-centreline v(x)
    u_rms = float(np.sqrt(np.mean((np.interp(GHIA_Y, yc, uc) - GHIA_U) ** 2)))
    v_rms = float(np.sqrt(np.mean((np.interp(GHIA_X, yc, vc) - GHIA_V) ** 2)))
    return N, Re, u_rms, v_rms, float(uc.min()), div, steps


def main():
    r = run()
    if r is None:
        return
    N, Re, u_rms, v_rms, umin, div, steps = r
    print("=== sdflow: lid-driven cavity vs Ghia, Ghia & Shin (1982) ===")
    print(f"  Re={Re:.0f}  N={N}  ({steps} steps)")
    print(f"  u(y) rms vs Ghia = {u_rms:.4f}   v(x) rms vs Ghia = {v_rms:.4f}")
    print(f"  min centreline u = {umin:.4f} (Ghia -0.2058)   max flux divergence = {div:.1e}")
    ok = u_rms < 0.02 and v_rms < 0.02 and div < 1e-6
    print(f"  result: {'PASS' if ok else 'FAIL'}  (centreline profiles match Ghia, incompressible)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
