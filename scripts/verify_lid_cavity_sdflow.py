#!/usr/bin/env python3
"""Verification (sdflow): the lid-driven cavity -- the canonical NATIVE-domain-BC benchmark. Three
no-slip walls (-x,+x,-y) + a lid (+y) moving in +x; quasi-2D (periodic z). Driven only by the lid, so
this exercises the whole BC framework: Dirichlet/no-slip velocity ghosts (mac_bc.cuh), a non-periodic
halo on x and y, and Neumann pressure on the walls (boundary-face openness zeroed). We compare the
centreline profiles to the tabulated Ghia, Ghia & Shin (1982) data at Re=100.

Uses the canonical `sdflow` module. NO immersed solid -- the cavity is set up with set_domain_bc +
set_pressure_geometry(all-fluid). Physical units: set_rho/set_mu; Re = U_lid * L / nu, L = N (grid units).
"""
import sys

import numpy as np

from _bootstrap import ensure_flow  # noqa: E402
sdflow = ensure_flow()

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
    """Run the lid-driven cavity to steady state and return the comparison to Ghia et al.

    Sets three no-slip walls and a moving lid (Dirichlet velocity U on +y), marches to steady state, and
    returns (N, Re, u-rms, v-rms, min centreline u, max divergence, steps) where the rms values are the
    centreline-profile errors against the tabulated Ghia, Ghia & Shin (1982) data at the given Reynolds
    number. Quasi-2D (periodic z). Returns None on non-root MPI ranks.
    """
    nu = U * N / Re
    s = sdflow.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(1.0); s.set_advection(True)
    s.set_domain_bc("-x", "wall"); s.set_domain_bc("+x", "wall"); s.set_domain_bc("-y", "wall")  # -x, +x, -y no-slip
    s.set_domain_bc("+y", "inflow", (U, 0.0, 0.0))                                   # +y lid moving in +x
    s.diagnostics.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=8)         # semi-coarsening MG (z frozen, x/y deep; auto-capped)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((N, N, nz), 1e30))                  # all-fluid + Neumann walls

    # Steady state is measured as the LARGEST velocity change anywhere on the sampled plane over the
    # last 50 steps, relative to the lid speed. It used to be the plane-MEAN of u, which is a bad
    # proxy: in a lid-driven cavity that mean is near zero by symmetry, so it is dominated by
    # round-off in the pressure solve rather than by the flow, and a relative test on it measures
    # noise settling rather than the field converging. That went unnoticed while the operator was
    # stored in float, whose higher noise floor happened to keep the test running long enough. With
    # double operator storage (the default since 2026-09-12) the mean settles sooner and the march
    # stopped at 400 steps instead of 650, comparing an UNCONVERGED field to Ghia: centreline
    # min u = -0.1934 against the tabulated -0.2058, rms 0.0247/0.0256 against a 0.02 gate. The
    # max-change criterion below converges both builds to the same answer.
    #
    # The threshold is read off the measured trajectory, not guessed. maxdu/U decays monotonically
    # (1.5e-1 at step 100, 1.4e-2 at 400, 4.3e-3 at 650, 9.6e-4 at 1000) while the Ghia error falls
    # to a plateau: u_rms 0.0247 at step 400, 0.0102 at 650, 0.0069 at 900, 0.0066 at 1050, and the
    # centreline minimum settles near -0.212. 1e-3 stops the march at about step 1000, on the
    # plateau and a factor of three inside the 0.02 gate, instead of on the way down to it.
    prev = None
    steps = max_steps
    for it in range(max_steps):
        s.step()
        if it % 50 == 49:
            u = s.get_u()
            done = False
            if s.rank() == 0:
                cur = np.array(u[:, :, nz // 2], dtype=float)
                if prev is not None:
                    done = it > 300 and float(np.abs(cur - prev).max()) < 1e-3 * U
                prev = cur
            if s.diagnostics.bcast_from_root(done):
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
    """Run the default cavity case, print the Ghia comparison, and exit non-zero if it fails the tolerance."""
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
