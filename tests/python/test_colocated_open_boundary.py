#!/usr/bin/env python
"""Gate: the COLLOCATED solver stays stable and mass-conserving at an OPEN boundary.

Why this test exists. Nothing in the registered suite exercised the collocated open boundary:
every collocated ctest is either Stokes (no advecting velocity at all) or periodic/walled. So when
the collocated advecting velocity became the projected face field (`doc/uf_advection.md`), the
first version fed the last cell's upwind flux the one face `bcCorrectOutflow` sets from GLOBAL
continuity rather than from the fluid beside it -- a positive feedback loop that took the
developing channel to max|u| = 4.5e+02 by step 50 and NaN by step 69 -- and all 161 ctests passed.
`scripts/verify_colocated_channel.py` and `scripts/verify_colocated_bfs.py` caught it, but they
are ~20-minute full-fidelity studies. This is the cheap version of the same signal: one coarse
channel, one resolution, long enough to develop.

Checks, in the order they would fail:
  1. the field is finite (the failure above was a blow-up, not a bias);
  2. mass in == mass out (the outflow face is the mass-balance closure -- if advection corrupts it,
     this is what goes);
  3. the developed profile is Poiseuille, u_max/U_mean -> 1.5;
  4. the projected face field is still discretely divergence-free.

Run:  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \
          python tests/python/test_colocated_open_boundary.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

H, L, NZ = 16, 112, 4
U, RE, DT, STEPS = 1.0, 100.0, 0.5, 400


def build():
    import peclet.flow as pf

    s = pf.SolverColocated(L, H, NZ)
    s.set_rho(1.0)
    s.set_mu(U * H / RE)
    s.set_dt(DT)
    s.set_advection(True)
    s.set_domain_bc("-x", "inflow", (U, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow")
    s.set_domain_bc("-y", "wall")
    s.set_domain_bc("+y", "wall")
    s.diagnostics.set_velocity_solver_params(60)
    s.set_pressure_pcg(True, 400, 1e-9)
    s.set_pressure_geometry(np.asfortranarray(np.full((L, H, NZ), 1e30)))
    return s


def test_colocated_open_boundary():
    """Pytest entry point (tests/python is collected by ctest as pytest too)."""
    s = build()
    for _ in range(STEPS):
        s.step()
    u = np.asarray(s.get_u())
    assert np.isfinite(u).all(), "collocated open boundary went non-finite"

    k = NZ // 2
    flux_in = float(u[2, :, k].sum())
    flux_out = float(u[L - 3, :, k].sum())
    mass_err = abs(flux_out - flux_in) / abs(flux_in)
    assert mass_err < 1e-3, f"mass not conserved: {mass_err:.2e}"

    prof = u[L - 4, :, k]
    ratio = float(prof.max() / prof.mean())
    assert 1.45 < ratio < 1.55, f"profile not developed Poiseuille: u_max/U_mean = {ratio:.4f}"

    div = float(s.max_open_divergence())
    assert div < 1e-3, f"face field not divergence-free: {div:.2e}"
    return mass_err, ratio, div


def main():
    try:
        import peclet.flow  # noqa: F401
    except ImportError:
        print("SKIP: peclet.flow not importable (set PYTHONPATH to the build tree)")
        return 77
    try:
        mass_err, ratio, div = test_colocated_open_boundary()
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    print(f"OK: collocated channel after {STEPS} steps -- mass_err {mass_err:.2e}, "
          f"u_max/U_mean {ratio:.4f}, max_open_divergence {div:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
