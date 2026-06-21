#!/usr/bin/env python3
"""Phase-3 verification (collocated grid): 2-D Taylor-Green vortex in a triply-periodic box, the canonical
test of the approximate (MAC) projection. The exact incompressible Navier-Stokes solution is

    u =  U0 sin(kx) cos(ky) e^{-2 nu k^2 t},   v = -U0 cos(kx) sin(ky) e^{-2 nu k^2 t},   w = 0,

with the nonlinear term exactly balanced by the pressure gradient. Advection is ON, so the projection has
to remove the divergence the discrete advection injects each step. We check (a) the projected face field is
divergence-free (max_open_divergence -> solver tol) and (b) the velocity matches the exact decayed field,
the L2 error shrinking with resolution. Grid spacing = 1; the vortex wavelength is the box length N (k=2pi/N).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build"))))
import sdflow  # noqa: E402


def tg_fields(N, nz, U0, amp):
    k = 2.0 * np.pi / N
    ix = np.arange(N)
    X, Y = np.meshgrid(ix, ix, indexing="ij")
    u2 = U0 * amp * np.sin(k * X) * np.cos(k * Y)
    v2 = -U0 * amp * np.cos(k * X) * np.sin(k * Y)
    u = np.repeat(u2[:, :, None], nz, axis=2)
    v = np.repeat(v2[:, :, None], nz, axis=2)
    w = np.zeros((N, N, nz))
    return (np.asfortranarray(u), np.asfortranarray(v), np.asfortranarray(w))


def run(SolverCls, N, nz=4, U0=1.0, rho=1.0, nu=0.05, dt=0.5, steps=100):
    k = 2.0 * np.pi / N
    s = SolverCls(N, N, nz)
    s.set_rho(rho)
    s.set_mu(rho * nu)
    s.set_dt(dt)
    s.set_advection(True)                      # exercise the projection: advection injects divergence
    s.set_velocity_solver_params(80)
    sdf = np.asfortranarray(np.ones((N, N, nz)) * 1e3)   # all-fluid SDF -> trivial cut-cell, full projection
    s.set_solid(sdf, cutcell_pressure=True)
    u0, v0, w0 = tg_fields(N, nz, U0, 1.0)
    s.set_state(u0, v0, w0)

    for _ in range(steps):
        s.step()

    T = dt * steps
    amp = np.exp(-2.0 * nu * k * k * T)         # exact (continuous) decay factor
    ue, ve, we = tg_fields(N, nz, U0, amp)
    uu, vv = s.get_u(), s.get_v()
    num = np.sqrt(np.mean((uu - ue) ** 2 + (vv - ve) ** 2))
    den = np.sqrt(np.mean(u0 ** 2 + v0 ** 2))   # normalize by the initial field norm
    l2 = num / den
    # measured vs analytic energy decay (diagnostic)
    e_now = float(np.mean(uu ** 2 + vv ** 2))
    e_ini = float(np.mean(u0 ** 2 + v0 ** 2))
    ratio_meas = e_now / e_ini
    ratio_ana = amp * amp
    div = float(s.max_open_divergence())
    return l2, div, ratio_meas, ratio_ana


def main():
    print("=== sdflow phase-3: collocated Taylor-Green vortex (approximate/MAC projection, advection ON) ===")
    print(f"{'N':>5} {'L2_err':>11} {'maxdiv':>11} {'E/E0(meas)':>11} {'E/E0(ana)':>11}")
    errs = []
    divs = []
    for N in (32, 64):
        l2, div, rm, ra = run(sdflow.SolverColocated, N)
        print(f"{N:5d} {l2:11.3e} {div:11.3e} {rm:11.5f} {ra:11.5f}")
        errs.append(l2)
        divs.append(div)
    shrinks = errs[1] < errs[0]
    divfree = max(divs) < 1e-6
    ok = shrinks and divfree and errs[1] < 0.02
    print(f"  L2 error shrinks 32->64: {shrinks} ({errs[0]:.2e} -> {errs[1]:.2e}, ratio {errs[0]/errs[1]:.1f}x)")
    print(f"  faces divergence-free (<1e-6): {divfree} (max {max(divs):.1e})")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
