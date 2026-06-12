#!/usr/bin/env python3
"""Verification (sdflow): developing plane channel flow -- the canonical INFLOW/OUTFLOW benchmark. A
uniform stream U enters at -x (inflow, Dirichlet velocity), leaves at +x (outflow: zero-gradient velocity
+ Dirichlet p=0), between no-slip walls at +-y; quasi-2D (periodic z). This exercises the open-boundary
machinery the cavity cannot: the operator/flux openness split (inflow Neumann pressure but its flux is
counted; outflow Dirichlet pressure), the zero-gradient outflow velocity, and the outflow projection
correction that lets mass leave.

A uniform inlet develops into the parabolic Poiseuille profile over the entrance length
L_e ~ 0.04*Re*H. With L >~ 6H the outlet is fully developed, so we check:
  * global mass conservation: flux(inlet) == flux(outlet)             (continuity, exact)
  * incompressibility:        max cut-cell flux divergence -> 0
  * developed profile:        u(y) at the outlet is parabolic, u_max/U_mean -> 1.5
  * (informational) developed pressure gradient ~ -12*mu*U_mean/H^2.

Uses the canonical `sdflow` module. NO immersed solid. Physical units: set_rho/set_mu; Re = U*H/nu.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import sdflow  # noqa: E402


def run(H=32, L=224, Re=100.0, U=1.0, nz=4, max_steps=8000, dt=0.5):
    nu = U * H / Re
    s = sdflow.Solver(L, H, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(True)
    s.set_domain_bc(0, 2, U, 0.0, 0.0)   # -x inflow: uniform stream
    s.set_domain_bc(1, 3)                # +x outflow
    s.set_domain_bc(2, 1); s.set_domain_bc(3, 1)  # -y, +y no-slip walls
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=4)       # multilevel MG (auto-clamped to 2 for nz=4; pre-geom)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((L, H, nz), 1e30))  # all-fluid + BC pressure faces

    prev = 0.0
    steps = max_steps
    for it in range(max_steps):
        s.step()
        if it % 50 == 49:
            u = s.get_u()
            m = float(u[L - 4, H // 2, nz // 2]) if s.rank() == 0 else 0.0  # outlet centreline u
            done = it > 1000 and abs(m - prev) < 1e-5 * (abs(m) + 1e-30)
            prev = m
            if s.bcast_from_root(done):
                steps = it + 1
                break
    u = s.get_u(); p = s.get_p(); div = s.max_open_divergence()
    if s.rank() != 0:
        return None
    # mass conservation: streamwise flux at an inlet station vs an outlet station (continuity)
    flux_in = float(u[2, :, nz // 2].sum())
    flux_out = float(u[L - 3, :, nz // 2].sum())
    mass_err = abs(flux_out - flux_in) / (abs(flux_in) + 1e-30)
    # developed outlet profile vs the plane-Poiseuille parabola u = 6*U_mean*eta*(1-eta), eta=(j+.5)/H
    prof = u[L - 4, :, nz // 2]
    U_mean = float(prof.mean())
    eta = (np.arange(H) + 0.5) / H
    parab = 6.0 * U_mean * eta * (1.0 - eta)
    prof_rms = float(np.sqrt(np.mean((prof - parab) ** 2)) / (abs(U_mean) + 1e-30))
    ratio = float(prof.max() / (U_mean + 1e-30))
    # informational: developed-region streamwise pressure gradient vs -12*mu*U_mean/H^2
    pc = p[:, H // 2, nz // 2]
    x0, x1 = L // 2, L - 6
    dpdx = float((pc[x1] - pc[x0]) / (x1 - x0))
    dpdx_analytic = -12.0 * nu * U_mean / H**2
    return dict(steps=steps, mass_err=mass_err, div=div, U_mean=U_mean, ratio=ratio,
                prof_rms=prof_rms, dpdx=dpdx, dpdx_analytic=dpdx_analytic, Re=Re, H=H, L=L)


def main():
    r = run()
    if r is None:
        return
    print("=== sdflow: developing plane channel (inflow/outflow) ===")
    print(f"  Re={r['Re']:.0f}  H={r['H']}  L={r['L']}  ({r['steps']} steps)")
    print(f"  mass conservation: |flux_out - flux_in|/flux_in = {r['mass_err']:.2e}")
    print(f"  max flux divergence = {r['div']:.1e}")
    print(f"  outlet u_max/U_mean = {r['ratio']:.4f} (Poiseuille 1.5)   profile rms = {r['prof_rms']:.4f}")
    print(f"  developed dp/dx = {r['dpdx']:.2e}  (analytic -12 mu U/H^2 = {r['dpdx_analytic']:.2e})")
    ok = (r["mass_err"] < 1e-3 and r["div"] < 1e-6 and 1.45 < r["ratio"] < 1.55 and r["prof_rms"] < 0.03)
    print(f"  result: {'PASS' if ok else 'FAIL'}  (mass-conserving, divergence-free, developed parabola)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
