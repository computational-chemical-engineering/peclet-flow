#!/usr/bin/env python3
"""Case A G5 diagnosis, step 2: the a-priori translation test (doc/full_viscous_stress.md §2.4).

A two-phase field translating rigidly, u = U everywhere, with the case-A density and viscosity
jumps (ratios 0.1), g = 0, triply periodic, is an exact solution of the two-phase Navier-Stokes
equations without surface tension: every force vanishes and the interface is carried by U.
A momentum form passes if it keeps u = U to rounding. With sigma != 0 (the case-A sigma, i.e. Bo 2
at the case-A g) the exact solution is still u = U (a translating sphere at rest in its own frame,
Laplace pressure inside), and |u - U| measures the spurious currents.

Forms: 'vel' = the velocity form (the case-A default), 'vel-adv' = the velocity form with
set_advection(True), 'mom' = enable_vof_momentum (conservative rho u transport, single VoF).
The structured VoF is used throughout (enable_vof_momentum refuses the block container) except
'vel-blk' / 'vel-adv-blk' = the case-A block container + block CSF.
set_superficial_velocity(True, 'x', U) as in case A; a = shift/dt reported in units of g = 9.81.

Usage:  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \\
            python tests/study/case_a/probe_translate.py --n 32 --steps 200 [--sigma]
Prints one line per (form, stress).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_case_a as ca

U0 = (0.07, 0.035, 0.0175)  # m/s; the case-A rise velocity is ~0.07 m/s


def build(n, form, stress, sigma):
    L = ca.L_OVER_D * ca.D
    s = ca.pf.Solver((n, n, n), extent=(L, L, L))
    s.set_rho(ca.RHO_L)
    s.set_mu(ca.MU_L)
    s.set_pressure_geometry(np.full((n,) * 3, 1.0, order="F"))
    s.enable_vof()
    s.set_vof(np.zeros((n,) * 3, order="F"))
    s.set_property_model("rho", "linear", "C", [ca.RHO_L, ca.RHO_G - ca.RHO_L])
    s.set_property_model("mu", "linear", "C", [ca.MU_L, ca.MU_G - ca.MU_L])
    s.set_viscous_stress(stress)
    if "adv" in form:
        s.set_advection(True)
    if sigma:
        s.set_surface_tension(ca.SIGMA)
    s.set_pressure_pcg(True, 800, 1e-8)
    h = L / n
    if form.endswith("blk"):
        s.enable_vof_blocks([(0.5 * L / h,) * 3 + (0.5 * ca.D / h,)])
        if sigma:
            s.enable_vof_block_csf()
    else:
        s.set_vof(ca.sphere_fractions(n, 0.5 * L / h, 0.5 * ca.D / h))
        if form == "mom":
            s.enable_vof_momentum(ca.RHO_G, ca.RHO_L)
    # pin the time reference (the first set_dt) BEFORE set_velocity and the first
    # vof_step_limits: before it, velocities go in and come out unscaled (case A starts at rest
    # and never met this)
    s.set_dt(0.05 * h / U0[0])
    for c in range(3):
        s.set_velocity(c, np.full((n,) * 3, U0[c], order="F"))
    s.set_superficial_velocity(True, "x", U0[0])
    s.update_properties()
    return s


def run(n, form, stress, sigma, steps):
    s = build(n, form, stress, sigma)
    emax, acc = 0.0, []
    Cin = np.array(s.get_vof())
    C0 = Cin.sum()
    for k in range(steps):
        L_ = s.vof_step_limits()
        dt = 0.25 * min(L_["cfl_dt"], L_["capillary_dt"])
        s.set_dt(dt)
        s.step()
        e = max(
            float(np.max(np.abs(np.array(g()) - U0[c])))
            for c, g in enumerate((s.get_u, s.get_v, s.get_w))
        )
        emax = max(emax, e)
        acc.append(s.diagnostics.last_superficial_velocity_shift() / dt / ca.G)
    Cend = np.array(s.get_vof())
    vol = Cend.sum() / C0 - 1.0
    moved = float(np.abs(Cend - Cin).max())  # O(1) when the interface moved
    acc = np.array(acc)
    return (
        f"TRANSLATE n={n} sigma={'case' if sigma else 0} form={form} stress={stress} "
        f"steps={steps} max|u-U|/U={emax / U0[0]:.3e} final|u-U|/U={e / U0[0]:.3e} "
        f"a/g(mean last half)={acc[len(acc) // 2 :].mean():+.3e} a/g(max abs)={np.abs(acc).max():.3e} "
        f"vol_err={vol:.1e} max|C-C0|={moved:.2f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--sigma", action="store_true")
    p.add_argument("--forms", default="vel,vel-adv,mom,vel-blk")
    a = p.parse_args()
    for form in a.forms.split(","):
        for stress in ("laplacian", "full"):
            print(run(a.n, form, stress, a.sigma, a.steps), flush=True)


if __name__ == "__main__":
    main()
