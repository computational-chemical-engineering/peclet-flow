#!/usr/bin/env python3
"""Case A G5 diagnosis: does ONE velocity-form step conserve momentum when every force sums to zero?

With g = 0, sigma = 0 and the superficial closure off, every force of the staggered velocity-form
predictor is a periodic divergence (pressure, viscous flux), so the predictor + projection must
leave  PRED = sum_f rho^n_f (u^{n+1} - u^n) / dt  = 0  to the solver tolerances (the density change
of the step is a separate, physical term and is not in PRED). Initial field: a divergence-free
shear u_x = A sin(2 pi y / L) + a uniform U, through the case-A sphere (rho and mu ratios
selectable). PRED is reported in units of <rho> g (g = 9.81, as in probe_budget.py).

Usage: OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \\
           python tests/study/case_a/probe_conserve.py --n 32
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_case_a as ca


def one(n, rho_ratio, mu_ratio, stress, steps, adv, dtf, grav=False, sigma="none"):
    L = ca.L_OVER_D * ca.D
    h = L / n
    rl, rg, ml, mg = ca.RHO_L, rho_ratio * ca.RHO_L, ca.MU_L, mu_ratio * ca.MU_L
    s = ca.pf.Solver((n, n, n), extent=(L, L, L))
    s.set_rho(rl)
    s.set_mu(ml)
    s.set_pressure_geometry(np.full((n,) * 3, 1.0, order="F"))
    s.enable_vof()
    C = ca.sphere_fractions(n, 0.5 * L / h, 0.5 * ca.D / h)
    s.set_vof(C)
    s.set_property_model("rho", "linear", "C", [rl, rg - rl])
    s.set_property_model("mu", "linear", "C", [ml, mg - ml])
    s.set_viscous_stress(stress)
    if adv:
        s.set_advection(True)
    if sigma != "none":
        s.set_surface_tension(ca.SIGMA)
    s.set_pressure_pcg(True, 800, 1e-8)
    if sigma == "blk":
        s.set_vof(np.zeros((n,) * 3, order="F"))
        s.enable_vof_blocks([(0.5 * L / h,) * 3 + (0.5 * ca.D / h,)])
        s.enable_vof_block_csf()
        C = np.array(s.get_vof())
    if grav:
        al = C.mean()
        rav_ = rl + (rg - rl) * al
        s.set_property_model(
            "force_x", "linear", "C", [-(rl - rav_) * ca.G, -(rg - rl) * ca.G]
        )
    U, A = 0.02, 0.07
    dt = dtf * 0.25 * h / (U + A)
    s.set_dt(dt)
    y = (np.arange(n) + 0.5) * h
    ux = np.asfortranarray(
        np.broadcast_to(
            (U + A * np.sin(2 * np.pi * y / L))[None, :, None], (n, n, n)
        ).copy()
    )
    s.set_velocity(0, ux)
    s.update_properties()
    rav = rl + (rg - rl) * C.mean()
    out = []
    for _ in range(steps):
        Cn = np.array(s.get_vof())
        un = [np.array(g()) for g in (s.get_u, s.get_v, s.get_w)]
        s.step()
        u1 = [np.array(g()) for g in (s.get_u, s.get_v, s.get_w)]
        rho = rl + (rg - rl) * Cn
        pr = []
        for c in range(3):
            rf = 0.5 * (rho + np.roll(rho, 1, axis=c))  # get_u()[i] = face (i-1, i)
            pr.append(float(np.sum(rf * (u1[c] - un[c]))) / dt / n**3 / (rav * ca.G))
        out.append(pr)
    o = np.array(out)
    return (
        f"CONSERVE n={n} rho_ratio={rho_ratio:g} mu_ratio={mu_ratio:g} stress={stress} "
        f"adv={adv} grav={grav} sigma={sigma} dt={dt:.3e} steps={steps} PRED/<rho>g first=({o[0, 0]:+.3e},{o[0, 1]:+.3e},"
        f"{o[0, 2]:+.3e}) mean_x={o[:, 0].mean():+.3e} max|.|={np.abs(o).max():.3e}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--set", default="base", choices=("base", "forces"))
    a = p.parse_args()
    cfgs = [
        (1.0, 1.0, "laplacian", False, 1.0),
        (0.1, 1.0, "laplacian", False, 1.0),
        (1.0, 0.1, "laplacian", False, 1.0),
        (1.0, 0.1, "full", False, 1.0),
        (0.1, 0.1, "laplacian", False, 1.0),
        (0.1, 0.1, "full", False, 1.0),
        (0.1, 0.1, "full", False, 0.5),
        (0.1, 0.1, "full", True, 1.0),
    ]
    if (
        a.set == "forces"
    ):  # the case-A forces one at a time (capillary dt binds with sigma)
        for grav, sig, dtf in (
            (True, "none", 0.5),
            (False, "single", 0.1),
            (False, "blk", 0.1),
            (True, "blk", 0.1),
        ):
            print(
                one(a.n, 0.1, 0.1, "full", a.steps, False, dtf, grav, sig), flush=True
            )
        return
    for rr, mr, st, adv, dtf in cfgs:
        print(one(a.n, rr, mr, st, a.steps, adv, dtf), flush=True)


if __name__ == "__main__":
    main()
