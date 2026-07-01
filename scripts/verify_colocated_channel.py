#!/usr/bin/env python3
"""Phase-5b verification (collocated grid): developing plane channel -- the canonical INFLOW/OUTFLOW
benchmark. A uniform stream enters at -x (inflow Dirichlet), leaves at +x (outflow: zero-gradient velocity
+ Dirichlet p=0), between no-slip walls at +-y; quasi-2D (periodic z). Exercises the collocated open-boundary
machinery: the inflow reflection ghost (carrying the prescribed normal velocity through the open face),
the zero-gradient outflow velocity ghost, and the outflow face correction on the MAC field that lets mass
leave (the shared operator/flux openness split is grid-agnostic).

Develops into the parabolic Poiseuille profile (u_max/U_mean -> 1.5). We check global mass conservation,
the developed profile, and incompressibility. NOTE on the divergence: the *face* field is divergence-free
to machine precision in the interior; at the open outflow the approximate projection leaves a small O(h^2)
residual (vs the staggered exact projection), so the divergence tolerance here is looser than the staggered
1e-6 and is checked to SHRINK with resolution. Global mass conservation (flux_in == flux_out) is the
primary continuity check.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow as sdflow  # noqa: E402


def run(H, L, Re=100.0, U=1.0, nz=4, max_steps=8000, dt=0.5):
    nu = U * H / Re
    s = sdflow.SolverColocated(L, H, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(True)
    s.set_domain_bc(0, 2, U, 0.0, 0.0)   # -x inflow: uniform stream
    s.set_domain_bc(1, 3)                # +x outflow
    s.set_domain_bc(2, 1); s.set_domain_bc(3, 1)  # -y, +y no-slip walls
    s.set_velocity_solver_params(60)
    s.set_pressure_pcg(True, 400, 1e-9)
    s.set_pressure_geometry(np.asfortranarray(np.full((L, H, nz), 1e30)))

    prev = 0.0
    steps = max_steps
    for it in range(max_steps):
        s.step()
        if it % 50 == 49:
            m = float(s.get_u()[L - 4, H // 2, nz // 2])
            if it > 1000 and abs(m - prev) < 1e-5 * (abs(m) + 1e-30):
                steps = it + 1
                break
            prev = m
    u = s.get_u(); div = float(s.max_open_divergence())
    flux_in = float(u[2, :, nz // 2].sum())
    flux_out = float(u[L - 3, :, nz // 2].sum())
    mass_err = abs(flux_out - flux_in) / (abs(flux_in) + 1e-30)
    prof = u[L - 4, :, nz // 2]
    U_mean = float(prof.mean())
    eta = (np.arange(H) + 0.5) / H
    parab = 6.0 * U_mean * eta * (1.0 - eta)
    prof_rms = float(np.sqrt(np.mean((prof - parab) ** 2)) / (abs(U_mean) + 1e-30))
    ratio = float(prof.max() / (U_mean + 1e-30))
    return dict(steps=steps, mass_err=mass_err, div=div, ratio=ratio, prof_rms=prof_rms, H=H, L=L)


def main():
    print("=== sdflow phase-5b: collocated developing channel (inflow/outflow) ===")
    print(f"{'H':>4} {'L':>5} {'steps':>6} {'mass_err':>10} {'maxdiv':>10} {'u_max/Um':>9} {'prof_rms':>9}")
    rows = []
    for H, L in ((16, 112), (32, 224)):
        r = run(H, L)
        print(f"{r['H']:4d} {r['L']:5d} {r['steps']:6d} {r['mass_err']:10.2e} {r['div']:10.2e} {r['ratio']:9.4f} {r['prof_rms']:9.4f}")
        rows.append(r)
    fine = rows[-1]
    mass_ok = fine["mass_err"] < 1e-3
    div_shrinks = rows[1]["div"] < rows[0]["div"]
    div_small = fine["div"] < 1e-4
    developed = 1.45 < fine["ratio"] < 1.55 and fine["prof_rms"] < 0.03
    ok = mass_ok and div_small and div_shrinks and developed
    print(f"  mass conservation (flux_out==flux_in, <1e-3): {mass_ok} ({fine['mass_err']:.2e})")
    print(f"  outflow divergence small (<1e-4) and shrinks with N: {div_small and div_shrinks} "
          f"({rows[0]['div']:.1e} -> {rows[1]['div']:.1e})")
    print(f"  developed Poiseuille (u_max/U_mean~1.5, rms<0.03): {developed} "
          f"({fine['ratio']:.4f}, {fine['prof_rms']:.4f})")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
