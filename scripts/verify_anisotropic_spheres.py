#!/usr/bin/env python3
"""Phase 2 gate G2, the LADDER half (doc/anisotropic_metric.md §8.3).

The regression suite's Zick & Homsy sphere (`tests/regression/sdflow_regression.py`, case
`zh_sphere`: solid fraction phi = 0.216, R = 0.3722 L, K_ref = 7.442, config rho 1 / mu 0.1 /
dt 60 / F 1e-3, 80 velocity sweeps, MG-PCG 300 / 1e-8, cut-cell pressure, advection off) on the
cube `L^3` resolved by cells **(N, 2N, N/2)** -- h = (dx, dx/2, 2dx) -- with the SDF sampled at the
PHYSICAL cell centres, beside the cubic control (N, N, N) in cell units at every rung.

What it gates (§8.3, the `script` bullet):
  * the Richardson-extrapolated `K_s,inf` is within **2 %** of 7.442 (the regression's `extrap_rel`);
  * the fitted order of the stretched sequence `p_s` lies in **[1.5, 2.5]**;
  * the pressure iterations per step on the stretched grid are **<= cubic + 2** at every N (the §5
    aspect-ratio coarsening rule is what buys that).

The order/extrapolation estimator is the regression's own `fit_order` (f(N) = f_inf + C N^-p), so
the cubic column is directly comparable to `perf_baseline.json`'s recorded `order` / `extrapolated`.

Usage (from the repo root, venv active):
    SDFLOW_BUILD=build python scripts/verify_anisotropic_spheres.py
    SDFLOW_BUILD=build python scripts/verify_anisotropic_spheres.py --grids 16,24,32
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                os.environ.get("SDFLOW_BUILD", "build"))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests",
                                                "regression")))
from peclet import flow as sdflow  # noqa: E402
from sdflow_regression import CFG, fit_order, zh_ref  # noqa: E402

PHI = 0.216
K_REF = zh_ref(PHI)  # 7.442


def run(nx, ny, nz, arm, L, cfg):
    """One rung. `arm=False` is the cell-unit cubic control; `arm=True` arms the cube L^3."""
    R = (PHI * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * L
    levels = max(2, int(np.floor(np.log2(nx))) - 1)
    kw = {} if not arm else {"extent": (L, L, L), "origin": (0.0, 0.0, 0.0)}
    s = sdflow.Solver((nx, ny, nz), **kw)
    s.set_rho(cfg["rho"]); s.set_mu(cfg["mu"]); s.set_dt(cfg["dt"])
    s.set_body_force(cfg["F"], 0.0, 0.0)
    s.set_advection(False)
    s.set_velocity_solver_params(cfg["vel_sweeps"])
    s.set_pressure_multigrid(True, levels=levels)
    s.set_pressure_pcg(True, cfg["pcg_maxit"], cfg["pcg_rtol"])
    cx, cy, cz = s.cell_centers()
    c = 0.5 * L
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    sdf = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True)

    t0 = time.time()
    prev, steps, p_iters = 0.0, 0, []
    for it in range(cfg["max_steps"]):
        s.step()
        steps += 1
        p_iters.append(s.last_pressure_iterations())
        if it % cfg["check_every"] == cfg["check_every"] - 1:
            m = float(s.get_u().mean())
            if it >= cfg["min_steps"] and abs(m - prev) < cfg["conv_tol"] * (abs(m) + 1e-30):
                break
            prev = m
    wall = time.time() - t0
    umean = float(s.get_u().mean())
    K = cfg["F"] * L ** 3 / (6.0 * np.pi * cfg["mu"] * R * umean)
    half = p_iters[len(p_iters) // 2:]
    return {"K": K, "iters": float(np.median(half)), "steps": steps,
            "div": float(s.max_open_divergence()), "spacing": tuple(s.spacing),
            "aniso": len(set(s.spacing)) > 1, "wall": wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="16,24,32,48,64")
    a = ap.parse_args()
    grids = [int(g) for g in a.grids.split(",")]
    cfg = dict(CFG)

    print(f"Zick & Homsy sphere, phi = {PHI}, K_ref = {K_REF}")
    print(f"{'N':>4} {'cells (stretched)':>20} {'spacing':>20} "
          f"{'K_cubic':>10} {'K_stretch':>10} {'err_c %':>9} {'err_s %':>9} "
          f"{'it_c':>5} {'it_s':>5} {'steps_s':>8}")
    Kc, Ks, itc, its = [], [], [], []
    ok = True
    for N in grids:
        c = run(N, N, N, False, float(N), cfg)
        s = run(N, 2 * N, N // 2, True, float(N), cfg)
        Kc.append(c["K"]); Ks.append(s["K"]); itc.append(c["iters"]); its.append(s["iters"])
        ec = 100.0 * abs(c["K"] - K_REF) / K_REF
        es = 100.0 * abs(s["K"] - K_REF) / K_REF
        sp = "(%.4g, %.4g, %.4g)" % s["spacing"]
        print(f"{N:>4} {f'{N}x{2*N}x{N//2}':>20} {sp:>20} "
              f"{c['K']:>10.5f} {s['K']:>10.5f} {ec:>9.4f} {es:>9.4f} "
              f"{c['iters']:>5.1f} {s['iters']:>5.1f} {s['steps']:>8d}")
        if not s["aniso"]:
            print("  FAIL: the stretched rung did not arm an anisotropic metric")
            ok = False
        if s["iters"] > c["iters"] + 2:
            print(f"  FAIL: pressure iters/step {s['iters']} > cubic {c['iters']} + 2")
            ok = False

    pc, Kc_inf = fit_order(grids, Kc)
    ps, Ks_inf = fit_order(grids, Ks)
    ec_inf = abs(Kc_inf - K_REF) / K_REF
    es_inf = abs(Ks_inf - K_REF) / K_REF
    print()
    print(f"  cubic     order p = {pc:.4f}   K_inf = {Kc_inf:.5f}   |K_inf - K_ref|/K_ref = "
          f"{100*ec_inf:.4f} %")
    print(f"  stretched order p = {ps:.4f}   K_inf = {Ks_inf:.5f}   |K_inf - K_ref|/K_ref = "
          f"{100*es_inf:.4f} %")
    print()
    if not (es_inf <= 0.02):
        print(f"  FAIL: extrapolated K_s,inf off by {100*es_inf:.4f} % (bound 2 %)"); ok = False
    if not (1.5 <= ps <= 2.5):
        print(f"  FAIL: stretched order {ps:.4f} outside [1.5, 2.5]"); ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
