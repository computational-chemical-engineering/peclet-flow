#!/usr/bin/env python3
"""Single-GPU accuracy + efficiency regression suite for the `sdflow` cut-cell IBM Stokes solver.

Three creeping-flow (Stokes) cases, each as a GRID-CONVERGENCE study:
  * zh_sphere      -- simple-cubic single sphere; drag factor K vs Zick & Homsy (1982) (external ref);
  * random_spheres -- a small packed bed of (reproducibly) jittered spheres; Darcy permeability k;
  * hollow_rings   -- a small packed bed of Raschig rings (hollow cylinders); Darcy permeability k.

For each grid N we record the ACCURACY metric (K or k) and the EFFICIENCY counters the solver exposes:
total pressure-solver (MG-PCG) iterations, per-step pressure iterations, Picard outer iterations, the
number of steps to steady state, the cut-cell flux divergence, and the wall-clock time. Across the grid
sweep we fit the observed order of convergence p (f(N) = f_inf + C N^-p) and the Richardson-extrapolated
value f_inf.

All numbers are saved to perf_baseline.json. Re-running compares against that baseline within tolerances,
so a code change that degrades accuracy OR efficiency is caught.

Usage:
  python tests/regression/sdflow_regression.py              # run + check against the baseline (exit 0/1)
  python tests/regression/sdflow_regression.py --update     # run + (re)write the baseline
  python tests/regression/sdflow_regression.py --cases zh_sphere,random_spheres
  python tests/regression/sdflow_regression.py --build build_mpi    # pick the sdflow build dir
  python tests/regression/sdflow_regression.py --quick      # coarser grids, looser march (fast smoke)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE = os.path.join(HERE, "perf_baseline.json")

# Zick & Homsy (1982), simple cubic: solid fraction c -> Stokes drag factor K.
ZH_PHI = [0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236]
ZH_K = [1.096, 1.212, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1]


def zh_ref(phi):
    return float(np.interp(phi, ZH_PHI, ZH_K))


# --------------------------------------------------------------------------- geometry (sdf[x,y,z], <0 solid)
def _grid(N):
    g = np.arange(N) + 0.5  # cell centres
    return np.meshgrid(g, g, g, indexing="ij")


def _minimg(d, N):
    return d - N * np.round(d / N)


def sdf_zh_sphere(N, phi=0.216):
    """Single SC sphere centred in the periodic cube; returns (sdf, info)."""
    R = (phi * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * N
    X, Y, Z = _grid(N)
    c = N / 2.0
    sdf = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R
    return sdf, {"R": R, "phi": phi, "K_ref": zh_ref(phi)}


def sdf_random_spheres(N, n=8, r_frac=0.18, jit=0.06, seed=12345):
    """Small packed bed: `n` spheres of radius r_frac*N on a jittered 2x2x2 lattice (fixed seed). The shape
    is self-similar in N (same relative geometry, finer grid) -> a true grid-convergence study of k*."""
    rng = np.random.default_rng(seed)
    R = r_frac * N
    base = np.array([[(i + 0.5) / 2.0, (j + 0.5) / 2.0, (k + 0.5) / 2.0]
                     for i in range(2) for j in range(2) for k in range(2)])
    centres = ((base + jit * rng.standard_normal(base.shape)) % 1.0) * N
    X, Y, Z = _grid(N)
    sdf = np.full((N, N, N), 1e30)
    for cx, cy, cz in centres:
        dx = _minimg(X - cx, N); dy = _minimg(Y - cy, N); dz = _minimg(Z - cz, N)
        sdf = np.minimum(sdf, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return sdf, {"R": R, "n": n, "r_frac": r_frac}


def _hollow_cyl_sdf(X, Y, Z, c, axis, r_out, r_in, H, N):
    """SDF of one Raschig ring (hollow cylinder): annulus [r_in,r_out] x slab |axial|<=H/2, CSG-intersection."""
    ax = np.asarray(axis, float); ax = ax / np.linalg.norm(ax)
    dx = _minimg(X - c[0], N); dy = _minimg(Y - c[1], N); dz = _minimg(Z - c[2], N)
    z = dx * ax[0] + dy * ax[1] + dz * ax[2]                  # axial coord
    rx = dx - z * ax[0]; ry = dy - z * ax[1]; rz = dz - z * ax[2]
    rho = np.sqrt(rx * rx + ry * ry + rz * rz)                # radial distance from the axis
    d_annulus = np.maximum(r_in - rho, rho - r_out)
    d_slab = np.abs(z) - 0.5 * H
    return np.maximum(d_annulus, d_slab)                      # <0 inside the ring wall


def sdf_hollow_rings(N):
    """Small packed bed of 3 Raschig rings at fixed positions/orientations (reproducible)."""
    rO, rI, H = 0.22 * N, 0.12 * N, 0.34 * N
    rings = [((0.30 * N, 0.32 * N, 0.30 * N), (1, 0, 0)),
             ((0.70 * N, 0.68 * N, 0.55 * N), (0, 1, 0)),
             ((0.45 * N, 0.50 * N, 0.78 * N), (0, 0, 1))]
    X, Y, Z = _grid(N)
    sdf = np.full((N, N, N), 1e30)
    for c, axis in rings:
        sdf = np.minimum(sdf, _hollow_cyl_sdf(X, Y, Z, c, axis, rO, rI, H, N))
    return sdf, {"r_out": rO, "r_in": rI, "H": H, "n_rings": len(rings)}


CASES = {
    "zh_sphere":      {"sdf": sdf_zh_sphere,      "grids": [16, 24, 32, 48, 64], "metric": "K"},
    "random_spheres": {"sdf": sdf_random_spheres, "grids": [24, 32, 48, 64],     "metric": "k*"},
    "hollow_rings":   {"sdf": sdf_hollow_rings,   "grids": [24, 32, 48, 64],     "metric": "k*"},
}

# Fixed solver config shared by every case (so the recorded efficiency is comparable across runs).
CFG = dict(rho=1.0, mu=0.1, dt=60.0, F=1e-3, vel_sweeps=80, pcg_maxit=300, pcg_rtol=1e-8,
           coarse="rediscretized", conv_tol=1e-5, check_every=5, max_steps=400, min_steps=15)


# --------------------------------------------------------------------------- run one (case, N)
def run_case(name, N, cfg, quiet=True, solver="staggered"):
    from peclet import flow as sdflow
    spec = CASES[name]
    sdf, info = spec["sdf"](N)
    levels = max(2, int(np.floor(np.log2(N))) - 1)

    SolverCls = sdflow.SolverColocated if solver == "colocated" else sdflow.Solver
    s = SolverCls(N, N, N)
    s.set_rho(cfg["rho"]); s.set_mu(cfg["mu"]); s.set_dt(cfg["dt"])
    s.set_body_force(cfg["F"], 0.0, 0.0)
    s.set_advection(False)  # creeping Stokes
    s.set_velocity_solver_params(cfg["vel_sweeps"])
    s.set_pressure_multigrid(True, levels=levels)
    s.set_pressure_pcg(True, cfg["pcg_maxit"], cfg["pcg_rtol"])
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse=cfg["coarse"])

    deep_solid = sdf < -2.0
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

    u = s.get_u()
    umean = float(u.mean())
    div = float(s.max_open_divergence())
    u_solid = float(np.abs(u[deep_solid]).max()) if deep_solid.any() else 0.0
    if spec["metric"] == "K":            # Zick & Homsy drag factor (dimensionless)
        metric = cfg["F"] * N ** 3 / (6.0 * np.pi * cfg["mu"] * info["R"] * umean)
    else:                                # dimensionless permeability k* = k/L^2 = mu <u> / (F N^2)
        metric = cfg["mu"] * umean / (cfg["F"] * N ** 2)
    half = p_iters[len(p_iters) // 2:]
    return {
        "N": N, "metric": float(metric),
        "pressure_iters_total": int(sum(p_iters)),
        "pressure_iters_per_step": float(np.median(half)),
        "outer_iters": int(s.last_outer_iterations()),
        "steps": int(steps), "divergence": div, "max_u_solid": u_solid,
        "walltime_s": float(wall),
    }


def fit_order(Ns, vals):
    """Fit f(N) = f_inf + C N^-p (grid-search p, linear LS for f_inf,C). Returns (order p, f_inf)."""
    Ns = np.asarray(Ns, float); vals = np.asarray(vals, float)
    best = None
    for p in np.linspace(0.3, 4.0, 371):
        A = np.vstack([np.ones_like(Ns), Ns ** (-p)]).T
        coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
        ssr = float(((vals - A @ coef) ** 2).sum())
        if best is None or ssr < best[0]:
            best = (ssr, float(p), float(coef[0]))
    return best[1], best[2]  # order, extrapolated f_inf


def run_all(cfg, cases, solver="staggered"):
    out = {}
    for name in cases:
        grids = CASES[name]["grids"]
        per = {}
        print(f"\n[{name}] ({solver}) grids {grids} ...", flush=True)
        for N in grids:
            r = run_case(name, N, cfg, solver=solver)
            per[str(N)] = r
            print(f"  N={N:3d}  {CASES[name]['metric']}={r['metric']:.5g}  "
                  f"p_iters_tot={r['pressure_iters_total']:5d} (/step {r['pressure_iters_per_step']:.0f})  "
                  f"steps={r['steps']:3d}  div={r['divergence']:.1e}  {r['walltime_s']:.1f}s", flush=True)
        Ns = grids
        vals = [per[str(N)]["metric"] for N in Ns]
        order, finf = fit_order(Ns, vals)
        entry = {"grids": grids, "metric_name": CASES[name]["metric"], "per_grid": per,
                 "order": order, "extrapolated": finf}
        if name == "zh_sphere":
            entry["reference"] = zh_ref(0.216)
            entry["errors_pct"] = {str(N): 100.0 * abs(per[str(N)]["metric"] - entry["reference"]) /
                                   entry["reference"] for N in Ns}
        out[name] = entry
        print(f"  -> order p={order:.2f}, extrapolated {CASES[name]['metric']}_inf={finf:.5g}", flush=True)
    return out


# --------------------------------------------------------------------------- baseline compare
TOL = dict(metric_rel=0.015, order_abs=0.4, extrap_rel=0.02,
           piter_total_rel=0.25, piter_step_abs=2.0, steps_rel=0.35, div_floor=1e-7)


def compare(base, cur):
    ok = True
    lines = []
    for name in cur:
        if name not in base:
            lines.append(f"[{name}] NEW case (no baseline) -- record with --update"); ok = False; continue
        b, c = base[name], cur[name]
        mname = c["metric_name"]
        lines.append(f"\n[{name}]  (metric={mname})")
        # order + extrapolated value
        d_ord = abs(c["order"] - b["order"])
        s_ord = "ok" if d_ord <= TOL["order_abs"] else "FAIL"; ok &= s_ord == "ok"
        lines.append(f"  order p:        base {b['order']:.2f}  cur {c['order']:.2f}  (d={d_ord:.2f})  [{s_ord}]")
        d_ext = abs(c["extrapolated"] - b["extrapolated"]) / (abs(b["extrapolated"]) + 1e-30)
        s_ext = "ok" if d_ext <= TOL["extrap_rel"] else "FAIL"; ok &= s_ext == "ok"
        lines.append(f"  {mname}_inf:        base {b['extrapolated']:.5g}  cur {c['extrapolated']:.5g}  "
                     f"(rel={d_ext*100:.2f}%)  [{s_ext}]")
        for N in [g for g in c["grids"] if str(g) in b.get("per_grid", {})]:
            bg, cg = b["per_grid"][str(N)], c["per_grid"][str(N)]
            dm = abs(cg["metric"] - bg["metric"]) / (abs(bg["metric"]) + 1e-30)
            sm = "ok" if dm <= TOL["metric_rel"] else "FAIL"; ok &= sm == "ok"
            di = abs(cg["pressure_iters_total"] - bg["pressure_iters_total"]) / (bg["pressure_iters_total"] + 1e-30)
            si = "ok" if di <= TOL["piter_total_rel"] else "FAIL"; ok &= si == "ok"
            dps = abs(cg["pressure_iters_per_step"] - bg["pressure_iters_per_step"])
            sps = "ok" if dps <= TOL["piter_step_abs"] else "FAIL"; ok &= sps == "ok"
            div_lim = max(TOL["div_floor"], 3.0 * bg["divergence"])
            sd = "ok" if cg["divergence"] <= div_lim else "FAIL"; ok &= sd == "ok"
            lines.append(
                f"  N={N:3d}  {mname} {cg['metric']:.5g} ({dm*100:+.2f}%)[{sm}]  "
                f"p_iter_tot {cg['pressure_iters_total']} ({di*100:+.1f}%)[{si}]  "
                f"/step {cg['pressure_iters_per_step']:.0f}[{sps}]  "
                f"div {cg['divergence']:.1e}[{sd}]  "
                f"steps {cg['steps']} vs {bg['steps']}  "
                f"t {cg['walltime_s']:.1f}s vs {bg['walltime_s']:.1f}s")
    return ok, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="(re)write the baseline instead of checking")
    ap.add_argument("--cases", default=",".join(CASES), help="comma-separated subset of cases")
    ap.add_argument("--build", default="build", help="sdflow build dir under the repo root")
    ap.add_argument("--solver", default="staggered", choices=["staggered", "colocated"],
                    help="which grid variant to run (sdflow.Solver / sdflow.SolverColocated)")
    ap.add_argument("--quick", action="store_true", help="coarser grids + looser march (fast smoke)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, args.build))
    baseline = BASELINE if args.solver == "staggered" else os.path.join(HERE, "perf_baseline_colocated.json")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    cfg = dict(CFG)
    if args.quick:
        cfg.update(max_steps=120, conv_tol=3e-4)
        for c in CASES.values():
            c["grids"] = c["grids"][:3]

    t0 = time.time()
    cur = run_all(cfg, cases, solver=args.solver)
    print(f"\n(total {time.time()-t0:.0f}s)")

    if args.update:
        payload = {"_meta": {"generated": time.strftime("%Y-%m-%d %H:%M"), "solver": args.solver,
                             "config": cfg, "tol": TOL}, **cur}
        with open(baseline, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote baseline -> {baseline}")
        return 0

    if not os.path.exists(baseline):
        print(f"\nNO baseline at {baseline}; run with --update first.")
        return 1
    base = json.load(open(baseline))
    ok, report = compare(base, cur)
    print(report)
    print(f"\n=== regression: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
