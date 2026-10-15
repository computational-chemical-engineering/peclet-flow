#!/usr/bin/env python3
"""D1 -- the pressure-tolerance study of doc/vof_step_performance_design.md sec. 5.12 (WO-0 starts
it, WO-13 analyses it). An instrument, not a gate: it records, it does not judge.

For every pressure-driver rtol in --rtols (default 1e-10 [reference], 1e-9, 1e-8, 1e-7, 1e-6; the
momentum solve follows the pressure driver's rtol by default) it runs up to three cases and writes
one JSON per (case, rtol) into --out:

  static   tests/study/vof_surface_tension.py's static droplet (all resolution rungs, 60 steps
           each): max|u| per rung  -> static_rtol<R>.json
  hysing   Hysing case 1 through the BLOCK path (tests/study/vof_blocks_ns.py run_hysing, nx 64,
           T = 3): peak rise velocity, its time, final centroid, volume drift -> hysing_rtol<R>.json
  column   the bubble column from ckpt_t43, --column-steps steps (default 2000), MG-PCG at the
           rtol, device flux constraint (the production protocol of run_peclet.py): per step dt,
           pressure iterations, momentum residual and time (the Python API exposes no momentum
           iteration count), max_open_divergence_projected(), every marker's volume and
           x-velocity; every 50 steps run_peclet.sample() (gas volume, drift = rise speed)
           -> column_rtol<R>.json

A (case, rtol) whose JSON already exists is skipped, so re-running resumes.

Usage:
    PYTHONPATH=<flow build> OMP_NUM_THREADS=8 OMP_PROC_BIND=false \\
        python tests/study/vof_perf/d1_tolerance.py --out DIR [--cases static,hysing,column]
            [--rtols 1e-10,1e-9,1e-8,1e-7,1e-6] [--column-steps 2000] [--ckpt PATH] [--case-dir DIR]
            [--vcycle-precision auto|fp64|fp32]   (the column's V-cycle precision, §4 of
                                                 doc/vof_projection_cost_design.md)
"""
import json
import os
import sys
import time

import numpy as np

A = list(sys.argv)


def arg(name, default, cast=str):
    return cast(A[A.index(name) + 1]) if name in A else default


OUT = arg("--out", "d1")
CASES = arg("--cases", "static,hysing,column").split(",")
RTOLS = [float(r) for r in arg("--rtols", "1e-10,1e-9,1e-8,1e-7,1e-6").split(",")]
NCOL = arg("--column-steps", 2000, int)
CASE_DIR = arg("--case-dir", os.path.expanduser(
    "~/Codes/peclet-examples/benchmarks/bubble-column/scripts"))
CKPT = arg("--ckpt", os.path.expanduser(
    "~/Codes/bubble_column_perf/ckpt_t43.npz"))
# doc/vof_projection_cost_design.md §4: the V-cycle precision of the COLUMN case (its PCG is the
# only Krylov solve here; the static and Hysing cases run Chebyshev, where the FP32 V-cycle never
# applies, so they keep the build's default)
VPREC = arg("--vcycle-precision", "")

# the two study scripts read sys.argv at import: hand them none of ours
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vof_surface_tension as vst  # noqa: E402
import vof_blocks_ns as vbn  # noqa: E402
sys.path.insert(0, CASE_DIR)
import run_peclet as rp  # noqa: E402
import peclet.flow as pf  # noqa: E402


def run_static(rtol):
    vst.RTOL = rtol
    vst.gate_static()
    return {"rungs": list(vst.STATIC_RESULTS),
            "max_u": max(r["max_u"] for r in vst.STATIC_RESULTS)}


def run_hysing(rtol):
    vbn.RTOL = rtol
    r = vbn.run_hysing(64, True, 3.0, verbose=False)
    return {"vmax": r["vmax"], "tvmax": r["tvmax"], "yc": r["yc"], "vdrift": r["vdrift"],
            "steps": r["steps"], "p_iters_max": r["solve"].iters, "div_max": r["solve"].div,
            "capped": not r["solve"].valid}


def run_column(rtol):
    step, t, blocks, vel, pres, series = rp.load_ckpt(CKPT)
    s = rp.build(blocks)
    for c, a in enumerate(vel):
        s.set_velocity(c, a)
    s.set_field("p", pres)
    s.diagnostics.set_vof_step_parity(step)
    s.set_pressure_pcg(True, 800, rtol)
    if VPREC:
        s.diagnostics.set_pressure_vcycle_precision(VPREC)
    v0 = np.array([b["volume"] for b in s.diagnostics.vof_block_stats()])
    rec = {k: [] for k in ("t", "dt", "p_iters", "mom_residual", "mom_time", "step_time",
                           "div_proj", "vol", "ux")}
    samples = [rp.sample(s, t)]
    for i in range(NCOL):
        L = s.vof_step_limits()
        dt = min(0.25 * L["cfl_dt"], 0.25 * L["capillary_dt"])
        s.set_dt(dt)
        s.step()
        t += dt
        d = s.diagnostics
        tm = d.last_step_timers()
        st = d.vof_block_stats()
        rec["t"].append(t)
        rec["dt"].append(dt)
        rec["p_iters"].append(d.last_pressure_iterations())
        rec["mom_residual"].append(d.last_momentum_residual())
        rec["mom_time"].append(tm["momentum"])
        rec["step_time"].append(tm["step"])
        rec["div_proj"].append(s.max_open_divergence_projected())
        rec["vol"].append([b["volume"] for b in st])
        rec["ux"].append([b["velocity"][0] / rp.S for b in st])
        if (i + 1) % 50 == 0:
            samples.append(rp.sample(s, t))
            print(f"    column rtol {rtol:g}: step {i+1}/{NCOL}  p_iters {rec['p_iters'][-1]}  "
                  f"div {rec['div_proj'][-1]:.2e}", flush=True)
    vol = np.array(rec["vol"])
    ux = np.array(rec["ux"])
    half = NCOL // 2
    rec.update({
        "vol0": v0.tolist(),
        "bubble_vol_drift_max": float(np.max(np.abs(vol / v0 - 1.0))),
        "total_vol_drift_max": float(np.max(np.abs(vol.sum(axis=1) / v0.sum() - 1.0))),
        "rise_mean_last_half": float(-np.mean(ux[half:])),    # + = towards -x (up), per marker
        "drift_mean_last_half": float(np.mean([q["drift"] for q in samples[len(samples) // 2:]])),
        "samples": samples})
    return rec


RUN = {"static": run_static, "hysing": run_hysing, "column": run_column}
os.makedirs(OUT, exist_ok=True)
print(f"D1: cases {CASES}  rtols {RTOLS}  module {pf.__file__}", flush=True)
for case in CASES:
    for rtol in RTOLS:
        path = os.path.join(OUT, f"{case}_rtol{rtol:.0e}.json")
        if os.path.exists(path):
            print(f"  {case} rtol {rtol:g}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        print(f"  {case} rtol {rtol:g}: start", flush=True)
        res = RUN[case](rtol)
        res.update({"case": case, "rtol": rtol, "wall_s": time.time() - t0,
                    "loadavg": open("/proc/loadavg").read().strip(), "module": pf.__file__})
        with open(path + ".tmp", "w") as f:
            json.dump(res, f)
        os.replace(path + ".tmp", path)
        print(f"  {case} rtol {rtol:g}: done in {res['wall_s']:.0f} s -> {path}", flush=True)
print("D1 DONE", flush=True)
