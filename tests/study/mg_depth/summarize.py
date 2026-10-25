#!/usr/bin/env python3
"""summarize.py <out_dir> <row> [--md]: one line per (row, depth) from run_matrix.sh's files.

Per depth: the bottom's global dims and engine (from the probe's debug lines: '[mg] direct bottom'
-> direct, '[agmg-build]' -> GraphAMG, neither -> smoothed), the median over rounds of the mean
ms/step, the pressure iterations mean/min/max (round 1; 'VAR' if the rounds differ), the max
per-step max_open_divergence_projected(), FP32 restarts (the solver's stderr warning lines, summed
over rounds), capped/failed solves, and the max load average seen (> 24 => NOT JUDGED). For M1
it adds the G-PERF-WS (a)-(c) checkpoint with L5 as the candidate (doc/mg_depth_policy.md §4.4).
--md prints a markdown table instead.
"""
import glob
import json
import os
import re
import sys

import numpy as np

OUT, ROW = sys.argv[1], sys.argv[2]
MD = "--md" in sys.argv
D = os.path.join(OUT, ROW)
LOADMAX = 24.0


def engine(probe):
    try:
        t = open(probe, errors="replace").read()
    except OSError:
        return "?", None
    eng = "direct" if "[mg] direct bottom:" in t else ("GraphAMG" if "[agmg-build]" in t else "smoothed")
    if "probe rc 0" not in t:
        eng += "(probe-fail)"
    lv = re.findall(r"\[mg\]\s+L(\d+) (?:dims|global)\s+(\d+)x\s*(\d+)x\s*(\d+)", t)
    return eng, (lv[-1] if lv else None)


rows = {}
for js in sorted(glob.glob(os.path.join(D, "*_r[0-9]*.json"))):
    m = re.match(r"(.*)_L(\d+)_r(\d+)\.json$", os.path.basename(js))
    sub, L, r = m.group(1), int(m.group(2)), int(m.group(3))
    res = json.load(open(js))
    log = js[:-5] + ".log"
    txt = open(log, errors="replace").read() if os.path.exists(log) else ""
    res["_fp32_restarts"] = txt.count("the FP32 V-cycle did not converge")
    g = re.findall(r"gpu_apps_before (\d+) gpu_apps_after (\d+)", txt)
    res["_gpu"] = max(int(x) for x in g[-1]) if g else -1
    rows.setdefault((sub, L), {})[r] = res
# runs that failed (no json)
fails = []
for lg in sorted(glob.glob(os.path.join(D, "*_r[0-9]*.log"))):
    if not os.path.exists(lg[:-4] + ".json"):
        fails.append(os.path.basename(lg))

out = {}
lines = []
for (sub, L), rr in sorted(rows.items()):
    r1 = rr[min(rr)]
    eng, _ = engine(os.path.join(D, f"{sub}_L{L}_probe.log"))
    ms = [v["step_ms_mean"] for v in rr.values()]
    med = float(np.median(ms))
    its = {(v["iters_mean"], v["iters_min"], v["iters_max"]) for v in rr.values()}
    loads = [x for v in rr.values() for x in (v["load_before"][0], v["load_after"][0])]
    lmax = max(loads)
    capped = sum(v["solve_failed"] for v in rr.values())
    caps = max(v["iters_max"] for v in rr.values())
    fr = sum(v["_fp32_restarts"] for v in rr.values())
    gpu = max(v["_gpu"] for v in rr.values())
    judged = "" if lmax <= LOADMAX else " NOT-JUDGED(load)"
    extra = ""
    for k in ("K", "k_star"):
        if k in r1:
            extra += f" {k}={r1[k]:.10g}"
    if "steps" in r1 and r1["case"] in ("zh_sphere", "random_spheres"):
        extra += f" steps={r1['steps']} itmed2={r1['iters_median_2nd_half']:g} umaxsolid={r1['max_u_solid']:.3e}"
    out[(sub, L)] = dict(ms=med, iters=r1["iters_mean"], bottom=r1["bottom"], eng=eng, lmax=lmax)
    b = "x".join(map(str, r1["bottom"]))
    line = (f"{sub} L{L}{'(default)' if L == 0 else ''}: levels {r1['levels']} bottom {b} ({r1['bottom_cells']}) "
            f"{eng} | {med:.2f} ms/step (rounds {', '.join(f'{x:.1f}' for x in ms)}) | iters "
            f"{r1['iters_mean']:.2f} [{r1['iters_min']},{r1['iters_max']}]{' VAR' if len(its) > 1 else ''}"
            f" | div max {r1['div_max']:.3e} | fp32 {r1['prec_last']} restarts {fr} | failed {capped}"
            f" | cheb_restarts {r1['cheb_restarts']} | load max {lmax:.1f}{judged} | gpu apps max {gpu}{extra}")
    lines.append((sub, L, r1, eng, med, ms, b, fr, capped, lmax, extra))
    if not MD:
        print("ROW " + line)
if MD:
    print("| row | L | levels | bottom (cells) | engine | ms/step median (rounds) | iters mean [min,max] |"
          " div max | FP32 restarts | failed | load max | extra |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sub, L, r1, eng, med, ms, b, fr, capped, lmax, extra in lines:
        print(f"| {sub} | {L} | {r1['levels']} | {b} ({r1['bottom_cells']}) | {eng} | {med:.2f} "
              f"({', '.join(f'{x:.1f}' for x in ms)}) | {r1['iters_mean']:.2f} [{r1['iters_min']},"
              f"{r1['iters_max']}] | {r1['div_max']:.2e} | {fr} | {capped} | {lmax:.1f}"
              f"{' NJ' if lmax > LOADMAX else ''} | {extra.strip()} |")
for f in fails:
    print(f"ROW {ROW} FAILED-RUN {f}")

if ROW == "M1" and all((ROW, L) in out for L in (4, 5, 6, 7)):
    t = {L: out[(ROW, L)]["ms"] for L in (4, 5, 6, 7)}
    it = {L: out[(ROW, L)]["iters"] for L in (4, 5, 6, 7)}
    nj = max(out[(ROW, L)]["lmax"] for L in (4, 5, 6, 7)) > LOADMAX
    a = t[5] <= 1.08 * min(t[6], t[7])
    b = t[5] <= 0.20 * t[4]
    c = it[5] <= it[6]
    v = lambda ok: ("NOT JUDGED" if nj else ("PASS" if ok else "MISS"))
    print(f"CHECKPOINT G-PERF-WS(M1, AUTO=L5): (a) t5 {t[5]:.2f} <= 1.08*min(t6 {t[6]:.2f}, t7 {t[7]:.2f})"
          f" = {1.08 * min(t[6], t[7]):.2f}: {v(a)}; (b) t5 <= 0.20*t4 {t[4]:.2f} = {0.2 * t[4]:.2f}: {v(b)};"
          f" (c) it5 {it[5]:.2f} <= it6 {it[6]:.2f}: {'PASS' if c else 'MISS'}"
          f"; L5 vs L6 {100 * (t[5] / t[6] - 1):+.1f} %")
