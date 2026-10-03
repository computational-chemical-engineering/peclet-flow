#!/usr/bin/env python3
"""Profile the bubble-column step from a checkpoint (the VoF step-performance harness, WO-0).

The case is the bubble column of peclet-examples (128 x 96 x 64, 16 VoF block markers, rho/mu
ratio 0.02, set_superficial_velocity), built by that repo's run_peclet.build() and restarted from
ckpt_t43.npz. It runs W warm-up steps (not timed), then N timed steps with the driver's per-step
protocol (dt = 0.25 min(cfl_dt, capillary_dt) from vof_step_limits()). See
doc/vof_step_performance_design.md (sec. 7 WO-0, sec. 8 G-BIT / G-NUM / G-PERF).

Usage:
    PYTHONPATH=<flow build> OMP_NUM_THREADS=8 OMP_PROC_BIND=false \\
        python tests/study/vof_perf/prof.py N [--pcg [--rtol R]] [--timing] [--dump out.npz]
            [--warm W] [--flux device|python|none] [--levels L] [--bottom MODE] [--fixdt DT]
            [--ckpt PATH] [--case-dir DIR] [--dummy N]

    --ckpt      checkpoint (default ~/Codes/bubble_column_perf/ckpt_t43.npz)
    --case-dir  directory holding run_peclet.py + case.py (default:
                ~/Codes/peclet-examples/benchmarks/bubble-column/scripts)
    --pcg       MG-PCG pressure with cap 800 and relative tolerance --rtol (default 1e-10)
    --dump      save u, v, w, p, C, the timed steps' dt and pressure iterations (iters) and every
                VoF block's colour array (col<id>) -- compare two dumps with cmp.py (G-BIT item 2)
    --timing    per-stage breakdown from diagnostics.vof_timing()
    --levels L  pressure multigrid depth; --bottom MODE = set_pressure_bottom(MODE) (E1)
    --bottom-solver ENGINE  diagnostics.set_pressure_bottom_solver(ENGINE): auto | direct |
                algebraic (§13: the A/B of the two bottom engines on one build)
    --div       with --dump: also store each timed step's max_open_divergence_projected() (div),
                read after the step (G-NUM item 3)
"""
import os
import sys
import time

import numpy as np

A = list(sys.argv)


def arg(name, default, cast=str):
    return cast(A[A.index(name) + 1]) if name in A else default


CASE_DIR = arg("--case-dir", os.path.expanduser(
    "~/Codes/peclet-examples/benchmarks/bubble-column/scripts"))
CKPT = arg("--ckpt", os.path.expanduser(
    "~/Codes/bubble_column_perf/ckpt_t43.npz"))
sys.path.insert(0, CASE_DIR)
import run_peclet as rp  # noqa: E402

N = int(A[1])
TIMING = "--timing" in A
FLUX = arg("--flux", "python")
DUMP = arg("--dump", None)
WARM = arg("--warm", 5, int)
FIXDT = arg("--fixdt", None, float)
RTOL = arg("--rtol", 1e-10, float)

step, t, blocks, vel, pres, series = rp.load_ckpt(CKPT)
t_b = time.time()
if "--dummy" in A:
    import peclet.flow as _pf
    _dummy = _pf.Solver(arg("--dummy", 17, int), 13, 11)
    _dummy.set_pressure_geometry(np.full((arg("--dummy", 17, int), 13, 11), 10.0, order="F"))
if "--levels" in A:
    _L = arg("--levels", 4, int)
    _orig = rp.pf.Solver

    class _Shim:
        def __new__(cls, *a, **k):
            o = _orig(*a, **k)
            o.set_pressure_multigrid(True, _L)
            return o
    rp.pf.Solver = _Shim
if not hasattr(rp.pf.Solver, "set_superficial_velocity") and hasattr(rp.pf.Solver, "set_bulk_velocity"):
    rp.pf.Solver.set_superficial_velocity = rp.pf.Solver.set_bulk_velocity  # pre-rename build
s = rp.build(blocks)
for c, a in enumerate(vel):
    s.set_velocity(c, a)
s.set_field("p", pres)
s.diagnostics.set_vof_step_parity(step)
# run_peclet.build() now enables the device constraint itself; the other modes switch it off
for _name in ("set_superficial_velocity", "set_bulk_velocity"):
    if hasattr(s, _name):
        getattr(s, _name)(FLUX == "device", "x", 0.0)
        break
if "--pcg" in A:
    s.set_pressure_pcg(True, 800, RTOL)
if "--bottom" in A:
    s.set_pressure_bottom(arg("--bottom", "auto"))
if "--bottom-solver" in A:
    s.diagnostics.set_pressure_bottom_solver(arg("--bottom-solver", "auto"))
DIV = "--div" in A
divs = []
print(f"build {time.time()-t_b:.2f} s", flush=True)

S = rp.S
dtsafe = 0.25
iters = []
T = {"limits": 0.0, "step": 0.0, "flux": 0.0}


def one(timed):
    global t
    t0 = time.perf_counter()
    if FIXDT:
        dt = FIXDT
    else:
        L = s.vof_step_limits()
        dt = min(dtsafe * L["cfl_dt"], dtsafe * L["capillary_dt"])
    s.set_dt(dt)
    t1 = time.perf_counter()
    s.step()
    t2 = time.perf_counter()
    if FLUX == "python":
        u = s.get_u()
        s.set_velocity(0, np.asfortranarray(u - float(u.mean())))
    t3 = time.perf_counter()
    if timed:
        T["limits"] += t1 - t0
        T["step"] += t2 - t1
        T["flux"] += t3 - t2
        iters.append(s.diagnostics.last_pressure_iterations())
        if DIV:
            divs.append(s.max_open_divergence_projected())
    t += dt
    return dt


for _ in range(WARM):
    one(False)
if TIMING:
    s.diagnostics.set_vof_timing(True)
w0 = time.perf_counter()
dts = [one(True) for _ in range(N)]
wall = time.perf_counter() - w0
print(f"steps {N}  wall {wall:.3f} s  {1000*wall/N:.2f} ms/step  dt mean {np.mean(dts):.3e}")
print("per-step ms: " + "  ".join(f"{k} {1000*v/N:.2f}" for k, v in T.items()))
print(f"pressure iters mean {np.mean(iters):.2f} min {min(iters)} max {max(iters)}")
if TIMING:
    v = s.diagnostics.vof_timing()
    n = v["steps"]
    print("vof_timing per step ms (steps=%d):" % n)
    for k, x in v.items():
        if k in ("steps", "k_sweeps", "kc_calls"):
            continue
        if x:
            print(f"   {k:22s} {1000*x/n:8.3f}")
if DUMP:
    cols = {f"col{b['id']}": np.asarray(s.vof_block_color(b["id"]))
            for b in s.diagnostics.vof_block_stats()}
    np.savez(DUMP, u=s.get_u(), v=s.get_v(), w=s.get_w(), p=s.get_field("p"), C=s.get_field("C"),
             dts=np.array(dts), iters=np.array(iters, dtype=np.int64), **cols,
             **({"div": np.array(divs)} if DIV else {}))
    print("dumped", DUMP, f"({len(cols)} block colours)")
