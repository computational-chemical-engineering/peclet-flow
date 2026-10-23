#!/usr/bin/env python3
"""The bubble-column step from ckpt_t43 on the DISTRIBUTED solver (init_mpi, any np): prof.py's case
and protocol on this rank's ORB block, for the np-gates of doc/vof_projection_distributed_design.md
§8 (G-DD3: 'auto' / forced 'fp32' against 'fp64' at the same np, G-NUM-P 1-3; G-DD4: projection
ms/step).

The case is run_peclet.build()'s, block by block (as ~/Codes/bubble_column_perf/run_mpi.py): the
global arrays of the checkpoint are sliced to the rank's block, the VoF markers are passed as the
global list, the column is closed by the device set_superficial_velocity (rank-aware). W warm-up
steps (not timed, default 5), then N timed steps with prof.py's dt rule. Unlike prof.py the flux
closure is always the device one (prof.py's default host shift is single-rank).

Usage:
    PYTHONPATH=<PECLET_FLOW_MPI build> OMP_NUM_THREADS=2 OMP_PROC_BIND=false \\
        mpirun -np P --bind-to none python tests/study/vof_perf/prof_mpi.py N [--pcg|--fcg]
            [--rtol R] [--vcycle-precision auto|fp64|fp32] [--initial-guess MODE]
            [--warm W] [--dump out.npz] [--ckpt PATH] [--case-dir DIR] [--tile TX[,TZ]]

    --tile  periodic replication of the checkpoint TX times along x and TZ times along z (both
            periodic; the -y/+y walls are kept): global grid (TX NX, NY, TZ NZ), u/v/w/p tiled, each
            VoF block marker copied once per tile with its box shifted by whole periods (a weak-
            scaling column at constant cells per rank; default 1,1 = the checkpoint itself)

    --dump  rank 0 saves the GATHERED u, v, w, p, C, the timed steps' dt, pressure iterations
            (iters), max_open_divergence_projected() (div) and projection ms (proj_ms, the max
            over ranks of diagnostics.last_step_timers()['projection'])
Prints one line on rank 0: np, block, ms/step, projection ms/step (mean and min), iterations, the
precision the last solve used; then one machine-readable `RESULT key=value ...` line with the same
numbers, the global grid and the requested --vcycle-precision. Rank 0 writes `[prof_mpi] timed steps
begin` to stderr before the first timed step, so restarts in the timed window are the warning lines
after it. An 'auto' FP64 restart prints its own warning line on rank 0's
stderr (no Python counter exists, design Q-4).
"""
import os
import sys
import time

import numpy as np
from mpi4py import MPI

A = list(sys.argv)


def arg(name, default, cast=str):
    return cast(A[A.index(name) + 1]) if name in A else default


CASE_DIR = arg("--case-dir", os.path.expanduser(
    "~/Codes/peclet-examples/benchmarks/bubble-column/scripts"))
CKPT = arg("--ckpt", os.path.expanduser("~/Codes/bubble_column_perf/ckpt_t43.npz"))
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402
import peclet.flow as pf  # noqa: E402
import run_peclet as rp  # noqa: E402

N = int(A[1])
WARM = arg("--warm", 5, int)
RTOL = arg("--rtol", 1e-10, float)
DUMP = arg("--dump", None)
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
S = case.CELLS_PER_D
_tile = [int(x) for x in arg("--tile", "1,1").split(",")]
TX, TZ = _tile[0], (_tile[1] if len(_tile) > 1 else 1)
NX, NY, NZ = TX * case.NX, case.NY, TZ * case.NZ

step, t, blocks, vel, pres, series = rp.load_ckpt(CKPT)
if TX * TZ > 1:
    vel = [np.asfortranarray(np.tile(a, (TX, 1, TZ))) for a in vel]
    pres = np.asfortranarray(np.tile(pres, (TX, 1, TZ)))
    tiled = []
    for box, col in blocks:  # box = (lo_x, lo_y, lo_z, hi_x, hi_y, hi_z), x/z unwrapped
        sx = -(box[0] // case.NX) * case.NX  # whole original periods: lo_x into [0, NX)
        sz = -(box[2] // case.NZ) * case.NZ
        for i in range(TX):
            for k in range(TZ):
                ox, oz = sx + i * case.NX, sz + k * case.NZ
                tiled.append(([box[0] + ox, box[1], box[2] + oz, box[3] + ox, box[4], box[5] + oz],
                              col))
    blocks = tiled
org, siz = pf.mpi_block(NX, NY, NZ)
sl = tuple(slice(o, o + n) for o, n in zip(org, siz))
s = pf.Solver(*siz)
s.init_mpi(NX, NY, NZ)
s.set_rho(case.RHO_L)
s.set_mu(S * S * case.MU_L)
s.set_advection(False)  # the pre-flip default (Stokes) kept; NS physics: CHANGELOG "advection default"
s.set_domain_bc("-y", "wall", (0, 0, 0))
s.set_domain_bc("+y", "wall", (0, 0, 0))
s.set_pressure_geometry(np.full(tuple(siz), 10.0, order="F"))
s.enable_vof()
s.set_vof(np.zeros(tuple(siz), order="F"))
s.set_property_model("rho", "linear", "C", [case.RHO_L, case.RHO_G - case.RHO_L])
s.set_property_model("mu", "linear", "C", [S * S * case.MU_L, S * S * (case.MU_G - case.MU_L)])
s.set_surface_tension(S ** 3 * case.SIGMA)
s.set_pressure_pcg(True, 800, 1e-8)  # run_peclet.build()'s; the driver flags below override it
s.enable_vof_blocks_from_colors([b for b, _ in blocks], [c for _, c in blocks])
s.enable_vof_block_csf()
vol = comm.allreduce(sum(b["volume"] for b in s.diagnostics.vof_block_stats()), op=MPI.SUM)
alpha = vol / (NX * NY * NZ)
rho_av = case.RHO_L + (case.RHO_G - case.RHO_L) * alpha
s.set_property_model("force_x", "linear", "C",
                     [S * (case.RHO_L - rho_av) * case.G, S * (case.RHO_G - case.RHO_L) * case.G])
s.set_superficial_velocity(True, "x", 0.0)
if "--pcg" in A:
    s.set_pressure_pcg(True, 800, RTOL)
if "--fcg" in A:
    s.set_pressure_fcg(True, 800, RTOL)
if "--vcycle-precision" in A:
    s.diagnostics.set_pressure_vcycle_precision(arg("--vcycle-precision", "fp64"))
if "--initial-guess" in A:
    s.diagnostics.set_pressure_initial_guess(arg("--initial-guess", "zero"),
                                             arg("--guess-history", 2, int))
for c, a in enumerate(vel):
    s.set_velocity(c, np.asfortranarray(a[sl]))
s.set_field("p", np.asfortranarray(pres[sl]))
s.diagnostics.set_vof_step_parity(step)

iters, divs, dts, projs, steps = [], [], [], [], []


def one(timed):
    L = s.vof_step_limits()
    dt = min(0.25 * L["cfl_dt"], 0.25 * L["capillary_dt"])
    s.set_dt(dt)
    t0 = time.perf_counter()
    s.step()
    t1 = time.perf_counter()
    if timed:
        dts.append(dt)
        iters.append(s.diagnostics.last_pressure_iterations())
        divs.append(comm.allreduce(float(s.max_open_divergence_projected()), op=MPI.MAX))
        projs.append(comm.allreduce(float(s.diagnostics.last_step_timers()["projection"]),
                                    op=MPI.MAX))
        steps.append(comm.allreduce(t1 - t0, op=MPI.MAX))


for _ in range(WARM):
    one(False)
comm.Barrier()
if rank == 0:
    print("[prof_mpi] timed steps begin", file=sys.stderr, flush=True)
w0 = time.perf_counter()
for _ in range(N):
    one(True)
comm.Barrier()
wall = time.perf_counter() - w0


def gather(a):
    parts = comm.gather((org, siz, np.asarray(a)), root=0)
    if rank != 0:
        return None
    g = np.zeros((NX, NY, NZ), order="F")
    for o, n, b in parts:
        g[o[0]:o[0] + n[0], o[1]:o[1] + n[1], o[2]:o[2] + n[2]] = b
    return g


fields = {k: gather(f()) for k, f in (("u", s.get_u), ("v", s.get_v), ("w", s.get_w),
                                      ("p", lambda: s.get_field("p")),
                                      ("C", lambda: s.get_field("C")))}
prec = s.diagnostics.pressure_vcycle_precision()
if rank == 0:
    pm = 1000 * np.asarray(projs)
    print(f"np {size} block {list(siz)} steps {N} wall {1000 * wall / N:.2f} ms/step "
          f"(step max-rank mean {1000 * np.mean(steps):.2f}) projection mean {pm.mean():.2f} "
          f"min {pm.min():.2f} ms/step | pressure iters mean {np.mean(iters):.2f} min "
          f"{min(iters)} max {max(iters)} | last precision {prec} | load "
          f"{os.getloadavg()[0]:.1f}", flush=True)
    print(f"RESULT np={size} grid={NX}x{NY}x{NZ} block={'x'.join(map(str, siz))} "
          f"mode={arg('--vcycle-precision', 'default')} precision={prec} steps={N} "
          f"proj_ms={pm.mean():.4f} proj_min_ms={pm.min():.4f} iters={np.mean(iters):.3f} "
          f"step_ms={1000 * np.mean(steps):.4f} wall_ms={1000 * wall / N:.4f}", flush=True)
    if DUMP:
        np.savez(DUMP, **fields, dts=np.array(dts), iters=np.array(iters, dtype=np.int64),
                 div=np.array(divs), proj_ms=pm)
        print("dumped", DUMP, flush=True)
