#!/usr/bin/env python3
"""Multi-GPU WEAK scaling of a periodic swarm of deformable bubbles (block VoF, momentum advection
on, the case-A groups): a fixed box of cells per rank, tiled along x, then y, then z.

    PYTHONPATH=<PECLET_FLOW_MPI build> \\
        srun --mpi=pmix --ntasks=P --gpus-per-task=1 --gpu-bind=per_task:1 \\
            python tests/study/vof_perf/weak_swarm.py N [--warm W] [--cells n] [--d-over-h D]
                [--bubbles-per-axis B] [--rtol R] [--single] [--out FILE.json] [--tag T]

The case is case A of the bubble-swarm validation (tests/study/case_a on branch case-a: Ar 29.9,
Bo 2, rho_b/rho_l = mu_b/mu_l = 0.1, buoyancy with the mixture weight removed, the volume mean u_x
held at 0 by set_superficial_velocity) with many bubbles instead of one, in CELL units as
prof_mpi.py / run_peclet.build() (D = rho_l = g = 1, lengths scaled by S = d/h: mu' = S^2 mu,
sigma' = S^3 sigma, g' = S g; test_vof_blocks_physical_units.py proves the physical domain equal
to 1e-12). Gravity along -x, triply periodic, starting at rest.

Per rank: an n^3 box (default 256^3) carrying B^3 bubbles (default 2^3 = 8) of diameter d/h cells
(default 20) on a lattice of spacing n/B, each centre perturbed by up to +-PERTURB/2 of the free
gap with the fixed seed SEED. The rank box is tiled (TX, TY, TZ) times, doubling x, then y, then z
(P = 1, 2, 4, 8, 16, 32 -> (1,1,1), (2,1,1), (2,2,1), (2,2,2), (4,2,2), (4,4,2)); every tile carries
the SAME perturbed bubble set, so every rank has the same work when the ORB blocks are the tiles.

At P = 1 the default is the DISTRIBUTED path (init_mpi at size 1, as prof_mpi.py); --single runs the
single-rank solver instead (no mpi_block/init_mpi, as run_mpi.py), the honest one-GPU reference.

W warm-up steps (default 5), then N timed steps, dt = 0.25 min(CFL, capillary) every step (case A).
Rank 0 prints one `RESULT key=value ...` line and, with --out, writes the same numbers as JSON:
ms/step (wall), max-rank step / projection / momentum / predictor / pressure-allreduce ms, pressure
iterations, cells per rank, the ORB block sizes, setup seconds, and `hash`: SHA-256 over every
rank's u, v, w, p, C (rank order) plus the timed dt and iteration sequences -- equal hashes at the
same P are the bit-identity gate across MPI transports (PECLET_CORE_GPU_AWARE_MPI=0/1).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from mpi4py import MPI

A = list(sys.argv)


def arg(name, default, cast=str):
    return cast(A[A.index(name) + 1]) if name in A else default


# --- case A groups (tests/study/case_a/run_case_a.py), units D = rho_l = g = 1 ----------------
AR, BO, RHO_RATIO, MU_RATIO = 29.9, 2.0, 0.1, 0.1
RHO_L, G, D = 1.0, 1.0, 1.0
RHO_G = RHO_RATIO * RHO_L
MU_L = np.sqrt(RHO_L * abs(RHO_G - RHO_L) * G * D ** 3) / AR
MU_G = MU_RATIO * MU_L
SIGMA = abs(RHO_G - RHO_L) * G * D ** 2 / BO
SEED, PERTURB = 20261024, 0.25

N = int(A[1])
WARM = arg("--warm", 5, int)
CELLS = arg("--cells", 256, int)
S = arg("--d-over-h", 20.0, float)
B = arg("--bubbles-per-axis", 2, int)
RTOL = arg("--rtol", 1e-8, float)
OUT = arg("--out", None)
TAG = arg("--tag", "")
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
SINGLE = "--single" in A
if SINGLE and size != 1:
    sys.exit("--single needs exactly one rank")


def tiles(p):
    """Doubling x, then y, then z: p must be a power of two."""
    if p & (p - 1):
        sys.exit(f"weak_swarm: {p} ranks is not a power of two")
    t = [1, 1, 1]
    ax = 0
    while t[0] * t[1] * t[2] < p:
        t[ax] *= 2
        ax = (ax + 1) % 3
    return t


TX, TY, TZ = tiles(size)
NX, NY, NZ = TX * CELLS, TY * CELLS, TZ * CELLS


def bubble_seeds():
    """One rank box's perturbed B^3 lattice (fixed seed), replicated over every tile; cell units."""
    rng = np.random.default_rng(SEED)
    sp = CELLS / B
    gap = sp - S
    if gap <= 0:
        sys.exit(f"weak_swarm: bubbles (d/h {S}) do not fit a lattice spacing of {sp} cells")
    box = []
    for k in range(B):
        for j in range(B):
            for i in range(B):
                c = (np.array([i, j, k]) + 0.5) * sp + (rng.random(3) - 0.5) * PERTURB * gap
                box.append(c)
    out = []
    for tz in range(TZ):
        for ty in range(TY):
            for tx in range(TX):
                o = np.array([tx, ty, tz]) * CELLS
                out += [(float(c[0] + o[0]), float(c[1] + o[1]), float(c[2] + o[2]), 0.5 * S)
                        for c in box]
    return out


t_setup0 = time.perf_counter()
import peclet.flow as pf  # after the argument checks: an import is seconds on a GPU

if SINGLE:
    org, siz = (0, 0, 0), (NX, NY, NZ)
else:
    org, siz = pf.mpi_block(NX, NY, NZ)
s = pf.Solver(*siz)
if not SINGLE:
    s.init_mpi(NX, NY, NZ)
s.set_rho(RHO_L)
s.set_mu(S * S * MU_L)
s.set_pressure_geometry(np.full(tuple(siz), 10.0, order="F"))       # all fluid (SDF > 0)
s.enable_vof()
s.set_vof(np.zeros(tuple(siz), order="F"))
# C = 1 is GAS (the block container's union starts empty)
s.set_property_model("rho", "linear", "C", [RHO_L, RHO_G - RHO_L])
s.set_property_model("mu", "linear", "C", [S * S * MU_L, S * S * (MU_G - MU_L)])
s.set_surface_tension(S ** 3 * SIGMA)
seeds = bubble_seeds()
s.enable_vof_blocks(seeds)
s.enable_vof_block_csf()
vol = comm.allreduce(sum(b["volume"] for b in s.diagnostics.vof_block_stats()), op=MPI.SUM)
alpha = vol / (NX * NY * NZ)
rho_av = RHO_L + (RHO_G - RHO_L) * alpha
# gravity along -x with the mixture weight removed: f_x = -(rho(C) - <rho>) g (case A)
s.set_property_model("force_x", "linear", "C",
                     [-S * (RHO_L - rho_av) * G, -S * (RHO_G - RHO_L) * G])
s.set_superficial_velocity(True, "x", 0.0)
s.set_pressure_pcg(True, 800, RTOL)       # last (the rho closure re-selects); case A's driver
s.update_properties()                     # so the first capillary_dt sees rho(C)
comm.Barrier()
t_setup = comm.allreduce(time.perf_counter() - t_setup0, op=MPI.MAX)

dts, iters, rows = [], [], []
KEYS = ("step", "projection", "momentum", "predictor", "pressure_allreduce")


def one(timed):
    lim = s.vof_step_limits()
    dt = 0.25 * min(lim["cfl_dt"], lim["capillary_dt"])
    s.set_dt(dt)
    t0 = time.perf_counter()
    s.step()
    t1 = time.perf_counter()
    if timed:
        tm = s.diagnostics.last_step_timers()
        dts.append(dt)
        iters.append(s.diagnostics.last_pressure_iterations())
        rows.append([t1 - t0] + [float(tm[k]) for k in KEYS])


for _ in range(WARM):
    one(False)
comm.Barrier()
if rank == 0:
    print("[weak_swarm] timed steps begin", file=sys.stderr, flush=True)
w0 = time.perf_counter()
for _ in range(N):
    one(True)
comm.Barrier()
wall = time.perf_counter() - w0

# per-step maxima over ranks (the slowest rank sets the step), then means over the timed steps
R = np.asarray(rows)
Rmax = np.empty_like(R)
comm.Allreduce(R, Rmax, op=MPI.MAX)
mean_ms = 1000.0 * Rmax.mean(axis=0)

h = hashlib.sha256()
for f in (s.get_u, s.get_v, s.get_w, lambda: s.get_field("p"), lambda: s.get_field("C")):
    h.update(np.ascontiguousarray(f()).tobytes())
digests = comm.gather(h.hexdigest(), root=0)
blocks = comm.gather("x".join(map(str, siz)), root=0)
if rank == 0:
    g = hashlib.sha256()
    for d in digests:
        g.update(d.encode())
    g.update(np.asarray(dts).tobytes())
    g.update(np.asarray(iters, dtype=np.int64).tobytes())
    ub = sorted(set(blocks))
    res = {
        "tag": TAG, "np": size, "path": "single" if SINGLE else "distributed",
        "gpu_aware_env": os.environ.get("PECLET_CORE_GPU_AWARE_MPI", "unset"),
        "flow_module": os.path.dirname(pf.__file__), "space": str(getattr(pf, "execution_space", "?")),
        "grid": [NX, NY, NZ], "tiles": [TX, TY, TZ], "cells_per_rank": CELLS ** 3,
        "blocks": ub, "d_over_h": S, "bubbles": len(seeds), "alpha": alpha, "rtol": RTOL,
        "warm": WARM, "steps": N, "setup_s": t_setup, "wall_ms": 1000.0 * wall / N,
        "step_ms": float(mean_ms[0]), "timer_step_ms": float(mean_ms[1]),
        "proj_ms": float(mean_ms[2]), "momentum_ms": float(mean_ms[3]),
        "predictor_ms": float(mean_ms[4]), "allreduce_ms": float(mean_ms[5]),
        "proj_frac": float(mean_ms[2] / mean_ms[0]),
        "iters_mean": float(np.mean(iters)), "iters_min": int(min(iters)),
        "iters_max": int(max(iters)), "dt_first": dts[0], "dt_last": dts[-1],
        "hash": g.hexdigest(),
    }
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return ",".join(map(str, v)) if isinstance(v, list) else str(v)
    print("RESULT " + " ".join(f"{k}={fmt(v)}" for k, v in res.items() if k != "flow_module"),
          flush=True)
    if OUT:
        with open(OUT, "w") as fo:
            json.dump(res, fo, indent=1)
