#!/usr/bin/env python3
"""SCALING_ISSUES #2 TRAP probe — does a WEIGHTED level-0 decomposition collapse the pressure
multigrid's telescope to one rank?

The claim (read from `src/mac_cutcell_mg.hpp`, not measured when it was written): after
`diagnostics.rebalance_by_weights(w)` the level-0 blocks are no longer even on the coarsenable
axes, the telescope fires at L = 0, and the agglomeration-depth search falls through to d = 0 —
the whole level-0 residual gathered onto ONE rank every V-cycle.

This measures it, flow-only (the coupling's `CfdDem.rebalance()` reaches the same call):

  1. build a distributed Solver on the default equal-weight aligned ORB, run a few steps;
  2. record the per-rank blocks (origin, size, parity) and the pressure iterations / step time;
  3. `rebalance_by_weights(w)` with a particle-bed-like weight (1 + Poisson counts in the bottom
     `--bed` fraction of z), record the same again.

The MG ladder (levels, ranks per level, which transitions TELESCOPE) is printed by flow itself on
rank 0 at every hierarchy build when PECLET_FLOW_MG_DEBUG=1 (a trace-only env var; it changes no
result). The ranks column of the level after a telescope point is the size of the roots'
sub-communicator, i.e. 2^d for the depth d the search selected (np a power of two).

    PECLET_FLOW_MG_DEBUG=1 OMP_NUM_THREADS=2 OMP_PROC_BIND=false \
      /usr/bin/mpirun -np 8 --map-by slot:PE=2 --bind-to core python tests/study/weighted_dec0_telescope_probe.py --n 96 --levels 4
"""
import argparse
import sys
import time

import numpy as np
from mpi4py import MPI

import peclet.flow as flow


def sphere_sdf(o, sz, n, per_axis=2, rfrac=0.35):
    """Periodic simple-cubic array of spheres (per_axis^3 of them), sampled at cell centres of this
    rank's block (cell units, origin 0). Negative inside the solid."""
    a = n / per_axis
    r = rfrac * a
    x = o[0] + np.arange(sz[0]) + 0.5
    y = o[1] + np.arange(sz[1]) + 0.5
    z = o[2] + np.arange(sz[2]) + 0.5
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    dx = (X % a) - 0.5 * a
    dy = (Y % a) - 0.5 * a
    dz = (Z % a) - 0.5 * a
    return np.asfortranarray(np.sqrt(dx * dx + dy * dy + dz * dz) - r)


def bed_weights(n, bed, lam, tilt=0.0, seed=7):
    """1 + Poisson(lam) particle counts below a bed surface, 1 above — a settled bed, as
    `CfdDem.rebalance(gamma=1)` would build it (1 + gamma * count). tilt = 0 is a FLAT bed at
    z = bed*n (uniform in x and y, so the x/y ORB splits stay at the equal-cell positions); tilt > 0
    is a HEAP whose surface falls linearly in x and y, z_s = bed*n*(1 + tilt*(1/2 - x/n) +
    tilt*(1/2 - y/n)), which moves every split. Global, x-fastest (Fortran ravel), identical on
    every rank (same seed)."""
    rng = np.random.default_rng(seed)
    c = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    zs = bed * (1.0 + tilt * (0.5 - X) + tilt * (0.5 - Y))
    counts = np.where(Z < zs, rng.poisson(lam, size=(n, n, n)), 0)
    return (1.0 + counts).ravel(order="F")


def blocks(comm, s):
    o = list(s.block_origin())
    sz = list(np.asarray(s.get_p()).shape)
    return comm.allgather((o, sz))


def describe(tag, comm, blks, w, n):
    if comm.rank != 0:
        return
    print(f"\n[probe] {tag}: per-rank blocks (origin / size / odd axes)")
    W = None if w is None else w.reshape((n, n, n), order="F")
    loads, cells = [], []
    for r, (o, sz) in enumerate(blks):
        odd = [ax for ax in range(3) if (o[ax] % 2) or (sz[ax] % 2)]
        ld = sz[0] * sz[1] * sz[2] if W is None else float(
            W[o[0]:o[0] + sz[0], o[1]:o[1] + sz[1], o[2]:o[2] + sz[2]].sum())
        loads.append(ld)
        cells.append(sz[0] * sz[1] * sz[2])
        print(f"[probe]   rank {r}: origin {o}  size {sz}  odd-axes {''.join('xyz'[a] for a in odd) or '-'}"
              f"  weight {ld:.0f}")
    loads = np.asarray(loads, dtype=float)
    cells = np.asarray(cells, dtype=float)
    print(f"[probe]   weight imbalance max/mean = {loads.max() / loads.mean():.3f}"
          f"   cell imbalance max/mean = {cells.max() / cells.mean():.3f}")
    sys.stdout.flush()


def run_steps(comm, s, k):
    """Per step: pressure iterations, wall time (barrier to barrier), and the max over ranks of the
    solver's own projection and momentum timers (`last_step_timers`)."""
    its, ts, tp, tm = [], [], [], []
    for _ in range(k):
        comm.Barrier()
        t0 = time.perf_counter()
        s.step()
        comm.Barrier()
        ts.append(time.perf_counter() - t0)
        its.append(s.diagnostics.last_pressure_iterations())
        t = s.diagnostics.last_step_timers()
        tp.append(comm.allreduce(float(t["projection"]), op=MPI.MAX))
        tm.append(comm.allreduce(float(t["momentum"]), op=MPI.MAX))
    return its, ts, tp, tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--levels", type=int, default=4, help="set_pressure_multigrid levels")
    ap.add_argument("--graph-amg", action="store_true", help="set_pressure_graph_amg(True)")
    ap.add_argument("--bottom", choices=("auto", "smoother", "agglomerated"), default="auto",
                    help="set_pressure_bottom (auto = the default)")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--bed", type=float, default=0.35, help="bed height fraction of z")
    ap.add_argument("--lam", type=float, default=4.0, help="mean particles per bed cell")
    ap.add_argument("--tilt", type=float, default=0.0, help="0 = flat bed, >0 = heap")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    comm = MPI.COMM_WORLD
    n = a.n

    o, sz = flow.mpi_block(n, n, n)
    s = flow.Solver(tuple(sz))
    s.init_mpi(n, n, n)
    s.set_pressure_multigrid(True, a.levels)
    if a.graph_amg:
        s.diagnostics.set_pressure_graph_amg(True)
    s.set_pressure_bottom(a.bottom)
    s.set_body_force((1e-4, 0.0, 0.0))
    s.set_dt(1.0)
    s.set_solid(sphere_sdf(o, sz, n), True)
    if comm.rank == 0:
        print(f"[probe] n={n}^3 np={comm.size} levels={a.levels} graph_amg={a.graph_amg} "
              f"bottom={a.bottom} telescope={s.pressure_telescope()}")
        sys.stdout.flush()

    describe("BEFORE (equal-weight aligned ORB)", comm, blocks(comm, s), None, n)
    run_steps(comm, s, a.warm)
    its0, ts0, tp0, tm0 = run_steps(comm, s, a.steps)

    w = bed_weights(n, a.bed, a.lam, a.tilt, a.seed)
    if comm.rank == 0:
        print("\n[probe] ---- rebalance_by_weights ----")
        sys.stdout.flush()
    comm.Barrier()
    t0 = time.perf_counter()
    s.diagnostics.rebalance_by_weights(w.tolist())
    comm.Barrier()
    trb = time.perf_counter() - t0
    describe("AFTER (weighted ORB)", comm, blocks(comm, s), w, n)
    its1, ts1, tp1, tm1 = run_steps(comm, s, a.steps)

    if comm.rank == 0:
        print(f"\n[probe] rebalance took {trb:.2f} s")
        print(f"[probe] BEFORE pressure iters/step {its0}  step s {[round(t, 3) for t in ts0]}"
              f"  median {np.median(ts0):.3f}  projection(max rank) {np.median(tp0):.3f}"
              f"  momentum(max rank) {np.median(tm0):.3f}")
        print(f"[probe] AFTER  pressure iters/step {its1}  step s {[round(t, 3) for t in ts1]}"
              f"  median {np.median(ts1):.3f}  projection(max rank) {np.median(tp1):.3f}"
              f"  momentum(max rank) {np.median(tm1):.3f}")
        print(f"[probe] SUMMARY n={n} np={comm.size} levels={a.levels} graph_amg={a.graph_amg}"
              f" bottom={a.bottom} tilt={a.tilt} seed={a.seed}"
              f" iters {np.mean(its0):.1f} -> {np.mean(its1):.1f}"
              f" step {np.median(ts0):.3f} -> {np.median(ts1):.3f} s"
              f" projection {np.median(tp0):.3f} -> {np.median(tp1):.3f} s"
              f" momentum {np.median(tm0):.3f} -> {np.median(tm1):.3f} s")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
