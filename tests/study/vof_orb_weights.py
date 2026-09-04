#!/usr/bin/env python3
"""WO-V9 item 4 — the INTERFACE-WEIGHTED ORB.

A two-phase step's cost is not uniform over the grid: the PLIC reconstruction, the geometric
fluxes and the whole curvature cascade are paid only where the colour is mixed, while the pressure
and momentum solves are paid everywhere.  A plain (cell-count) ORB therefore hands one rank the
whole interface and another rank none of it.  This measures the imbalance that costs, and what
re-weighting the ORB by

        w(cell) = 1  +  W * [the cell is interfacial]

through the existing `rebalance_by_weights` buys.  `W` is the ratio of interfacial to bulk work
per cell; it is swept.

Scene: a 64-marker bubble swarm (the WO-W12 swarm of `tests/study/vof_blocks_ns.py`'s `swarm`
gate, here as a single global colour field so the GRID decomposition is what is being measured),
surface tension on, ratio 100.

    mpirun -np 4 python tests/study/vof_orb_weights.py --steps 20 --w 8
"""
import argparse
import math
import os
import sys
import time

import numpy as np

import peclet.flow as flow

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vof_profile as vp  # noqa: E402

EPS = 1e-8


def swarm_colour(n, seeds, sub=4):
    """C = 1 inside the markers (the dispersed phase), sub^3-subsampled fractions."""
    q = (np.arange(sub) + 0.5) / sub
    C = np.zeros((n, n, n))
    for (cx, cy, cz, r) in seeds:
        i0, i1 = max(0, int(cx - r - 2)), min(n, int(cx + r + 3))
        j0, j1 = max(0, int(cy - r - 2)), min(n, int(cy + r + 3))
        k0, k1 = max(0, int(cz - r - 2)), min(n, int(cz + r + 3))
        px = (np.arange(i0, i1)[:, None] + q[None, :]).ravel()
        py = (np.arange(j0, j1)[:, None] + q[None, :]).ravel()
        pz = (np.arange(k0, k1)[:, None] + q[None, :]).ravel()
        d = ((px[:, None, None] - cx) ** 2 + (py[None, :, None] - cy) ** 2
             + (pz[None, None, :] - cz) ** 2) < r * r
        blk = d.reshape(i1 - i0, sub, j1 - j0, sub, k1 - k0, sub).mean(axis=(1, 3, 5))
        C[i0:i1, j0:j1, k0:k1] = np.maximum(C[i0:i1, j0:j1, k0:k1], blk)
    return np.asfortranarray(C)


def seeds_64(n, layout="lattice"):
    """`lattice` is WO-W12's swarm -- 4x4x4, uniformly filling the box.  It is exactly the
    configuration a CELL-COUNT ORB already balances perfectly, so it cannot show the lever; it is
    kept as the control.  `plume` is the configuration the lever is for: the same 64 markers
    packed into the bottom third of the column, i.e. a rising swarm before it has spread."""
    out, k = [], 0
    if layout == "lattice":
        for i in range(4):
            for j in range(4):
                for m in range(4):
                    r = 2.0 + 3.5 * ((k % 8) / 7.0)
                    out.append(((i + 0.5) * n / 4, (j + 0.5) * n / 4, (m + 0.5) * n / 4, r))
                    k += 1
        return out
    rng = np.random.default_rng(3)
    tries = 0
    while len(out) < 64 and tries < 200000:
        tries += 1
        r = 2.0 + 2.0 * ((len(out) % 8) / 7.0)
        p = (rng.uniform(r + 1, n - r - 1), rng.uniform(r + 1, n - r - 1),
             rng.uniform(r + 1, 0.5 * n - r - 1))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
               > (r + q[3] + 1.5) ** 2 for q in out):
            out.append((p[0], p[1], p[2], r))
    return out


def build(n, Cglob, ratio=100.0):
    o, sz = flow.mpi_block(n, n, n)
    s = flow.Solver(*sz)
    rho_g, rho_l = 1.0, ratio          # C = 1 is the DISPERSED phase here
    s.set_rho(rho_l)
    s.set_mu(0.05)
    s.init_mpi(n, n, n)
    s.set_pressure_geometry(np.full(tuple(sz), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 600, 1e-12)
    s.enable_vof()
    s.set_vof(np.asfortranarray(
        Cglob[o[0]:o[0] + sz[0], o[1]:o[1] + sz[1], o[2]:o[2] + sz[2]]))
    s.set_property_model("rho", "linear", "C", [rho_l, rho_g - rho_l])
    s.set_property_model("mu", "linear", "C", [0.05, 0.05 / ratio - 0.05])
    s.set_surface_tension(1.0)
    s.set_property_model("force_z", "linear", "C", [-1e-4 * rho_l, -1e-4 * (rho_g - rho_l)])
    s.set_vof_curvature_worklist(os.environ.get("V9_CURV_WL", "1") == "1")
    s.set_vof_worklist(os.environ.get("V9_ADV_WL", "1") == "1")
    s.set_dt(0.4 * s.capillary_dt())
    return s, o, sz


def mixed_count(s):
    C = np.asarray(s.get_vof())
    return int(((C > EPS) & (C < 1.0 - EPS)).sum())


def gather_ints(s, v):
    """max/mean over ranks of an integer per-rank load, using the solver's own allreduce-free
    route: mpi4py if present, else a plain allgather through it."""
    from mpi4py import MPI
    return MPI.COMM_WORLD.allgather(int(v))


def timed(s, steps, cap):
    pick = vp.dt_from_limits(0.4)
    return vp.profile(s, steps, pick, cap, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--w", type=float, default=8.0, help="interfacial work multiplier")
    ap.add_argument("--layout", choices=("lattice", "plume"), default="plume")
    a = ap.parse_args()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    n = a.n
    Cglob = swarm_colour(n, seeds_64(n, a.layout))
    s, o, sz = build(n, Cglob)
    pick = vp.dt_from_limits(0.4)
    for i in range(a.warm):
        pick(s, i)
        s.step()

    def report(tag):
        loads = gather_ints(s, mixed_count(s))
        cells = gather_ints(s, int(np.prod(np.asarray(s.get_vof()).shape)))
        tm = timed(s, a.steps, 600)
        if rank == 0:
            mx, mean = max(loads), sum(loads) / len(loads)
            cmx, cmean = max(cells), sum(cells) / len(cells)
            print(f"\n--- {tag} (np = {size}, {n}^3, {a.layout}) ---")
            print(f"  mixed cells per rank : {loads}")
            print(f"  imbalance (mixed)    : {mx/mean:.4f}   (max {mx}, mean {mean:.1f})")
            print(f"  cells per rank       : {cells}")
            print(f"  imbalance (cells)    : {cmx/cmean:.4f}")
            print(f"  ms/step              : {tm['ms_per_step']:.3f}   "
                  f"pressure {tm['press_max']}/600"
                  + ("  CAPPED" if tm["capped"] else ""))
            k = max(tm["steps"], 1)
            print(f"  VoF stage            : {1e3*(tm['vof_advect']+tm['curvature']+tm['csf'])/k:.3f}"
                  f" ms/step   projection {1e3*tm['projection']/k:.3f}  "
                  f"momentum {1e3*tm['momentum_solve']/k:.3f}")
        return tm

    t_before = report("plain ORB (cell-count weights)")

    # The weight field: the CURRENT interface, gathered from every rank's own colour so the weights
    # describe the state being rebalanced (not the initial condition).
    Cl = np.asarray(s.get_vof())
    mixed = ((Cl > EPS) & (Cl < 1.0 - EPS)).astype(np.float64)
    Wglob = np.zeros((n, n, n), order="F")
    Wl = np.zeros((n, n, n), order="F")
    Wl[o[0]:o[0] + sz[0], o[1]:o[1] + sz[1], o[2]:o[2] + sz[2]] = mixed
    comm.Allreduce(Wl, Wglob, op=MPI.SUM)
    w = (1.0 + a.w * Wglob).ravel(order="F")
    if rank == 0:
        print(f"\n  weights: 1 + {a.w} * [mixed];  {int(Wglob.sum())} interfacial cells of "
              f"{n**3} ({100*Wglob.sum()/n**3:.2f} %)")
    s.rebalance_by_weights(list(w))
    t_after = report(f"interface-weighted ORB (W = {a.w})")
    if rank == 0:
        b, af = t_before["ms_per_step"], t_after["ms_per_step"]
        print(f"\n  ms/step {b:.3f} -> {af:.3f}   ({100*(af/b-1):+.2f} %)")


if __name__ == "__main__":
    main()
