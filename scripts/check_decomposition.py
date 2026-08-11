#!/usr/bin/env python
"""Inspect how a grid + rank count decomposes, and how deep the pressure multigrid gets on it.

Answers, without spending any GPU time, the questions that decide distributed multigrid performance:

  * how balanced is the ORB partition (max block / min block)?
  * is a direction split that must not be (e.g. the wall-normal one in a channel)?
  * how many multigrid levels does the hierarchy ACTUALLY reach, and where does each axis stop?
  * does the coarse-first decomposition help here, and at what depth?

Run it before committing a grid or a rank count to a large job. It needs only an MPI build of
`flow` (the OpenMP one is fine, no GPU), and `mpirun --oversubscribe` happily runs more ranks than
there are cores because nothing here is timed.

    # one grid, several rank counts, both decomposition modes:
    python check_decomposition.py --grid 480,80,160 --levels 6 --np 1,2,4,8,12,16,24 --mode 0,coarse

    # a whole weak-scaling ladder, ORB only (no solver allocation -> any size, instantly):
    python check_decomposition.py --grid 960,160,320 --np 1 --orb-only
    python check_decomposition.py --grid 3072,512,1024 --np 32 --orb-only

    # what the hierarchy looks like level by level:
    python check_decomposition.py --grid 1508,240,503 --levels 5 --np 4 --verbose

`--orb-only` skips constructing a Solver, so it reports the partition (balance, splits) for grids far
too large to allocate on a host — but it cannot report the achieved level count, which needs the real
MG init. Without it, the grid must fit in host memory (a few hundred M cells at most).

Requires PYTHONPATH to point at an MPI-enabled flow build, e.g.
    PYTHONPATH=$PWD/build_mpi_omp python scripts/check_decomposition.py ...
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.abspath(__file__)


def halvings(d):
    """How many times an axis can be halved: the multigrid's per-axis depth budget."""
    n = 0
    while d % 2 == 0 and d // 2 >= 2:
        d //= 2
        n += 1
    return n


# ---- child: runs under mpirun, prints one RESULT line ------------------------------------------
def child(args):
    from mpi4py import MPI

    w = MPI.COMM_WORLD
    import numpy as np

    from peclet import flow

    gx, gy, gz = args.grid
    origin, size = flow.mpi_block(gx, gy, gz)
    lnx, lny, lnz = size
    cells = lnx * lny * lnz
    hi = w.allreduce(cells, op=MPI.MAX)
    lo = w.allreduce(cells, op=MPI.MIN)
    split = [w.allreduce(1 if size[k] < (gx, gy, gz)[k] else 0, op=MPI.SUM) for k in range(3)]

    if not args.orb_only:
        # Constructing the Solver is what builds the MG hierarchy; with PECLET_FLOW_MG_DEBUG=1 it
        # prints the level table, which the parent parses for the achieved depth.
        s = flow.Solver(lnx, lny, lnz)
        s.init_mpi(gx, gy, gz)
        s.set_rho(1.0)
        s.set_mu(0.1)
        s.set_dt(0.02)
        s.set_pressure_multigrid(True, args.levels)
        s.set_pressure_pcg(True, 200, 1e-6)
        if args.walls:
            s.set_domain_bc(2, 1)
            s.set_domain_bc(3, 1)
        s.set_pressure_geometry(np.asfortranarray(np.full((lnx, lny, lnz), 1e30)))

    if w.rank == 0:
        print(
            f"RESULT np={w.size} block={lnx}x{lny}x{lnz} imbalance={hi / lo:.3f} "
            f"split=({split[0]},{split[1]},{split[2]})",
            flush=True,
        )
    MPI.Finalize()
    os._exit(0)


# ---- parent: spawn one mpirun per (mode, np) and tabulate ---------------------------------------
MODES = {"0": ("aligned", "0"), "aligned": ("aligned", "0"), "legacy": ("aligned", "0")}


def mode_spec(tok, levels):
    """'0'/'aligned' -> the legacy aligned ORB; 'coarse'/'coarse-first'/<int> -> coarse-first."""
    if tok in MODES:
        return MODES[tok]
    if tok in ("coarse", "coarse-first", "cf"):
        return ("coarse-first", str(levels))
    return (f"coarse-first({tok})", tok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, help="GNX,GNY,GNZ")
    ap.add_argument("--np", default="1", help="comma-separated rank counts")
    ap.add_argument("--levels", type=int, default=5, help="multigrid levels requested (default 5)")
    ap.add_argument("--mode", default="0", help="comma-separated: 0/aligned, coarse, or a depth")
    ap.add_argument("--walls", action="store_true", help="no-slip -y/+y (the channel's domain-BC MG path)")
    ap.add_argument("--orb-only", action="store_true", help="partition only; no Solver, so any size")
    ap.add_argument("--verbose", action="store_true", help="also print the per-level dims/ratios")
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.grid = [int(v) for v in args.grid.split(",")]
    if args.child:
        return child(args)

    gx, gy, gz = args.grid
    print(f"grid {gx}x{gy}x{gz} = {gx * gy * gz / 1e6:.1f} Mcells   "
          f"halvings x/y/z = {halvings(gx)}/{halvings(gy)}/{halvings(gz)}"
          f"{'   <-- an axis with 0 halvings NEVER coarsens' if 0 in (halvings(gx), halvings(gy), halvings(gz)) else ''}")
    print(f"multigrid levels requested: {args.levels}\n")
    print(f"{'mode':<20} {'np':>4} {'block':>18} {'imbalance':>10} {'split x,y,z':>12} {'levels':>7}")

    for tok in args.mode.split(","):
        label, decomp = mode_spec(tok.strip(), args.levels)
        for np_ in [int(v) for v in args.np.split(",")]:
            env = dict(os.environ, PECLET_FLOW_MG_DEBUG="1", PECLET_FLOW_DECOMP_LEVELS=decomp)
            cmd = [args.mpirun, "--oversubscribe", "-np", str(np_), sys.executable, HERE, "--child",
                   "--grid", ",".join(map(str, args.grid)), "--levels", str(args.levels)]
            if args.walls:
                cmd.append("--walls")
            if args.orb_only:
                cmd.append("--orb-only")
            try:
                out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900).stdout
            except subprocess.TimeoutExpired:
                print(f"{label:<20} {np_:>4}   TIMED OUT")
                continue
            res = re.search(r"RESULT np=(\d+) block=(\S+) imbalance=(\S+) split=\((\S+)\)", out)
            if not res:
                print(f"{label:<20} {np_:>4}   FAILED (rerun by hand for the error)")
                continue
            lev = re.search(r"-> (\d+) levels", out)
            cf = re.search(r"coarse-first depth (\d+) \(refine (\S+),", out)
            note = f"  [coarse-first depth {cf.group(1)}, refine {cf.group(2)}]" if cf else ""
            print(f"{label:<20} {res.group(1):>4} {res.group(2):>18} {res.group(3):>10} "
                  f"{res.group(4):>12} {lev.group(1) if lev else '-':>7}{note}")
            if args.verbose:
                for line in out.splitlines():
                    if line.startswith("[mg]  L"):
                        print("    " + line)
    print("\nreminders: an ODD axis never coarsens (measured 3.2x slower on one GPU);")
    print("           in MPI a level coarsens an axis only if EVERY rank's block is even on it,")
    print("           so the achievable depth is set by the per-rank block, not the global grid.")


if __name__ == "__main__":
    main()
