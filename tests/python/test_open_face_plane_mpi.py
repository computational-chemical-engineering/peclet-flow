#!/usr/bin/env python
"""Gate: the high-side face plane under MPI -- a distributed run equals the serial one (both grids).

Why this test exists. The fix that keeps the HIGH-side domain face of a MAC face field across the
ghost fills (doc/uf_outlet_fix.md, test_open_face_plane.py) acts only on the rank that owns that
global face (touchesGlobalFace); every other rank must take the plain halo exchange. Nothing else
registered covers that branch: the mpi_np2 state hash is fully periodic. Here the domain is split
along x only (asserted), so the +x face belongs to one rank and, at np = 3, the middle rank touches
neither open face.

Cases (each on SolverColocated and Solver, distributed vs a serial solver on rank 0):
  A. forward channel (-x inflow, +x outflow), developed 30 steps, then a c = 1 scalar (D = 0, Koren)
     for 3 steps: c and u equal serial to 1e-12, and max|c - 1| <= 1e-10 (a wrapped outlet plane
     moves the outlet column by O(1), on every rank count);
  B. the channel driven backwards (fluid enters through the +x outflow face): the backflow census
     max_reverse per step equals serial to 1e-12;
  C. a HIGH-side inflow (+x inflow, -x outflow), 60 steps: u equals serial to 1e-12.
The distributed-vs-serial differences measured 2026-10-22 were <= 1.8e-14 at np 1/2/3/4/6.

Run:  OMP_NUM_THREADS=2 OMP_PROC_BIND=false PYTHONPATH=<build> \
          mpirun --bind-to none -np 3 python tests/python/test_open_face_plane_mpi.py
Exit 0 pass, 1 fail, 77 skipped (no module, no MPI build, no mpi4py, or a single rank).
"""
import sys

import numpy as np

NX, NY, NZ = 48, 16, 4
MU, DT = 0.1, 0.5
TOL_SERIAL = 1e-12
TOL_C = 1e-10


def make(pf, cls, u_in, low, blk, dist):
    lx, ly, lz = blk[3:]
    s = cls(lx, ly, lz)
    if dist:
        s.init_mpi(NX, NY, NZ)
    s.set_rho(1.0); s.set_mu(MU); s.set_dt(DT)
    s.set_advection(True); s.set_advection_scheme("koren")
    if cls is pf.SolverColocated:
        s.set_collocated_scheme("gauge-exact")
    if low == "inflow":
        s.set_domain_bc("-x", "inflow", (u_in, 0.0, 0.0)); s.set_domain_bc("+x", "outflow")
    else:
        s.set_domain_bc("-x", "outflow"); s.set_domain_bc("+x", "inflow", (u_in, 0.0, 0.0))
    s.set_domain_bc("-y", "wall"); s.set_domain_bc("+y", "wall")
    s.set_pressure_geometry(np.asfortranarray(np.full((lx, ly, lz), 10.0)))
    s.set_pressure_pcg(True, 400, 1e-11)
    return s


def run_cases(pf, blk, dist):
    """The three cases on one block; returns {name: local array} (census: the per-step series)."""
    res = {}
    for cls in (pf.SolverColocated, pf.Solver):
        n = cls.__name__
        s = make(pf, cls, 1.0, "inflow", blk, dist)
        for _ in range(30):
            s.step()
        s.add_scalar("c", diffusivity=0.0, scheme="koren", iters=1)
        s.set_scalar_bc("c", "-x", "dirichlet", 1.0); s.set_scalar_bc("c", "+x", "neumann", 0.0)
        s.set_field("c", np.asfortranarray(np.ones(blk[3:])))
        for _ in range(3):
            s.step()
        res[n + "_A_c"] = np.array(s.get_field("c"), dtype=np.float64)
        res[n + "_A_u"] = np.array(s.get_u(), dtype=np.float64)
        s = make(pf, cls, -1.0, "inflow", blk, dist)
        cen = []
        for _ in range(10):
            s.step()
            cen.append(s.diagnostics.outflow_backflow()["max_reverse"])
        res[n + "_B_census"] = np.array(cen)
        s = make(pf, cls, -1.0, "outflow", blk, dist)
        for _ in range(60):
            s.step()
        res[n + "_C_u"] = np.array(s.get_u(), dtype=np.float64)
    return res


def main():
    try:
        import peclet.flow as pf
        from mpi4py import MPI
    except ImportError as e:
        print(f"SKIP: {e}")
        return 77
    if not hasattr(pf, "mpi_block"):
        print("SKIP: peclet.flow built without PECLET_FLOW_MPI")
        return 77
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        print("SKIP: run under mpirun with np >= 2")
        return 77
    (ox, oy, oz), (lx, ly, lz) = pf.mpi_block(NX, NY, NZ)
    blk = (ox, oy, oz, lx, ly, lz)
    xsplit = comm.allreduce(int((oy, oz, ly, lz) == (0, 0, NY, NZ)), op=MPI.MIN)
    owners = comm.allreduce(int(ox + lx == NX), op=MPI.SUM)

    local = run_cases(pf, blk, True)
    parts = comm.gather((blk, local), root=0)
    ok = True
    if comm.rank == 0:
        ref = run_cases(pf, (0, 0, 0, NX, NY, NZ), False)  # the serial reference, on rank 0 alone
        try:
            assert xsplit == 1 and owners == 1, (
                f"np {comm.size}: the decomposition is not an x-only split with one owner of +x "
                f"(xsplit {xsplit}, owners {owners})")
            line = [f"np {comm.size}"]
            for k in sorted(ref):
                if k.endswith("census"):
                    d = float(np.abs(parts[0][1][k] - ref[k]).max())
                else:
                    g = np.zeros_like(ref[k])
                    for (bx, by, bz, bl, bm, bn), p in parts:
                        g[bx:bx + bl, by:by + bm, bz:bz + bn] = p[k]
                    d = float(np.abs(g - ref[k]).max())
                    if k.endswith("_A_c"):
                        cdev = float(np.abs(g - 1.0).max())
                        assert cdev <= TOL_C, f"{k}: distributed c = 1 moved by {cdev:.3e}"
                line.append(f"{k} {d:.1e}")
                assert d <= TOL_SERIAL, f"{k}: distributed vs serial {d:.3e} > {TOL_SERIAL:.0e}"
            for n in ("SolverColocated", "Solver"):
                cdev = float(np.abs(ref[n + "_A_c"] - 1.0).max())
                assert cdev <= TOL_C, f"{n}_A_c: serial c = 1 moved by {cdev:.3e}"
            print("  " + "  ".join(line))
        except AssertionError as e:
            print(f"FAIL: {e}")
            ok = False
    ok = comm.bcast(ok, root=0)
    if not ok:
        return 1
    if comm.rank == 0:
        print("OK: the high-side face plane is kept on its owning rank only; distributed == serial")
    return 0


if __name__ == "__main__":
    sys.exit(main())
