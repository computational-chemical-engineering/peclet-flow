#!/usr/bin/env python
"""Distributed (MPI) flow validation driver — mirrors tests/kokkos_mpi/test_sdflow_mpi.cpp in Python.

Builds a GLOBAL 2x2x2 periodic sphere-packing SDF on every rank, uses peclet.flow.mpi_block(N,N,N) to
get this rank's ORB block, slices the local SDF, constructs Solver(*size), init_mpi(N,N,N), configures
rho/mu/dt/body-force, sets the local solid, runs STEPS steps, and reduces the global velocity sum via
mpi4py. The reduced permeability k = mu*<u>/F must be the SAME across np=1,2,4 (the distributed solve is
bit-exact to single-rank, to the MG-PCG reduction-order floor).

Run:
  PYTHONPATH=$PWD/build_mpi mpirun -np 1 ./.venv/bin/python scripts/verify_mpi_spheres_sdflow.py
  PYTHONPATH=$PWD/build_mpi mpirun -np 2 ./.venv/bin/python scripts/verify_mpi_spheres_sdflow.py
  PYTHONPATH=$PWD/build_mpi mpirun -np 4 ./.venv/bin/python scripts/verify_mpi_spheres_sdflow.py
"""
import numpy as np
from mpi4py import MPI
import peclet.flow

N, STEPS = 32, 120
RHO, MU, F, DT = 1.0, 0.1, 1e-3, 60.0


def packing_sdf(rfrac=0.18):
    """Global 2x2x2 sphere-packing SDF as an (N,N,N) Fortran-order array (negative inside, periodic)."""
    R = rfrac * N
    cs = np.array([0.25 * N, 0.75 * N])
    xs = np.arange(N)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")  # (N,N,N) x-fastest via order F below
    best = np.full((N, N, N), 1e30)
    for sx in cs:
        for sy in cs:
            for sz in cs:
                dx = X - sx; dx -= N * np.round(dx / N)
                dy = Y - sy; dy -= N * np.round(dy / N)
                dz = Z - sz; dz -= N * np.round(dz / N)
                best = np.minimum(best, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return np.asfortranarray(best)


def configure(s):
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(DT); s.set_body_force(F, 0.0, 0.0)
    s.set_advection(False)
    s.set_velocity_solver_params(80)
    s.set_pressure_multigrid(True, 4)
    s.set_pressure_pcg(True, 200, 1e-9)


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    gsdf = packing_sdf()  # (N,N,N) F-order on every rank
    gcells = float(N * N * N)

    origin, bsize = peclet.flow.mpi_block(N, N, N)
    ox, oy, oz = origin
    lnx, lny, lnz = bsize

    # Slice this rank's LOCAL inner block from the global SDF (x-fastest / F-order indexing [x,y,z]).
    lsdf = np.asfortranarray(gsdf[ox:ox + lnx, oy:oy + lny, oz:oz + lnz])

    s = peclet.flow.Solver(lnx, lny, lnz)
    s.init_mpi(N, N, N)
    configure(s)
    s.set_solid(lsdf, cutcell_pressure=True)
    for _ in range(STEPS):
        s.step()

    lsum = float(np.sum(s.get_u()))
    gsum = comm.allreduce(lsum, op=MPI.SUM)
    k = MU * (gsum / gcells) / F
    div = s.max_open_divergence()

    if rank == 0:
        print(f"np={size}: rank={s.rank()} size={s.size()}  k={k:.12e}  div={div:.3e}  "
              f"block(rank0)=origin{origin} size{bsize}")
    return k, div


if __name__ == "__main__":
    main()
