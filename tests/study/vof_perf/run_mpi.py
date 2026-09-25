#!/usr/bin/env python3
"""Distributed timing of the bubble column from ckpt_t43 (host MPI x OpenMP; the VoF
step-performance harness, WO-0 -- doc/vof_step_performance_design.md).

    export OMPI_MCA_hwloc_base_binding_policy=none
    PYTHONPATH=<PECLET_FLOW_MPI build> OMP_NUM_THREADS=3 OMP_PROC_BIND=false \
        mpirun -np 8 --bind-to none python tests/study/vof_perf/run_mpi.py N [--warm W] [--cheb]
            [--ckpt PATH] [--case-dir DIR]

Same case as run_peclet.build(), on this rank's ORB block; set_superficial_velocity closes the
column (a build older than it shifts u on the host: global mean via Allreduce). --ckpt and
--case-dir default as in prof.py. Prints one line on rank 0: ms/step and mean pressure iterations.
"""
import os
import sys
import time

import numpy as np
from mpi4py import MPI


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


CASE_DIR = arg("--case-dir", os.path.expanduser(
    "~/Codes/peclet-examples-bubble-column/benchmarks/bubble-column/scripts"))
CKPT = arg("--ckpt", os.path.expanduser(
    "~/Codes/peclet-examples-bubble-column/benchmarks/bubble-column/data/ckpt_t43.npz"))
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402
import peclet.flow as pf  # noqa: E402
import run_peclet as rp  # noqa: E402

N = int(sys.argv[1])
WARM = int(sys.argv[sys.argv.index("--warm") + 1]) if "--warm" in sys.argv else 5
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
S = case.CELLS_PER_D
NX, NY, NZ = case.NX, case.NY, case.NZ

step, t, blocks, vel, pres, series = rp.load_ckpt(CKPT)
org, siz = pf.mpi_block(NX, NY, NZ)
sl = tuple(slice(o, o + n) for o, n in zip(org, siz))
s = pf.Solver(*siz)
s.init_mpi(NX, NY, NZ)
s.set_rho(case.RHO_L)
s.set_mu(S * S * case.MU_L)
s.set_domain_bc("-y", "wall", (0, 0, 0))
s.set_domain_bc("+y", "wall", (0, 0, 0))
s.set_pressure_geometry(np.full(tuple(siz), 10.0, order="F"))
s.enable_vof()
s.set_vof(np.zeros(tuple(siz), order="F"))
s.set_property_model("rho", "linear", "C", [case.RHO_L, case.RHO_G - case.RHO_L])
s.set_property_model("mu", "linear", "C", [S * S * case.MU_L, S * S * (case.MU_G - case.MU_L)])
s.set_surface_tension(S ** 3 * case.SIGMA)
s.set_pressure_chebyshev(True, 800, 1e-10)
s.enable_vof_blocks_from_colors([b for b, _ in blocks], [c for _, c in blocks])
s.enable_vof_block_csf()
vol = comm.allreduce(sum(b["volume"] for b in s.diagnostics.vof_block_stats()), op=MPI.SUM)
alpha = vol / (NX * NY * NZ)
rho_av = case.RHO_L + (case.RHO_G - case.RHO_L) * alpha
s.set_property_model("force_x", "linear", "C",
                     [S * (case.RHO_L - rho_av) * case.G, S * (case.RHO_G - case.RHO_L) * case.G])
_closure = getattr(s, "set_superficial_velocity", None) or getattr(s, "set_bulk_velocity", None)
if _closure is not None:
    _closure(True, "x", 0.0)
HOST_FLUX = _closure is None  # a build older than the device constraint: shift on the host
if "--cheb" not in sys.argv:  # MG-PCG, as run_peclet.build() (933cc79); --cheb keeps Chebyshev
    s.set_pressure_pcg(True, 800, 1e-10)
for c, a in enumerate(vel):
    s.set_velocity(c, np.asfortranarray(a[sl]))
s.set_field("p", np.asfortranarray(pres[sl]))
s.diagnostics.set_vof_step_parity(step)


def one():
    L = s.vof_step_limits()
    dt = min(0.25 * L["cfl_dt"], 0.25 * L["capillary_dt"])
    s.set_dt(dt)
    s.step()
    if HOST_FLUX:
        u = s.get_u()
        mean = comm.allreduce(float(u.sum()), op=MPI.SUM) / comm.allreduce(u.size, op=MPI.SUM)
        s.set_velocity(0, np.asfortranarray(u - mean))
    return dt


for _ in range(WARM):
    one()
comm.Barrier()
w0 = time.perf_counter()
it = []
for _ in range(N):
    one()
    it.append(s.diagnostics.last_pressure_iterations())
comm.Barrier()
wall = time.perf_counter() - w0
if rank == 0:
    print(f"np {comm.Get_size()} block {list(siz)}  steps {N}  {1000 * wall / N:.2f} ms/step  "
          f"pressure iters mean {np.mean(it):.2f}  alpha {alpha:.5f}", flush=True)
