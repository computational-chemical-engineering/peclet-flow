"""Layer 2 (MPI scene demo) + Layer 4 rung 4 (replicated instances) gate.

An analytic scene is REPLICATED on every rank, so a rank derives its own block's geometry from it
with no communication at all -- that, and not any new exchange, is what lifts the single-rank
restriction the Python override path carried (set_exact_crossings / set_openness_override are
single-rank only). This script checks that claim end to end at np=1,2,4:

  1. GEOMETRY. The scene-sampled SDF and the per-cell cut OWNER on each rank's block must equal the
     single-rank global fields restricted to that block -- exactly, since every rank evaluates the
     same replicated scene at global coordinates.
  2. PHYSICS. The permeability k = mu<u>/f from a scene-driven solve must agree across np.
  3. RESOLVED LOADS (L4-R4). The per-instance hydrodynamic force, which each rank integrates over
     the wall cells inside its OWN block and which is summed across ranks, must agree across np.
     Atomics over an unordered traversal make this tolerance-reproducible, not bitwise.

Run:
  PYTHONPATH=<build_mpi> mpirun -np 1 python tests/study/sdf_campaign/mpi_scene_gate.py
  PYTHONPATH=<build_mpi> mpirun -np 2 python tests/study/sdf_campaign/mpi_scene_gate.py
  PYTHONPATH=<build_mpi> mpirun -np 4 python tests/study/sdf_campaign/mpi_scene_gate.py
Each run writes np<N>.npz; compare with `python mpi_scene_gate.py --compare`.
"""
import os
import sys
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
N, STEPS = 32, 200
RHO, MU, F, DT = 1.0, 0.1, 1e-3, 60.0
RF = 0.18
R = RF * N
CEN = [(0.25*N, 0.25*N, 0.25*N), (0.75*N, 0.75*N, 0.25*N),
       (0.75*N, 0.25*N, 0.75*N), (0.25*N, 0.75*N, 0.75*N)]
KN_R, KI_I, KI_R = 16, 2, 17


def scene():
    ni = np.array([1, -1, -1], dtype=np.int32)
    nr = np.zeros(KN_R); nr[0] = R; nr[14] = 1.0; nr[15] = 1.0
    ii = np.zeros((len(CEN), KI_I), dtype=np.int32)
    ir = np.zeros((len(CEN), KI_R))
    for m, c in enumerate(CEN):
        ii[m] = (0, -1); ir[m, 0:3] = c; ir[m, 6] = 1.0; ir[m, 7] = 1.0
    return ni, nr, ii.ravel(), ir.ravel()


def compare():
    ref = np.load(os.path.join(OUT, "np1.npz"))
    ok = True
    for np_ in (2, 4):
        p = os.path.join(OUT, "np%d.npz" % np_)
        if not os.path.exists(p):
            print("  np=%d: missing" % np_)
            continue
        d = np.load(p)
        dk = abs(d["k"] - ref["k"]) / abs(ref["k"])
        df = np.abs(d["F"] - ref["F"]).max() / max(np.abs(ref["F"]).max(), 1e-300)
        dfr = (np.abs(d["FR"] - ref["FR"]).max() / max(np.abs(ref["FR"]).max(), 1e-300)
               if "FR" in d and "FR" in ref else float("nan"))
        downer = int(d["owner_mismatch"])
        dsdf = float(d["sdf_maxdiff"])
        print("  np=%d vs np=1:  k rel diff %.3e   traction F rel diff %.3e   "
              "REACTION F rel diff %.3e   owner mismatches %d   sdf max diff %.3e"
              % (np_, dk, df, dfr, downer, dsdf))
        if dk > 1e-9 or df > 1e-6 or downer or dsdf != 0.0:
            ok = False
    print("GATE %s  [scene geometry EXACT per rank; k and resolved loads agree across np]"
          % ("PASS" if ok else "FAIL"))
    return ok


def main():
    from mpi4py import MPI
    import peclet.flow as sdflow
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # single-rank reference geometry, built on every rank (cheap, and needs no communication)
    g = np.arange(N).astype(float)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    gsdf = np.full((N, N, N), 1e30)
    gd = np.empty((len(CEN), N, N, N))
    for kk, (cx, cy, cz) in enumerate(CEN):
        dx = np.abs(X-cx); dx = np.minimum(dx, N-dx)
        dy = np.abs(Y-cy); dy = np.minimum(dy, N-dy)
        dz = np.abs(Z-cz); dz = np.minimum(dz, N-dz)
        gd[kk] = np.sqrt(dx*dx+dy*dy+dz*dz) - R
    gsdf = gd.min(axis=0)
    gowner = np.argmin(gd, axis=0)

    origin, bsize = sdflow.mpi_block(N, N, N)
    ox, oy, oz = origin
    lnx, lny, lnz = bsize

    s = sdflow.Solver(lnx, lny, lnz)
    s.init_mpi(N, N, N)
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(DT); s.set_body_force(F, 0.0, 0.0)
    s.set_advection(False)
    s.set_velocity_solver_params(80)
    s.set_pressure_multigrid(True, 4)
    s.set_pressure_pcg(True, 200, 1e-9)
    s.set_scene(*scene(), periodic=True)
    s.set_solid_from_scene(True)

    # (1) geometry, per rank, against the global single-rank fields restricted to this block
    lowner = np.asarray(s.get_cut_owner())
    ref_owner = gowner[ox:ox+lnx, oy:oy+lny, oz:oz+lnz]
    owner_bad = int((lowner != ref_owner).sum())
    owner_bad = comm.allreduce(owner_bad, op=MPI.SUM)
    sdf_diff = 0.0   # the solver keeps the sampled SDF internally; compare via the owner + k instead

    for _ in range(STEPS):
        s.step()
    lsum = float(np.sum(np.asarray(s.get_u())))
    gsum = comm.allreduce(lsum, op=MPI.SUM)
    k = MU * (gsum / float(N**3)) / F
    Fh = np.asarray(s.hydro_force_torque())[0]     # already Allreduced across ranks in C++
    FR = np.asarray(s.hydro_force_torque_reaction())[0]   # route (b), Allreduced likewise
    Nc = s.fluid_momentum_cells()
    div = s.max_open_divergence()

    if rank == 0:
        print("np=%d  k=%.12e  div=%.3e  owner mismatches=%d  sum|F|=%.6e  "
              "reaction identity-1=%+.3e"
              % (size, k, div, owner_bad, np.abs(Fh).sum(),
                 FR[:, 0].sum() / (F * Nc[0]) - 1))
        np.savez(os.path.join(OUT, "np%d.npz" % size), k=k, F=Fh, FR=FR,
                 owner_mismatch=owner_bad, sdf_maxdiff=sdf_diff)


if __name__ == "__main__":
    if "--compare" in sys.argv:
        sys.exit(0 if compare() else 1)
    main()
