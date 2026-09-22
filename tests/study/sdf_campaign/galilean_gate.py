"""Layer 3 rungs 2-3 gate: GALILEAN INVARIANCE of the moving-wall cut-cell solver.

Stokes flow through a periodic sphere bed, driven by a body force, solved twice:

  A (lab frame)      spheres static, wall velocity 0                  -> u_A
  B (boosted frame)  IDENTICAL geometry, every sphere given linVel=-V -> u_B

In a frame translating at V the same physical flow reads u' = u - V and the walls move at -V, so
the discrete claim is  u_B + V == u_A  in every fluid cell.

Why this gates rung 3 and not just rung 2. The true boosted solution has
    div_open(u_A - V) = -V_x (oE - oW)  != 0  at every cut cell,
which is exactly cancelled by the wall term u_w . A_wall with A_wall = -(oE-oW, oN-oS, oT-oB).
Drop the wall term and the projection wrongly forces div_open(u)=0, so the run is repeated with
set_wall_flux_divergence(False) to EXHIBIT the failure rather than assert it.

Stokes (advection off) is required: with the convective term, a fixed geometry plus a moving wall
is not a Galilean boost of the static problem -- the missing -(V.grad)u term is real physics, not
a discretisation error.
"""
import sys
import numpy as np
from peclet import flow as sdflow

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
RF = 0.18
R = RF * N
V = 0.7                      # boost speed, cell units per time
F = 1e-3
import os
MU, RHO, DT = 0.1, 1.0, 60.0
STEPS = int(os.environ.get("GAL_STEPS", "260"))

CEN = [(0.25*N, 0.25*N, 0.25*N), (0.75*N, 0.75*N, 0.25*N),
       (0.75*N, 0.25*N, 0.75*N), (0.25*N, 0.75*N, 0.75*N)]

KN_R, KI_I, KI_R = 16, 2, 17
node_ints  = np.array([1, -1, -1], dtype=np.int32)
node_reals = np.zeros(KN_R); node_reals[0] = R
node_reals[14] = 1.0; node_reals[15] = 1.0
inst_ints  = np.zeros((len(CEN), KI_I), dtype=np.int32)
inst_reals = np.zeros((len(CEN), KI_R))
for m, c in enumerate(CEN):
    inst_ints[m] = (0, -1)
    inst_reals[m, 0:3] = c
    inst_reals[m, 6] = 1.0
    inst_reals[m, 7] = 1.0


def run(boost, wall_flux=True):
    s = sdflow.Solver(N, N, N)
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(DT)
    s.set_body_force(F, 0.0, 0.0)
    s.set_advection(False)
    s.set_velocity_solver_params(80)
    s.set_pressure_solver_params(20)
    # MG depth 4, not 1. levels=1 leaves the 'coarse' solve on the FULL grid, which on
    # CUDA costs 920 ms/step at N=64 against 19.5 ms at depth 4 (measured). Depth changes the
    # solver path, not the converged answer, and both runs of a comparison use the same depth.
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(node_ints, node_reals, inst_ints.ravel(), inst_reals.ravel(), periodic=True)
    if boost:
        s.set_wall_flux_divergence(wall_flux)
        for i in range(len(CEN)):
            s.set_instance_motion(i, lin_vel=[-V, 0.0, 0.0], ang_vel=[0.0, 0.0, 0.0])
    s.set_solid_from_scene(True)
    prev = 0.0
    for it in range(STEPS):
        s.step()
        um = float(np.asarray(s.get_u()).mean())
        prev = um   # fixed step count: both runs get IDENTICAL solver work, so the
                    # residual difference below is a convergence level, not a stopping artefact
    return (np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w()),
            s.max_open_divergence(), s.wall_flux_imbalance(), it + 1,
            np.asarray(s.hydro_force_torque()),
            np.asarray(s.hydro_force_torque_reaction()))


# PER-COMPONENT masks. u lives at (i-1/2,j,k), v at (i,j-1/2,k), w at (i,j,k-1/2); a cell whose
# CENTRE is fluid can have its u-point inside the solid, where the solver stores a masked 0 in both
# runs -- so a cell-centred mask reports a spurious error of exactly V/max|u_A| there. (That is what
# the first version of this gate measured, and it looked like a failure.)
def sdf_at(offx, offy, offz):
    g = np.arange(N).astype(float)
    X, Y, Z = np.meshgrid(g + offx, g + offy, g + offz, indexing="ij")
    d = np.full((N, N, N), 1e30)
    for (cx, cy, cz) in CEN:
        dx = np.abs(X-cx); dx = np.minimum(dx, N-dx)
        dy = np.abs(Y-cy); dy = np.minimum(dy, N-dy)
        dz = np.abs(Z-cz); dz = np.minimum(dz, N-dz)
        d = np.minimum(d, np.sqrt(dx*dx+dy*dy+dz*dz) - R)
    return d

sdU = sdf_at(-0.5, 0.0, 0.0)
sdV = sdf_at(0.0, -0.5, 0.0)
sdW = sdf_at(0.0, 0.0, -0.5)
mU, mV, mW = sdU > 0.0, sdV > 0.0, sdW > 0.0            # every fluid unknown, cut band included
bU, bV, bW = mU & (sdU < 1.5), mV & (sdV < 1.5), mW & (sdW < 1.5)   # the cut band alone
dU, dV, dW = sdU > 2.0, sdV > 2.0, sdW > 2.0            # deep fluid

uA, vA, wA, divA, imbA, itA, ftA, frA = run(False)
print("A static   : steps=%3d  max|div|=%.3e  <u>=%.6e" % (itA, divA, uA.mean()))
print("             sum F_hydro = %+.6e   (traction diagnostic, LAB frame)"
      % ftA[0][:, 0].sum())
print("             reaction F  = %+.6e   (route b, LAB frame)" % frA[0][:, 0].sum())

for wf in (True, False):
    uB, vB, wB, divB, imbB, itB, ftB, frB = run(True, wall_flux=wf)
    du = (uB + V) - uA
    dv = vB - vA
    dw = wB - wA
    scale = max(abs(uA[mU]).max(), 1e-30)

    def worst(masks):
        return max(abs(du[masks[0]]).max(), abs(dv[masks[1]]).max(),
                   abs(dw[masks[2]]).max()) / scale

    def l2(masks):
        n = masks[0].sum() + masks[1].sum() + masks[2].sum()
        ssq = (du[masks[0]]**2).sum() + (dv[masks[1]]**2).sum() + (dw[masks[2]]**2).sum()
        return np.sqrt(ssq / n) / scale

    e_all, e_band, e_deep = worst((mU, mV, mW)), worst((bU, bV, bW)), worst((dU, dV, dW))
    print("B boosted  : wall_flux=%-5s steps=%3d  max|div|=%.3e  wall-flux imbalance=%.3e"
          % (wf, itB, divB, imbB))
    # FORCE IS FRAME-INVARIANT. A boost shifts u by a constant, changing neither p nor grad u,
    # so the traction integral must return the SAME force. If it does not, the moving-wall force is
    # wrong -- and a resolved CFD-DEM loop driven by it runs away instead of settling, which is
    # exactly the failure this check was added to localise.
    print("             sum F_hydro = %+.6e   ratio to the lab frame %.6f  (traction)"
          % (ftB[0][:, 0].sum(), ftB[0][:, 0].sum() / ftA[0][:, 0].sum()))
    print("             reaction F  = %+.6e   ratio to the lab frame %.9f  (route b -- at steady "
          "state the reaction is IDENTICAL between frames up to solver residual: Delta u and the "
          "viscous differences of u* are boost-invariant)"
          % (frB[0][:, 0].sum(), frB[0][:, 0].sum() / frA[0][:, 0].sum()))
    print("             max|u_B + V - u_A| / max|u_A|:  all fluid %.3e   cut band %.3e   "
          "deep %.3e   L2(all) %.3e" % (e_all, e_band, e_deep, l2((mU, mV, mW))))
    if wf:
        keep = e_all
        ok = e_all < 1e-4
        print("GATE (rung 3 ON)  %s   [claim: Galilean invariant to solver tolerance]"
              % ("PASS" if ok else "FAIL"))
    else:
        print("CONTRAST (rung 3 OFF) all-fluid error %.3e = %.0fx the rung-3-ON error -- the wall "
              "flux term is load bearing" % (e_all, e_all / max(keep, 1e-300)))
