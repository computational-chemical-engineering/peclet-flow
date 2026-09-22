"""Layer 4 rung 2 gate: the HYDRODYNAMIC FORCE, checked against the momentum balance the solved
field must already satisfy -- BOTH evaluations:

  * hydro_force_torque_reaction (route b, THE COUPLING FORCE): the discrete reaction. Its identity
    is sum F_c = f_c * N_c with N_c the solver's own count of fluid momentum cells
    (fluid_momentum_cells), and it must hold to the momentum solver's residual -- this is the
    IMPLEMENTATION-COMPLETENESS check, not an accuracy check (the identity is a tautology of a
    correctly assembled budget; if it holds only to percent, a term is missing).
  * hydro_force_torque (traction, DIAGNOSTIC): the reconstructed surface integral, kept so its
    resolution-independent ~29% under-read stays visible.

Setup: Stokes flow through a fixed periodic sphere array driven by a uniform body force f (force
per unit volume, applied to the fluid). At steady state the momentum balance over the periodic cell
is exact and closes with no empirical input:

    sum over spheres of F_hydro  =  f * V_fluid

-- the body force fed into the fluid has nowhere to go but the grain surfaces. So integrating the
traction and comparing it against f * V_fluid is a genuine SELF-CONSISTENCY test of the surface
integral against the same field the velocity-based permeability is read from, exactly as the
Layer-4 spec asks; it needs no Zick-Homsy table to state, and it fails loudly if the aperture
wall-area vector, the traction, or the owner attribution is wrong.

V_fluid is reported two ways, because they differ at O(h) and the reader should see which is
which: the ANALYTIC fluid volume of the sphere array, and the DISCRETE one the solver's own
cut-cell apertures imply.

Also reported: the Darcy permeability k = mu <u> / f measured from the velocity field, so the run
can be lined up against validate_zick_homsy_sdflow.py's numbers, and the per-sphere force spread
(the array is symmetric, so four equal spheres must carry equal drag -- a direct check that the
owner attribution partitions the surface correctly).
"""
import os
import sys
import numpy as np
from peclet import flow as sdflow

MU, RHO, DT = float(os.environ.get("FORCE_MU", "0.1")), 1.0, float(os.environ.get("FORCE_DT", "60"))
F = float(os.environ.get("FORCE_F", "1e-3"))
STEPS = int(os.environ.get("FORCE_STEPS", "400"))
# R0: the same identity with the explicit advective term in the budget. Re is reported from the
# measured superficial velocity so the run can be placed on the plan's "Re ~ 30" target.
ADVECT = os.environ.get("FORCE_ADVECT", "0") == "1"
KN_R, KI_I, KI_R = 16, 2, 17
F_SCALE = os.environ.get("FORCE_F_SCALE", "1") == "1"


def run(N, RF=0.18, wallgrad=False):  # wallgrad kept only for the log line
    R = RF * N
    # ---- the advective leg must hold the CELL Reynolds number fixed across the ladder ---------
    # set_advection() puts the advective term in the RHS, lagged at the Picard iterate -- it is
    # EXPLICIT (set_implicit_advection moves it into the operator, but the reaction budget refuses
    # that path). The step is therefore the textbook semi-implicit one, whose linear stability
    # limit for a mode k is |u.k|^2 <= 2*nu*k^2/dt + (nu*k^2)^2, i.e. at large dt just the CELL
    # REYNOLDS NUMBER  Re_h = rho*|u|*h/mu  <~ pi.
    # With f and mu held fixed this lattice's permeability grows like N^2 (cells^2), so |u| and
    # Re_h grow like N^2 too: Re_h = 0.81 at N=16 (stable at every dt tried, up to 240) but 3.1 at
    # N=32 and 12.5 at N=64. That is why a fixed f NaNs at every N >= 24: the momentum field
    # diverges over ~70 steps and the pressure preconditioner then returns non-finite z. It is a
    # property of the scheme, not a solver defect -- no code change moves it.
    # Scaling f like N^-2 pins Re_h at the N=16 value (0.81, a factor ~4 below the limit) for every
    # rung, and lets Re_d = rho*|u|*2R/mu grow like N -- ~12 at N=64, ~23 at N=128, i.e. the plan's
    # "Re ~ 30" target lands on the default ladder. FORCE_F_SCALE=0 restores the raw (unstable) f.
    f = F * (16.0 / N) ** 2 if (ADVECT and F_SCALE) else F
    CEN = [(0.25*N, 0.25*N, 0.25*N), (0.75*N, 0.75*N, 0.25*N),
           (0.75*N, 0.25*N, 0.75*N), (0.25*N, 0.75*N, 0.75*N)]
    node_ints = np.array([1, -1, -1], dtype=np.int32)
    node_reals = np.zeros(KN_R); node_reals[0] = R
    node_reals[14] = 1.0; node_reals[15] = 1.0
    ii = np.zeros((len(CEN), KI_I), dtype=np.int32)
    ir = np.zeros((len(CEN), KI_R))
    for m, c in enumerate(CEN):
        ii[m] = (0, -1); ir[m, 0:3] = c; ir[m, 6] = 1.0; ir[m, 7] = 1.0

    s = sdflow.Solver(N, N, N)
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(DT)
    s.set_body_force(f, 0.0, 0.0)
    s.set_advection(ADVECT)
    s.set_velocity_solver_params(100)
    s.set_pressure_solver_params(25)
    # MG depth 4, not 1. levels=1 leaves the 'coarse' solve on the FULL grid, which on
    # CUDA costs 920 ms/step at N=64 against 19.5 ms at depth 4 (measured). Depth changes the
    # solver path, not the converged answer, and both runs of a comparison use the same depth.
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(node_ints, node_reals, ii.ravel(), ir.ravel(), periodic=True)
    s.set_solid_from_scene(True)
    for _ in range(STEPS):
        s.step()
    u = np.asarray(s.get_u())
    ft = np.asarray(s.hydro_force_torque())
    Fh, Th, Fp, Fv = ft[0], ft[1], ft[2], ft[3]
    fr = np.asarray(s.hydro_force_torque_reaction())
    FR, TR = fr[0], fr[1]
    Nc = s.fluid_momentum_cells()

    # analytic fluid volume (spheres do not overlap at RF=0.18 on this lattice)
    Vsolid = len(CEN) * 4.0 / 3.0 * np.pi * R**3
    Vfluid_an = N**3 - Vsolid
    # discrete fluid volume from the solver's own cell volume fractions
    g = np.arange(N).astype(float)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    sd = np.full((N, N, N), 1e30)
    for (cx, cy, cz) in CEN:
        dx = np.abs(X-cx); dx = np.minimum(dx, N-dx)
        dy = np.abs(Y-cy); dy = np.minimum(dy, N-dy)
        dz = np.abs(Z-cz); dz = np.minimum(dz, N-dz)
        sd = np.minimum(sd, np.sqrt(dx*dx+dy*dy+dz*dz) - R)
    Vfluid_di = float(np.clip(0.5 + sd, 0.0, 1.0).sum())

    k = MU * float(u.mean()) / f
    Re = RHO * float(np.abs(u).mean()) * (2 * R) / MU
    tot = Fh.sum(axis=0)
    exp_an, exp_di = f * Vfluid_an, f * Vfluid_di
    spread = (Fh[:, 0].max() - Fh[:, 0].min()) / abs(Fh[:, 0].mean())
    probe = np.asarray(s.wall_area_probe())
    print("  N=%3d  porosity %.4f  k = mu<u>/f = %.4e  max|div| %.2e  advect=%s  Re_d=%.2f"
          "  f=%.4e  Re_h=%.2f (semi-implicit limit ~pi)"
          % (N, Vfluid_an / N**3, k, s.max_open_divergence(), ADVECT, Re, f,
             RHO * float(np.abs(u).max()) / MU))
    print("         A_wall probe / -V_solid = %.5f %.5f %.5f  (exactly 1 if the aperture wall-area "
          "vectors are right)" % tuple(probe / -Vsolid))
    print("         sum F_hydro           = (%+.6e, %+.6e, %+.6e)" % tuple(tot))
    print("         f * V_fluid analytic  = %+.6e   -> ratio %.5f  (err %+.2f%%)"
          % (exp_an, tot[0] / exp_an, 100 * (tot[0] / exp_an - 1)))
    print("         f * V_fluid discrete  = %+.6e   -> ratio %.5f  (err %+.2f%%)"
          % (exp_di, tot[0] / exp_di, 100 * (tot[0] / exp_di - 1)))
    print("         transverse leakage    = %.2e (relative to F_x; symmetry says 0)"
          % (max(abs(tot[1]), abs(tot[2])) / abs(tot[0])))
    print("         per-sphere F_x spread = %.2e (4 identical spheres -> owner attribution splits "
          "the surface evenly)" % spread)
    print("         split: pressure %+.6e  viscous %+.6e  (of the %+.6e needed)"
          % (Fp[:, 0].sum(), Fv[:, 0].sum(), exp_di))
    print("         sum torque            = (%+.2e, %+.2e, %+.2e)" % tuple(Th.sum(axis=0)))
    rtot = FR.sum(axis=0)
    bt = np.asarray(s.reaction_budget_terms())  # [0]=unsteady, [1]=advective, per component
    unst, advs = bt[0], bt[1]
    exp_re = f * Nc[0] + advs[0] - unst[0]
    rspread = (FR[:, 0].max() - FR[:, 0].min()) / abs(FR[:, 0].mean())
    print("         REACTION (route b):  sum F = (%+.6e, %+.2e, %+.2e)" % tuple(rtot))
    print("           identity target    = %+.6e   ratio-1 = %+.3e   (must be solver residual, "
          "NOT percent -- a percent-level miss means a budget term is missing)"
          % (exp_re, rtot[0] / exp_re - 1))
    print("           terms: f*N_fluid %+.6e   sum A (advective wall flux) %+.3e (%+.3f%% of f*N)"
          "   sum unsteady %+.3e (%+.3f%%)"
          % (f * Nc[0], advs[0], 100 * advs[0] / (f * Nc[0]), unst[0],
             100 * unst[0] / (f * Nc[0])))
    print("           per-sphere spread %.2e   transverse %.2e   sum torque (%+.2e, %+.2e, %+.2e)"
          % (rspread, max(abs(rtot[1]), abs(rtot[2])) / abs(rtot[0]), *TR.sum(axis=0)))
    ok = abs(rtot[0] / exp_re - 1) < 1e-4
    print("         REACTION GATE %s" % ("PASS" if ok else "FAIL"))
    return abs(tot[0] / exp_di - 1), spread, abs(rtot[0] / exp_re - 1), ok


if __name__ == "__main__":
    Ns = [int(x) for x in (sys.argv[1:] or ["64", "128"])]
    wg = os.environ.get("FORCE_WALLGRAD", "0") == "1"
    errs = [run(N, wallgrad=wg) for N in Ns]
    print("  traction (diagnostic) momentum-balance error:")
    for N, r in zip(Ns, errs):
        print("    N=%3d  %.3f%%" % (N, 100 * r[0]))
    print("  reaction (route b) identity residual:")
    allok = True
    for N, r in zip(Ns, errs):
        print("    N=%3d  %.3e" % (N, r[2]))
        allok = allok and r[3]
    print("GATE %s  [reaction identity to solver residual at every N]"
          % ("PASS" if allok else "FAIL"))
