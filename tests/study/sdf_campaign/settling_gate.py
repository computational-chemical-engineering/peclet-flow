"""Layer 4 rung 3 gate: the SETTLING SPHERE, against a terminal velocity predicted from the drag
coefficient MEASURED IN THE SAME BOX.

Two runs in one periodic box, so every periodicity/blockage correction is common to both and
cancels -- no literature wall-correction fit is involved, which is the point:

  (a) CALIBRATION. Sphere held fixed; a uniform body force f drives the fluid past it. At steady
      state the box's own drag coefficient is
          lambda := F_drag / (6 pi mu R <u>)
      with <u> the superficial (whole-box volume-average) velocity. For an isolated sphere in
      unbounded Stokes flow lambda would be 1; in a periodic box at this solid fraction it is
      several, and that IS the number the settling run must reproduce.

  (b) SETTLING. The same sphere, now free, carrying a body force F_g, with a compensating fluid
      body force -F_g / V_fluid so total momentum is conserved and a steady state exists (without
      it the whole periodic system accelerates forever and there is no terminal velocity to find).
      At terminal state the hydrodynamic force balances F_g, so the prediction is
          |<u> - U_particle| = F_g / (6 pi mu R lambda).

The dem solver integrates the particle (one grain, no contacts) exactly as it would in a real
resolved bed, so this gates the whole loop: scene bridge -> geometry rebuild -> flow step ->
traction integral -> dem sub-steps -> new state.
"""
import os
import sys
import numpy as np
from peclet import flow as sdflow
from peclet import dem as pdem
from peclet_coupling.resolved import ResolvedCfdDem

N = int(os.environ.get("SETTLE_N", "48"))
RF = 0.12
R = RF * N
MU, RHO_F = 1.0, 1.0
RHO_P = 5.0
# Two timesteps, for two different limits.
#  - CALIBRATION has no particle degree of freedom, so it wants a LARGE dt: backward Euler then
#    reaches the steady Stokes solution in few steps.
#  - SETTLING is a WEAK EXPLICIT exchange between the grain and the fluid it drags. Its stability
#    limit is dt * k / m_reduced < 2 with k = 6 pi mu R lambda the drag rate and m_reduced the
#    reduced mass of grain and entrained fluid. At dt=20 that product was 1.7 -- just inside the
#    bound on paper, and the run blew up to 1e16 in 60 steps. dt=2 puts it at 0.17.
DT_CAL = float(os.environ.get("SETTLE_DT_CAL", "20.0"))
DT = float(os.environ.get("SETTLE_DT", "2.0"))
NSTEP = int(os.environ.get("SETTLE_STEPS", "600"))
NCAL = int(os.environ.get("SETTLE_CAL_STEPS", "400"))
KN_R, KI_I, KI_R = 16, 2, 17
V_SPHERE = 4.0 / 3.0 * np.pi * R**3
V_FLUID = N**3 - V_SPHERE
CEN = (0.5 * N, 0.5 * N, 0.5 * N)


def make_flow(body_force, dt=None):
    node_ints = np.array([1, -1, -1], dtype=np.int32)
    node_reals = np.zeros(KN_R); node_reals[0] = R
    node_reals[14] = 1.0; node_reals[15] = 1.0
    ii = np.zeros((1, KI_I), dtype=np.int32); ii[0] = (0, -1)
    ir = np.zeros((1, KI_R)); ir[0, 0:3] = CEN; ir[0, 6] = 1.0; ir[0, 7] = 1.0
    s = sdflow.Solver(N, N, N)
    s.set_rho(RHO_F); s.set_mu(MU); s.set_dt(DT if dt is None else dt)
    s.set_advection(False)
    s.set_velocity_solver_params(80)
    s.set_pressure_solver_params(20)
    # MG depth 4, not 1. levels=1 leaves the 'coarse' solve on the FULL grid, which on
    # CUDA costs 920 ms/step at N=64 against 19.5 ms at depth 4 (measured). Depth changes the
    # solver path, not the converged answer, and both runs of a comparison use the same depth.
    s.set_pressure_multigrid(True, levels=4)
    s.set_body_force(body_force, 0.0, 0.0)
    s.set_scene(node_ints, node_reals, ii.ravel(), ir.ravel(), periodic=True)
    return s


def calibrate(f=1e-3, steps=None):
    s = make_flow(f, dt=DT_CAL)
    s.set_solid_from_scene(True)
    for _ in range(steps or NCAL):
        s.step()
    u = np.asarray(s.get_u())
    # REACTION force (route b) -- the same method the driver feeds dem, so the drag coefficient
    # measured here and the force felt in the settling run are one number, and the gate's
    # self-consistency claim is real. (With the traction the two halves used the same *biased*
    # method too, but its frame error broke the cancellation for a moving wall.)
    Fd = np.asarray(s.hydro_force_torque_reaction())[0][0]
    Ftr = np.asarray(s.hydro_force_torque())[0][0]
    Nc = s.fluid_momentum_cells()
    umean = float(u.mean())
    lam = Fd[0] / (6.0 * np.pi * MU * R * umean)
    print("(a) calibration: %d steps at dt=%.1f  f=%.1e  <u>=%.6e  F_drag=(%+.4e,%+.1e,%+.1e)"
          % (steps or NCAL, DT_CAL, f, umean, *Fd))
    print("    REACTION identity F / (f*N_fluid) - 1 = %+.3e   (must be ~unsteady tail + solver "
          "residual; percent-level means a budget term is missing)" % (Fd[0] / (f * Nc[0]) - 1))
    print("    [traction diagnostic on the same run: F / (f*V_fluid) = %.5f -- the ~29%% under-read "
          "kept visible]" % (Ftr[0] / (f * V_FLUID)))
    print("    lambda = F_drag / (6 pi mu R <u>) = %.4f   (1 would be an isolated sphere)" % lam)
    return lam, umean


def settle(lam, Fg=None):
    # F_g chosen so the predicted slip is a comfortable fraction of a cell per fluid step
    if Fg is None:
        Fg = 6.0 * np.pi * MU * R * lam * 0.02
    fcomp = -Fg / V_FLUID                       # compensating fluid body force: momentum conserved
    s = make_flow(fcomp)

    d = pdem.Simulation(1)
    d.initialize(shape_type=1, radius=R)            # sphere
    d.set_domain((0.0, 0.0, 0.0), (float(N), float(N), float(N)))
    d.enable_periodicity(True, True, True)
    d.set_gravity(0.0, 0.0, 0.0)                    # gravity arrives as an external force instead
    pos = np.zeros((1, 4), dtype=np.float32)
    pos[0, 0:3] = CEN
    pos[0, 3] = 1.0                                 # w = 1: a zero w silently remaps the position
    d.set_positions(pos)
    d.set_velocities(np.zeros((1, 4), dtype=np.float32))
    # dem assigns unit mass regardless of size. A unit-mass sphere of radius ~6 cells makes the
    # WEAK explicit coupling unstable (the drag impulse per dem sub-step, dt*6*pi*mu*R*lambda/m,
    # must stay below 2 and would be ~650 here), so set a physical mass through set_inv_mass:
    # m = rho_p * V. This is a real requirement of resolved coupling, not a tuning knob -- a grain
    # lighter than the fluid it displaces is the classic added-mass instability.
    rho_p = RHO_P
    m = rho_p * V_SPHERE
    d.set_inv_mass(np.array([1.0 / m], dtype=np.float32))
    if os.environ.get("SEED_FRESH", "0") == "1":
        s.set_fresh_cell_seed(True)
    drv = ResolvedCfdDem(s, d, radius=R, mu=MU, rho_f=RHO_F, fluid_dt=DT, dem_substeps=10,
                         periodic=True, gravity=(Fg / m, 0.0, 0.0), rho_p=rho_p, buoyancy=False,
                         move=True)
    pred = Fg / (6.0 * np.pi * MU * R * lam)
    hist = []
    mom = []
    for k in range(NSTEP):
        drv.step()
        up = float(np.asarray(d.get_velocities())[0, 0])
        uf = float(np.asarray(s.get_u()).mean())
        hist.append(abs(uf - up))          # MAGNITUDE: the particle leads, so uf - up is negative
        # ACTION-REACTION DIAGNOSTIC. The external forces sum to zero by construction (F_g on the
        # grain, -F_g/V_fluid spread over the fluid), so the total x-momentum must be constant. It
        # is not, because the force REPORTED to the grain is an independent reconstruction of the
        # traction rather than the discrete reaction the operator applied to the fluid -- the same
        # gap that makes the static drag 29% low. The drift rate measures it directly.
        mom.append(m * up + RHO_F * uf * N**3)
        if k % max(1, NSTEP // 10) == max(1, NSTEP // 10) - 1:
            print("    step %3d  |U_slip|=%.6e  (pred %.6e, ratio %.4f)  U_p=%+.4e  <u>=%+.4e  "
                  "p_tot=%+.4e" % (k + 1, hist[-1], pred, hist[-1] / pred, up, uf, mom[-1]))
    slip = hist[-1]
    tail = max(1, NSTEP // 20)
    drift = abs(hist[-1] - hist[-1 - tail]) / abs(pred)
    print("(b) settling: F_g=%.4e  m=%.4e (rho_p=%.3f)" % (Fg, m, rho_p))
    print("    terminal |<u> - U_p| = %.6e" % slip)
    print("    predicted from the MEASURED lambda = %.6e" % pred)
    print("    ratio %.4f   (slip drift over the last %d steps: %.2e of the prediction)"
          % (slip / pred, tail, drift))
    # With the REACTION force the action-reaction gap is zero by construction; the residual drift
    # here is the SETUP's own O(h) mismatch -- the compensating fluid body force is -F_g/V_fluid
    # with the ANALYTIC volume, while the discrete fluid receives f*N_fluid-cells -- plus rebuild
    # fresh-cell blips. Measured 44x smaller than the traction-force leak and of setup origin.
    print("    momentum: p_tot %+.4e -> %+.4e over %d steps  (%.2e F_g per unit time; the "
          "traction-force version leaked 0.58 F_g per unit time)"
          % (mom[0], mom[-1], NSTEP,
             abs(mom[-1] - mom[0]) / (NSTEP * DT * max(abs(Fg), 1e-300))))
    ok = abs(slip / pred - 1.0) < 0.05
    print("GATE %s   [claim: terminal velocity matches the same box's measured drag within 5%%]"
          % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    lam, _ = calibrate()
    settle(lam)
