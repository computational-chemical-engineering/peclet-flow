"""ROTATION GATE: the reaction TORQUE against the exact Stokes torque 8*pi*mu*a^3*Omega.

Part A (statics): a sphere spinning in place. The geometry never moves (a sphere is invariant
under its own rotation), so this isolates the torque evaluation completely. Before the v3
transposed-stress wall term the reaction torque was a STRUCTURAL -31% (the budget measures the
Laplacian-form wall flux; the physical traction adds mu*(grad u)^T.n = mu*(n x Omega), whose force
integral is zero -- which is why the force never saw it -- and whose torque integral is exactly one
third). With it, the error is the aperture first-moment discretisation and CONVERGES.

Part B (dynamics): the full ResolvedCfdDem hand-off. The same sphere, given physical inertia
I = (2/5) m R^2 and an initial spin, decays by the torque the flow hands to dem
(apply_torque=True). Reference: the decay rate 1/tau = 8*pi*mu*R^3*lambda_rot / I with lambda_rot
the SAME BOX's own rotational drag measured in part A -- the settling-gate calibration pattern, so
the discretisation bias cancels between calibration and prediction.

Needs flow (+ dem + coupling for part B) on PYTHONPATH.
"""
import os
import sys
import numpy as np
from peclet import flow as sdflow

RHO, MU = 1.0, 0.1
OM0 = 1e-3
KN_R, KI_I, KI_R = 16, 2, 17


def make(N, R, dt, om, advect=False):
    x0 = 0.5 * N
    ni = np.array([1, -1, -1], dtype=np.int32)
    nr = np.zeros(KN_R); nr[0] = R; nr[14] = 1.0; nr[15] = 1.0
    ii = np.zeros((1, KI_I), dtype=np.int32); ii[0] = (0, -1)
    ir = np.zeros((1, KI_R)); ir[0, 0:3] = (x0, x0, x0); ir[0, 6] = 1.0; ir[0, 7] = 1.0
    s = sdflow.Solver(N, N, N)
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(dt); s.set_advection(advect)
    s.set_velocity_solver_params(100); s.set_pressure_solver_params(25)
    # The net-force-at-round-off criterion below needs the momentum solve run to the FIXED sweep
    # count: on branches where a residual stop (1e-5) is the default, that stop leaves a 1e-6..1e-5
    # net force on a spinning sphere (measured 2026-09-02) -- physics unchanged, gate meaning lost.
    if hasattr(s, "set_velocity_residual_tolerance"):
        s.set_velocity_residual_tolerance(0.0)
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(ni, nr, ii.ravel(), ir.ravel(), periodic=True)
    s.set_instance_motion(0, lin_vel=[0, 0, 0], ang_vel=[0, 0, om], center=[x0, x0, x0])
    s.set_solid_from_scene(True)
    return s


def static_lambda(N, R, steps=1000, dt=20.0):
    s = make(N, R, dt, OM0)
    for _ in range(steps):
        s.step()
    T = abs(np.asarray(s.hydro_force_torque_reaction())[1][0][2])
    F = np.linalg.norm(np.asarray(s.hydro_force_torque_reaction())[0][0])
    return T / (8 * np.pi * MU * R ** 3 * OM0), F / (MU * R ** 2 * OM0)


def part_a():
    print("A. static spin vs T = 8 pi mu a^3 Omega   (lambda_rot = T_measured / T_exact)")
    rungs = [(64, 9.6), (96, 9.6), (96, 14.4)]
    if os.environ.get("ROT_FULL", "0") == "1":
        rungs += [(128, 9.6), (128, 19.2)]
    lam = {}
    ok = True
    prev = None
    for N, R in rungs:
        l, fnet = static_lambda(N, R)
        lam[(N, R)] = l
        print("   N=%3d R=%5.2f  lambda_rot = %.5f  (err %+5.2f%%)   |F|net/(mu R^2 Om) = %.1e"
              % (N, R, l, 100 * (l - 1), fnet))
        ok = ok and abs(l - 1) < 0.05 and fnet < 1e-9
    # convergence: the two box rungs at R=9.6 must move TOWARD 1
    e64, e96 = abs(lam[(64, 9.6)] - 1), abs(lam[(96, 9.6)] - 1)
    conv = e96 < e64
    print("   GATE A %s  [|err| < 5%% at every rung; box growth moves it toward the exact "
          "answer (%.2f%% -> %.2f%%); the net force stays at round-off]"
          % ("PASS" if (ok and conv) else "FAIL", 100 * e64, 100 * e96))
    return ok and conv, lam[(64, 9.6)]


def part_b(lam):
    from peclet import dem as pdem
    from peclet_coupling.resolved import ResolvedCfdDem
    N, R, DT = 64, 9.6, 5.0
    RHO_P = 5.0
    m = RHO_P * 4 / 3 * np.pi * R ** 3
    I = 0.4 * m * R ** 2
    tau_pred = I / (8 * np.pi * MU * R ** 3 * lam)
    nstep = int(1.2 * tau_pred / DT)
    print("B. spin-decay through ResolvedCfdDem (apply_torque=True): I dOmega/dt = -T(Omega)")
    print("   tau_pred = I/(8 pi mu R^3 lambda_rot) = %.1f  (%d steps at dt=%.0f)"
          % (tau_pred, nstep, DT))
    d = pdem.Simulation(8)
    d.set_gravity(0.0, 0.0, 0.0)
    d.set_sphere_shape(1.0)
    d.set_global_scale(R)
    d.set_positions(np.array([[0.5 * N, 0.5 * N, 0.5 * N]], dtype=np.float32))
    d.set_inv_mass(np.array([1.0 / m], dtype=np.float32))
    d.set_inv_inertia(np.array([[1.0 / I] * 3], dtype=np.float32))
    d.set_angular_velocities(np.array([[0.0, 0.0, OM0]], dtype=np.float32))
    s = sdflow.Solver(N, N, N)
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(DT); s.set_advection(False)
    s.set_velocity_solver_params(100); s.set_pressure_solver_params(25)
    # The net-force-at-round-off criterion below needs the momentum solve run to the FIXED sweep
    # count: on branches where a residual stop (1e-5) is the default, that stop leaves a 1e-6..1e-5
    # net force on a spinning sphere (measured 2026-09-02) -- physics unchanged, gate meaning lost.
    if hasattr(s, "set_velocity_residual_tolerance"):
        s.set_velocity_residual_tolerance(0.0)
    s.set_pressure_multigrid(True, levels=4)
    drv = ResolvedCfdDem(s, d, radius=R, mu=MU, rho_f=RHO, fluid_dt=DT, dem_substeps=10,
                         periodic=True, rho_p=RHO_P, buoyancy=False, move=True,
                         apply_torque=True)
    ts, oms = [], []
    for k in range(nstep):
        drv.step()
        ts.append((k + 1) * DT)
        oms.append(float(np.asarray(d.get_angular_velocities())[0, 2]))
    ts = np.array(ts); oms = np.array(oms)
    # fit the decay rate over the first predicted tau (before the co-rotating fluid feeds back)
    mfit = ts <= tau_pred
    rate = -np.polyfit(ts[mfit], np.log(oms[mfit] / OM0), 1)[0]
    ratio = rate * tau_pred
    print("   Omega decayed %.4f -> %.4f of Omega0 over %.0f time units"
          % (1.0, oms[-1] / OM0, ts[-1]))
    print("   fitted decay rate * tau_pred = %.4f   (1 = the same box's own calibrated drag)"
          % ratio)
    ok = abs(ratio - 1) < 0.10
    print("   GATE B %s  [claim: the dem-integrated decay matches the same box's calibrated "
          "rotational drag within 10%%; the co-rotating-fluid feedback makes the tail slower "
          "than exponential, which is physics, so the fit window is the first tau]"
          % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    ok_a, lam = part_a()
    ok_b = True
    if os.environ.get("ROT_STATIC_ONLY", "0") != "1":
        ok_b = part_b(lam)
    print("ROTATION GATE %s" % ("PASS" if (ok_a and ok_b) else "FAIL"))
