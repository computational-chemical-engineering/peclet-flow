import importlib.util, os, subprocess, sys
_local = os.environ.get("PECLET_LOCAL_BUILD")
if _local:
    for p in _local.split(os.pathsep):
        sys.path.insert(0, p)
elif importlib.util.find_spec("peclet") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peclet"], check=True)

import time
import numpy as np
import matplotlib.pyplot as plt
from peclet import flow as sdflow
from peclet import dem as pdem
from peclet.core import geom

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})

# The four experiments, from the paper's Table I (the printed viscosity-header unit is an
# erratum; values are Pa s). u_inf is NOT measured: it is the unbounded terminal velocity from
# the Abraham correlation, which is how the paper defines Re. u_max/u_inf are the measured
# ratios from Table II.
CASES = {   #     rho_f     mu      u_inf   ratio   Re
    "E1": (970., 0.373, 0.03829, 0.947,  1.5),
    "E2": (965., 0.212, 0.05992, 0.953,  4.1),
    "E3": (962., 0.113, 0.09062, 0.959, 11.6),
    "E4": (960., 0.058, 0.12839, 0.955, 31.9)}
RHO_P_SI, D_SI, G_SI = 1120., 0.015, 9.81

USTAR, WALL = 0.02, 4.3          # u_inf in cell units; tank wall thickness (off-lattice: 4.3)
KI_I, KI_R = 2, 17

def run_case(case, DH, sweeps=60, gpu_dt=None, advect=True):
    rho_f, mu_si, u_inf, ratio_exp, Re = CASES[case]
    # Grid: round UP to multiples of 8 so the 4-level multigrid can actually coarsen (a 62-wide
    # grid has one factor of two); the extra padding goes into the wall thickness.
    NX = int(np.ceil((100 / 15 * DH + 2 * WALL) / 8) * 8)
    NY = int(np.ceil((160 / 15 * DH + 2 * WALL) / 8) * 8)
    WX = (NX - 100 / 15 * DH) / 2
    WY = (NY - 160 / 15 * DH) / 2
    NU = USTAR * DH / Re
    GSTAR = G_SI * D_SI / u_inf ** 2 * USTAR ** 2 / DH
    RATIO = RHO_P_SI / rho_f
    # dt: a fraction of the particle response time, and small enough that the sphere moves a
    # fraction of a cell per rebuild
    tau = RATIO * DH * Re / (18 * USTAR)
    dt = gpu_dt if gpu_dt else min(max(tau / 12, 1.0), 0.35 / USTAR / 4)
    b = geom.SceneBuilder()
    slab = b.add_leaf("box", [NX * 0.7, NY * 0.7, NX * 0.7])
    cavity = b.add_leaf("box", [(NX - 2 * WX) / 2, (NY - 2 * WY) / 2, (NX - 2 * WX) / 2])
    tank = b.add_difference(slab, cavity)
    sph = b.add_leaf("sphere", [DH / 2])
    ni, nr, _, _ = b.encode()
    OFF = 0.3                       # shift the tank so no wall sits exactly on a grid plane
    x0 = 0.5 * NX + OFF
    y0 = WY + OFF + 8.5 * DH
    ii = np.zeros((2, KI_I), dtype=np.int32); ir = np.zeros((2, KI_R))
    ii[0] = (tank, -1); ir[0, 0:3] = (0.5 * NX + OFF, 0.5 * NY + OFF, 0.5 * NX + OFF)
    ir[0, 6] = 1.0; ir[0, 7] = 1.0
    ii[1] = (sph, -1); ir[1, 0:3] = (x0, y0, x0); ir[1, 6] = 1.0; ir[1, 7] = 1.0
    s = sdflow.Solver(NX, NY, NX)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(dt); s.set_advection(advect)
    s.set_velocity_solver_params(sweeps); s.set_pressure_solver_params(20)
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(np.asarray(ni, np.int32), np.asarray(nr, float), ii.ravel(), ir.ravel(),
                periodic=True)
    s.set_solid_from_scene(True)

    m = RATIO * (np.pi / 6) * DH ** 3
    Vp = (np.pi / 6) * DH ** 3
    # Virtual-mass stabilization for the explicit coupling: at rho_p/rho_f = 1.15 the resolved
    # hydrodynamic force contains an added-mass part evaluated one step late, which rings — and
    # near the floor the added mass grows and the ringing diverges. Integrating with m + ma and
    # adding the lagged ma*a back keeps every steady and smooth trajectory EXACT (m dv/dt = F - Fg
    # when a is converged) while cutting the loop gain on the fluctuation.
    ma = 2.0 * Vp
    d = pdem.Simulation(8)
    d.set_gravity(0.0, 0.0, 0.0)
    d.set_sphere_shape(1.0)
    d.set_positions(np.array([[x0, y0, x0]], dtype=np.float32))
    d.set_inv_mass(np.array([1.0 / (m + ma)], dtype=np.float32))
    d.set_inv_inertia(np.array([[0, 0, 0]], dtype=np.float32))   # translation-only, like the paper
    Fg = (RATIO - 1.0) * Vp * GSTAR       # buoyant weight (the fluid carries no hydrostatic field)
    # Hard bound: 2.2x the time to fall the full release height at 0.8 u*. The physical stops
    # below (near-floor gap, deep post-peak deceleration) fire first; this cannot hang.
    kmax = int(2.2 * (8.5 * DH / (0.8 * USTAR)) / dt)
    t0 = time.time(); tr = []; vpk = 0.0; a_prev = np.zeros(3)
    for k in range(kmax):
        p = np.asarray(d.get_positions())[0].astype(float)
        v = np.asarray(d.get_velocities())[0].astype(float)
        s.set_instance_transform(1, p.tolist())
        s.set_instance_motion(1, lin_vel=v.tolist())
        s.rebuild_geometry(); s.step()
        F = np.asarray(s.hydro_force_torque_reaction())[0][1].astype(float)
        F[1] -= Fg
        F += ma * a_prev
        d.set_external_forces(np.array([F], dtype=np.float32))
        for _ in range(10):
            d.step(dt / 10)
        a_prev = (np.asarray(d.get_velocities())[0].astype(float) - v) / dt
        tr.append(((k + 1) * dt, p[1] - (WY + OFF) - DH / 2, v[1]))   # gap = bottom apex to floor
        vpk = max(vpk, -v[1])
        if tr[-1][1] < 0.5 * DH:
            break                          # sub-cell gap: lubrication unresolved, dem would take over
        if vpk > 0.3 * USTAR and -v[1] < 0.45 * vpk:
            break                          # well past the peak: the approach to rest is asymptotic
    tr = np.array(tr)
    return dict(t=tr[:, 0], gap=tr[:, 1] / DH, v=tr[:, 2] / USTAR, case=case, DH=DH,
                ratio_exp=ratio_exp, Re=Re, u_inf=u_inf, wall=time.time() - t0,
                nstep=len(tr), dt=dt)



def run_unbounded(case, DH, N=96, advect=True, nstep=4000):
    rho_f, mu_si, u_inf, ratio_exp, Re = CASES[case]
    NU = USTAR * DH / Re
    GSTAR = G_SI * D_SI / u_inf ** 2 * USTAR ** 2 / DH
    RATIO = RHO_P_SI / rho_f
    tau = RATIO * DH * Re / (18 * USTAR)
    dt = min(max(tau / 12, 1.0), 0.35 / USTAR / 4)
    b = geom.SceneBuilder()
    sph = b.add_leaf("sphere", [DH / 2])
    ni, nr, _, _ = b.encode()
    x0 = 0.5 * N + 0.3
    ii = np.zeros((1, KI_I), dtype=np.int32); ir = np.zeros((1, KI_R))
    ii[0] = (sph, -1); ir[0, 0:3] = (x0, x0, x0); ir[0, 6] = 1.0; ir[0, 7] = 1.0
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(dt); s.set_advection(advect)
    s.set_velocity_solver_params(60); s.set_pressure_solver_params(20)
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(np.asarray(ni, np.int32), np.asarray(nr, float), ii.ravel(), ir.ravel(),
                periodic=True)
    s.set_solid_from_scene(True)
    _Fg = (RATIO - 1.0) * (np.pi / 6) * DH ** 3 * GSTAR
    s.set_body_force(0.0, _Fg / (N ** 3), 0.0)   # back-pressure: zero net force on the suspension
    m = RATIO * (np.pi / 6) * DH ** 3
    Vp = (np.pi / 6) * DH ** 3
    ma = 2.0 * Vp
    d = pdem.Simulation(8)
    d.set_gravity(0.0, 0.0, 0.0); d.set_sphere_shape(1.0)
    d.set_positions(np.array([[x0, x0, x0]], dtype=np.float32))
    d.set_inv_mass(np.array([1.0 / (m + ma)], dtype=np.float32))
    d.set_inv_inertia(np.array([[0, 0, 0]], dtype=np.float32))
    Fg = (RATIO - 1.0) * Vp * GSTAR
    a_prev = np.zeros(3); vs = []
    for k in range(nstep):
        p = np.asarray(d.get_positions())[0].astype(float)
        v = np.asarray(d.get_velocities())[0].astype(float)
        s.set_instance_transform(0, p.tolist())
        s.set_instance_motion(0, lin_vel=v.tolist())
        s.rebuild_geometry(); s.step()
        F = np.asarray(s.hydro_force_torque_reaction())[0][0].astype(float)
        F[1] -= Fg
        F += ma * a_prev
        d.set_external_forces(np.array([F], dtype=np.float32))
        for _ in range(10):
            d.step(dt / 10)
        a_prev = (np.asarray(d.get_velocities())[0].astype(float) - v) / dt
        vs.append(-v[1] / USTAR)
        if k > 400 and abs(vs[-1] - vs[-200]) < 5e-4 * abs(vs[-1]):
            break
    c = (np.pi / 6) * DH ** 3 / N ** 3
    hasi = 1.0 - 1.7601 * c ** (1 / 3)
    print("%s unbounded N=%d: plateau=%.4f  Hasimoto-corrected expectation=%.4f  steps=%d"
          % (case, N, vs[-1], hasi, len(vs)))
    return vs

vs = run_unbounded("E1", 8)
