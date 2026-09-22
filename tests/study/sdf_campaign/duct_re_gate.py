"""Confinement at finite Re, static twin of the ten Cate tank.

Lab frame: tank at rest, fluid at rest, sphere falls at -U. Sphere frame: sphere FIXED, duct walls
translate at +U, far fluid moves at +U -- plug flow, zero wall shear, exactly the quiescent lab
state. No moving geometry (the duct is y-invariant, so its translation is a pure wall-velocity
datum), so the finite-Re Galilean gate's clean bill applies and what is measured here is the
solver's confinement physics alone.

K(Re) = Cd_duct(Re) / Cd_periodic(Re) is the confinement factor. Creeping expectation for a sphere
on the axis of a square duct with d/W = 0.15 (Faxen): K ~ 1.39. At Re = 1.5 with the wall at
3.3 d >> nu/U = 0.67 d the Stokeslet is screened and the experiment implies K ~ 1.05-1.15. A solver
that returns ~1.39 at Re = 1.5 is not screening.

Usage: PYTHONPATH=<flow build> python duct_re_gate.py [DH=8] [Re=1.5] [STEPS=800]
"""
import sys
import numpy as np
from peclet import flow as sdflow
from peclet.core import geom

DH = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
RE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
U = 0.02
NU = U * DH / RE
DT = 3.2 if RE > 0.5 else 8.0
KI_I, KI_R = 2, 17
OFF = 0.3
WALL = 4.3
L = int(np.ceil((100 / 15 * DH + 2 * WALL) / 8) * 8)      # the ten Cate cross-section
WX = (L - 100 / 15 * DH) / 2
NY = int(np.ceil(12 * DH / 8) * 8)                        # periodic streamwise length 12 d


def abraham_cd(re):
    return 24.0 / 9.06 ** 2 * (9.06 / np.sqrt(re) + 1.0) ** 2


def run(duct):
    b = geom.SceneBuilder()
    sph = b.add_leaf("sphere", [DH / 2])
    if duct:
        slab = b.add_leaf("box", [L / 2 + 1.0, NY * 2.0, L / 2 + 1.0])
        cav = b.add_leaf("box", [(L - 2 * WX) / 2, NY * 2.0, (L - 2 * WX) / 2])
        duc = b.add_difference(slab, cav)
    ni, nr, _, _ = b.encode()
    nI = 2 if duct else 1
    ii = np.zeros((nI, KI_I), dtype=np.int32); ir = np.zeros((nI, KI_R))
    x0 = 0.5 * L + OFF
    ii[0] = (sph, -1); ir[0, 0:3] = (x0, 0.5 * NY + OFF, x0); ir[0, 6] = 1.0; ir[0, 7] = 1.0
    if duct:
        ii[1] = (duc, -1); ir[1, 0:3] = (x0, 0.5 * NY, x0); ir[1, 6] = 1.0; ir[1, 7] = 1.0
    s = sdflow.Solver(L, NY, L)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(DT); s.set_advection(True)
    s.set_velocity_solver_params(60); s.set_pressure_solver_params(20)
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(np.asarray(ni, np.int32), np.asarray(nr, float), ii.ravel(), ir.ravel(),
                periodic=True)
    if duct:
        s.set_instance_motion(1, lin_vel=[0.0, U, 0.0])   # walls move WITH the plug: zero shear
    s.set_solid_from_scene(True)
    s.set_velocity(1, np.full((L, NY, L), U, dtype=np.float64, order="F"))
    # Hold the mean: body force = estimated drag / fluid volume (creeping duct: Faxen x Stokes).
    fest = abraham_cd(RE) * 0.5 * U ** 2 * np.pi * (DH / 2) ** 2 * (1.4 if duct else 1.05)
    vfl = ((L - 2 * WX) ** 2 * NY if duct else L * L * NY) - np.pi / 6 * DH ** 3
    import os
    if os.environ.get("DUCT_NOBF", "0") != "1":
        s.set_body_force(0.0, fest / vfl, 0.0)
    v0 = float(np.asarray(s.get_v()).sum())
    print("  [%s] initial sum(v)/vfl = %.5f  (fluid v-points with |v|>0: %d, vfl=%d)"
          % ("duct" if duct else "periodic", v0 / vfl, int((np.abs(np.asarray(s.get_v())) > 0).sum()), int(vfl)))
    Fh = []; vm = []
    for k in range(STEPS):
        s.step()
        if k < 100 and k % 10 == 9:
            print("    k=%3d  momentum fraction %.4f   F_y=%.4e" % (k + 1, float(np.asarray(s.get_v()).sum()) / v0,
                  float(np.asarray(s.hydro_force_torque_reaction())[0][0][1])))
        Fh.append(float(np.asarray(s.hydro_force_torque_reaction())[0][0][1]))
        if k % 100 == 99 or k == STEPS - 1:
            vm.append((k + 1, float(np.asarray(s.get_v()).sum()) / vfl))
    Fh = np.array(Fh)
    n4 = STEPS * 3 // 4
    return Fh[n4:].mean(), Fh, vm, vfl


print("Duct confinement gate: DH=%g Re=%g  duct %dx%dx%d (width %.2f d), periodic ref %dx%dx%d"
      % (DH, RE, L, NY, L, (L - 2 * WX) / DH, L, NY, L))
fP, FP, vP, _ = run(False)
fD, FD, vD, _ = run(True)
for tag, f, vm in (("periodic", fP, vP), ("duct", fD, vD)):
    u = vm[-1][1]
    re = u * DH / NU
    cd = f / (0.5 * u ** 2 * np.pi * (DH / 2) ** 2)
    print("  %-8s F_y=%.5e  fluid-mean v=%.5f (drift %s)  Re=%.3f  Cd=%.4f  Cd/Abraham=%.4f"
          % (tag, f, u, "/".join("%.5f" % t[1] for t in vm[::2]), re, cd, cd / abraham_cd(re)))
print("  quasi-steadiness (last-quarter slope / mean): periodic %.4f  duct %.4f"
      % ((FP[-1] - FP[STEPS * 3 // 4]) / fP, (FD[-1] - FD[STEPS * 3 // 4]) / fD))
K = fD / fP * (vP[-1][1] / vD[-1][1])
print("  K(Re=%g) = Cd_duct / Cd_periodic = %.4f   [creeping Faxen ~1.39; screened experiment ~1.05-1.15]"
      % (RE, K))
