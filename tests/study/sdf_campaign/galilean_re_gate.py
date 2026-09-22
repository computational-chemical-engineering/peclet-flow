"""Finite-Re Galilean gate: a sphere FIXED in a uniform stream vs the SAME sphere TOWED through
fluid at rest -- same box, same Re, same dt, same impulsive start, same body force. In a periodic
box a uniform translation is an exact symmetry of the Navier-Stokes equations, so the two force
histories must agree to discretisation error at EVERY step. The fixed case never exercises the
moving-geometry machinery (fresh cells, wall-velocity advection inputs, the time term at a moving
cut wall, rebuild per step); the towed case exercises all of it. Any gap is a moving-geometry
defect, isolated from resolution and from confinement.

Usage: PYTHONPATH=<flow build> python galilean_re_gate.py [DH=8] [Re=1.5] [N=96] [STEPS=800]
Env:   GAL_ADV=0 turns advection off (the Stokes limit, where the old gate lives).
"""
import os
import sys
import numpy as np
from peclet import flow as sdflow
from peclet.core import geom

DH = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
RE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
N = int(sys.argv[3]) if len(sys.argv) > 3 else 96
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 800
ADV = os.environ.get("GAL_ADV", "1") == "1"
U = 0.02                        # stream / tow speed (cells per time unit)
NU = U * DH / RE
DT = 3.2
KI_I, KI_R = 2, 17
OFF = 0.3


def abraham_cd(re):
    return 24.0 / 9.06 ** 2 * (9.06 / np.sqrt(re) + 1.0) ** 2


def make(y0):
    b = geom.SceneBuilder()
    sph = b.add_leaf("sphere", [DH / 2])
    ni, nr, _, _ = b.encode()
    ii = np.zeros((1, KI_I), dtype=np.int32); ir = np.zeros((1, KI_R))
    x0 = 0.5 * N + OFF
    ii[0] = (sph, -1); ir[0, 0:3] = (x0, y0, x0); ir[0, 6] = 1.0; ir[0, 7] = 1.0
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(DT); s.set_advection(ADV)
    s.set_velocity_solver_params(60); s.set_pressure_solver_params(20)
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(np.asarray(ni, np.int32), np.asarray(nr, float), ii.ravel(), ir.ravel(),
                periodic=True)
    s.set_solid_from_scene(True)
    return s, x0


# Body force that keeps the box mean flow steady: the Abraham drag at U, per unit volume. The
# same value in BOTH runs (a body force is Galilean-invariant); its residual mismatch drifts the
# mean by (F - B V) t / (rho V) -- measured and reported, not assumed.
F_EST = abraham_cd(RE) * 0.5 * U ** 2 * np.pi * (DH / 2) ** 2
B = F_EST / N ** 3


def run(moving):
    y0 = 0.72 * N if moving else 0.5 * N + OFF
    s, x0 = make(y0)
    s.set_body_force(0.0, B, 0.0)
    if not moving:
        v0 = np.full((N, N, N), U, dtype=np.float64, order="F")
        s.set_velocity(1, v0)
    y = y0; Fh = []; vm = []
    for k in range(STEPS):
        if moving:
            s.set_instance_transform(0, [x0, y, x0])
            s.set_instance_motion(0, lin_vel=[0.0, -U, 0.0])
            s.rebuild_geometry()
        s.step()
        F = np.asarray(s.hydro_force_torque_reaction())[0][0].astype(float)
        Fh.append(F[1])
        if k % 50 == 49 or k == STEPS - 1:
            vmean = float(np.asarray(s.get_v()).mean())
            vm.append((k + 1, vmean))
        if moving:
            y -= U * DT
    bud = np.asarray(s.reaction_budget_terms()).astype(float)
    return np.array(Fh), vm, bud


print("Finite-Re Galilean gate: DH=%g Re=%g N=%d dt=%g steps=%d adv=%s  (B=%.3e, F_est=%.4e)"
      % (DH, RE, N, DT, STEPS, ADV, B, F_EST))
FS, vS, bS = run(False)
FM, vM, bM = run(True)
# Mean fluid velocity: fixed case should hold ~U, towed case ~0 (both drift by the same B mismatch).
print("  mean v (fixed) : " + "  ".join("%d:%.5f" % t for t in vS[::4]))
print("  mean v (towed) : " + "  ".join("%d:%.5f" % t for t in vM[::4]))
n4 = STEPS * 3 // 4
fS, fM = FS[n4:].mean(), FM[n4:].mean()
uS = vS[-1][1] / (1.0 - np.pi / 6 * DH ** 3 / N ** 3)   # fluid-mean (solid rows are 0)
uM = U + vM[-1][1] / (1.0 - np.pi / 6 * DH ** 3 / N ** 3)  # relative speed in the towed case
print("  quasi-steady F_y (last quarter):  fixed %.6e   towed %.6e   towed/fixed = %.4f"
      % (fS, fM, fM / fS))
print("  step-by-step |F_M - F_S| / |F_S|:  " + "  ".join(
    "k=%d:%.3f" % (k, abs(FM[k] - FS[k]) / abs(FS[k])) for k in (10, 50, 100, 200, 400, STEPS - 1)))
for tag, f, u in (("fixed", fS, uS), ("towed", fM, uM)):
    re = u * DH / NU
    cd = f / (0.5 * u ** 2 * np.pi * (DH / 2) ** 2)
    print("  %s: u_rel=%.5f Re=%.3f  Cd=%.4f  Abraham(Re)=%.4f  ratio=%.4f"
          % (tag, u, re, cd, abraham_cd(re), cd / abraham_cd(re)))
print("  budget [unsteady_y, advective_y]: fixed [%.3e, %.3e]  towed [%.3e, %.3e]"
      % (bS[0][1], bS[1][1], bM[0][1], bM[1][1]))
gap = abs(fM / fS - 1.0)
print("GALILEAN-RE GATE %s  [towed/fixed quasi-steady drag within 2%%: %.2f%%]"
      % ("PASS" if gap < 0.02 else "FAIL", 100 * gap))
