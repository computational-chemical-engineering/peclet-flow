"""Centre of rotation: NaN follows the body, any finite point pins -- the world origin included.

Semantics gated (§7 item 3):
  (a) builder default (NaN) -> centre = the translation, instance_center_pinned False;
  (b) builder center=(0,0,0) -> PINNED at the world origin (the case zero-as-unset made impossible);
  (c) raw array with zeros -> legacy 'follows the body' (every shipped page);
  (d) set_instance_motion(center=...) -> pinned, and it STAYS pinned when the body is moved by
      set_instance_transform (the old float comparison re-tracked it whenever the body passed
      through its own pin).
Physics gated: a sphere spinning about a point a distance R_p from its centre translates at
U = |w x R_p| at that instant, so its net reaction force must match the drag of the same sphere
towed at U through the same box (Stokes, advection off); a sphere spinning about its OWN centre
has no net force beyond the discretisation asymmetry of an off-lattice sphere (< 1% of the pivot force).

Usage: PYTHONPATH=<flow build>:<core geom build> python centre_of_rotation_gate.py
"""
import numpy as np
from peclet import flow as sdflow
from peclet.core import geom

N, R = 64, 6.4
MU, DT = 1.0, 2.0          # nu dt/h^2 = 2: the 100-sweep RB-GS converges to round-off (at dt=100 it leaves a 1e-4 net force)
OMEGA = 1e-3
KI_I, KI_R = 2, 17
x0 = 0.5 * N + 0.3
CEN = (x0, x0, x0)


def solver(ni, nr, ii, ir):
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(MU); s.set_dt(DT); s.set_advection(False)
    s.set_velocity_solver_params(100); s.set_pressure_solver_params(25)
    if hasattr(s, "set_velocity_residual_tolerance"):
        s.set_velocity_residual_tolerance(0.0)
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(np.asarray(ni, np.int32), np.asarray(nr, float), np.asarray(ii).ravel(),
                np.asarray(ir).ravel(), periodic=True)
    return s


def builder_scene(**kw):
    b = geom.SceneBuilder()
    sph = b.add_leaf("sphere", [R])
    b.add_instance(sph, translation=list(CEN), **kw)
    return b.encode()


def raw_scene():
    b = geom.SceneBuilder()
    sph = b.add_leaf("sphere", [R])
    ni, nr, _, _ = b.encode()
    ii = np.zeros((1, KI_I), dtype=np.int32); ir = np.zeros((1, KI_R))
    ii[0] = (sph, -1); ir[0, 0:3] = CEN; ir[0, 6] = 1.0; ir[0, 7] = 1.0
    return ni, nr, ii, ir


def net_force(s, steps=300):
    s.set_solid_from_scene(True)
    for _ in range(steps):
        s.step()
    return np.asarray(s.hydro_force_torque_reaction())[0][0].astype(float)


ok = True
def check(cond, msg):
    global ok
    ok &= bool(cond)
    print("  %-70s %s" % (msg, "ok" if cond else "FAIL"))

# (a) builder default: follows the body
s = solver(*builder_scene(ang_vel=[0, 0, OMEGA]))
c = np.asarray(s.instance_center(0))
check(np.allclose(c, CEN) and not s.instance_center_pinned(0), "(a) builder NaN default -> centre = translation, not pinned")
# (b) builder explicit world origin: pinned at (0,0,0)
s = solver(*builder_scene(ang_vel=[0, 0, OMEGA], center=[0.0, 0.0, 0.0]))
c = np.asarray(s.instance_center(0))
check(np.allclose(c, 0.0) and s.instance_center_pinned(0), "(b) builder center=(0,0,0) -> pinned at the world ORIGIN")
# (c) raw zeros: legacy follows the body
s = solver(*raw_scene())
s.set_instance_motion(0, ang_vel=[0, 0, OMEGA])
c = np.asarray(s.instance_center(0))
check(np.allclose(c, CEN) and not s.instance_center_pinned(0), "(c) raw array zeros -> legacy: follows the body")
# (d) explicit setter pins and survives a transform through the pin
s = solver(*raw_scene())
s.set_instance_motion(0, ang_vel=[0, 0, OMEGA], center=[x0, x0, x0])     # pin AT the current centre
s.set_instance_transform(0, [x0 + 5.0, x0, x0])                              # move the body off it
c = np.asarray(s.instance_center(0))
check(np.allclose(c, CEN) and s.instance_center_pinned(0), "(d) setter pin stays put when the body moves off it")

# Physics: spin about the body's centre -> zero net force; spin about a pivot -> towing drag
RP = 8.0
s = solver(*raw_scene()); s.set_instance_motion(0, ang_vel=[0, 0, OMEGA])
F0 = net_force(s)
s = solver(*raw_scene()); s.set_instance_motion(0, ang_vel=[0, 0, OMEGA], center=[x0 - RP, x0, x0])
Fp = net_force(s)
U = OMEGA * RP                                        # instantaneous translation speed, along +y
s = solver(*raw_scene()); s.set_instance_motion(0, lin_vel=[0, U, 0])
Ft = net_force(s)
# A sphere centred OFF the lattice (+0.3, as every page places it) carries a small
# discretisation-asymmetry force when it spins (9e-3 mu R^2 Om at R/h = 6.4, 1.2e-3 at 9.6, and
# 5e-13 for a lattice-centred sphere -- rotation_gate.py); it is identical for a tracked and an
# explicit centre, so the gate asks only that it be negligible against the pivot force.
rel = np.linalg.norm(F0) / np.linalg.norm(Fp)
print("  |F| spin about own centre / |F| about the pivot = %.2e   (discretisation asymmetry of an off-lattice sphere)" % rel)
print("  F_y spin about pivot = %.5e   F_y towed at U = %.5e   ratio = %.4f" % (Fp[1], Ft[1], Fp[1] / Ft[1]))
check(rel < 1e-2, "spin about own centre: net force < 1% of the pivot force")
check(abs(Fp[1] / Ft[1] - 1.0) < 0.05 and abs(Fp[0]) < 0.05 * abs(Fp[1]), "spin about a pivot: net force = towing drag at |w x R_p| (5%)")
print("CENTRE-OF-ROTATION GATE %s" % ("PASS" if ok else "FAIL"))
