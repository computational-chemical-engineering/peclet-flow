"""The lattice-plane trap, gated: a MOVING box whose faces sit exactly on grid planes owns no cut
cell, so its wall velocity never enters the momentum operator and the body is silently inert.
set_solid_from_scene must (a) count > 0 DEGENERATE staggered points (sdf exactly zero) for that
instance and warn, and (b) count 0 for the same box shifted 0.3 cells off the lattice -- and the
fluid must actually move in (b).

Usage: PYTHONPATH=<flow build> python lattice_plane_gate.py
"""
import numpy as np
from peclet import flow as sdflow
from peclet.core import geom

N = 32
U = 0.05
KI_I, KI_R = 2, 17


def run(half):
    b = geom.SceneBuilder()
    plate = b.add_leaf("box", [N * 0.6, half, N * 0.6])       # spans the box in x,z (a legitimate union)
    ni, nr, _, _ = b.encode()
    ii = np.zeros((1, KI_I), dtype=np.int32); ir = np.zeros((1, KI_R))
    ii[0] = (plate, -1); ir[0, 0:3] = (0.5 * N, 8.0, 0.5 * N); ir[0, 6] = 1.0; ir[0, 7] = 1.0
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(2.0)
    s.set_velocity_solver_params(60); s.set_pressure_solver_params(20)
    s.set_pressure_multigrid(True, levels=3)
    s.set_scene(np.asarray(ni, np.int32), np.asarray(nr, float), ii.ravel(), ir.ravel(),
                periodic=True)
    s.set_instance_motion(0, lin_vel=[U, 0.0, 0.0])           # a plate sliding in x
    s.set_solid_from_scene(True)
    deg = np.asarray(s.moving_instance_degenerate_points())
    for _ in range(40):
        s.step()
    umax = float(np.abs(np.asarray(s.get_u())).max())
    return int(deg[0]) if deg.size else -1, umax


ok = True
for tag, half, want_deg in (("faces ON grid planes (y = 16.0)", 8.0, True),
                            ("faces off the lattice (y = 16.3)", 8.3, False)):
    deg, umax = run(half)
    good = (deg > 0) == want_deg and ((umax > 0.1 * U) != want_deg)
    ok &= good
    print("  %-34s degenerate points %6d   max|u| after 40 steps %.3e (wall U=%.2f)  %s"
          % (tag, deg, umax, U, "ok" if good else "FAIL"))
print("LATTICE-PLANE GATE %s" % ("PASS" if ok else "FAIL"))
