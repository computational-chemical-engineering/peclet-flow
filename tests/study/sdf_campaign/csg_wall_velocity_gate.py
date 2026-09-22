"""Does a MOVING CSG instance carry its wall velocity? A plug flow at U inside a square duct whose
walls translate at +U has zero shear everywhere and must keep its mean velocity EXACTLY; the same
plug against walls at rest loses momentum to the growing wall layers (~45% in 100 steps at these
numbers). Three ducts: a box-DIFFERENCE instance moving, the same at rest (control), and four
plain box-leaf instances moving.

Usage: PYTHONPATH=<flow build> python csg_wall_velocity_gate.py
"""
import numpy as np
from peclet import flow as sdflow
from peclet.core import geom

L, NY = 64, 32
U, NU, DT = 0.02, 0.1067, 3.2
KI_I, KI_R = 2, 17
OFF, WX = 0.3, 5.335
STEPS = 100
HW = (L - 2 * WX) / 2          # cavity half-width


def solver():
    s = sdflow.Solver(L, NY, L)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(DT); s.set_advection(True)
    s.set_velocity_solver_params(60); s.set_pressure_solver_params(20)
    s.set_pressure_multigrid(True, levels=3)
    return s


def run(kind, moving):
    b = geom.SceneBuilder()
    x0 = 0.5 * L + OFF
    if kind == "difference":
        slab = b.add_leaf("box", [L * 0.7, NY * 2.0, L * 0.7])
        cav = b.add_leaf("box", [HW, NY * 2.0, HW])
        node = b.add_difference(slab, cav)
        ni, nr, _, _ = b.encode()
        ii = np.zeros((1, KI_I), dtype=np.int32); ir = np.zeros((1, KI_R))
        ii[0] = (node, -1); ir[0, 0:3] = (x0, 0.5 * NY, x0); ir[0, 6] = 1.0; ir[0, 7] = 1.0
        ninst = 1
    elif kind in ("sphere0+duct1", "duct0+sphere1"):
        slab = b.add_leaf("box", [L * 0.7, NY * 2.0, L * 0.7])
        cav = b.add_leaf("box", [HW, NY * 2.0, HW])
        node = b.add_difference(slab, cav)
        sph = b.add_leaf("sphere", [4.0])
        ni, nr, _, _ = b.encode()
        ii = np.zeros((2, KI_I), dtype=np.int32); ir = np.zeros((2, KI_R))
        d, p = (1, 0) if kind == "sphere0+duct1" else (0, 1)
        ii[d] = (node, -1); ir[d, 0:3] = (x0, 0.5 * NY, x0)
        ii[p] = (sph, -1); ir[p, 0:3] = (x0, 0.5 * NY + OFF, x0)
        ir[:, 6] = 1.0; ir[:, 7] = 1.0
        ninst = 2
        MOVE = [d]          # only the duct moves; the sphere is a fixed obstacle
    else:  # four box leaves: slabs outside the cavity in x and z
        thick = (L - 2 * HW) / 2 + 2.0
        bx = b.add_leaf("box", [thick / 2, NY * 2.0, L * 0.7])
        bz = b.add_leaf("box", [L * 0.7, NY * 2.0, thick / 2])
        ni, nr, _, _ = b.encode()
        ii = np.zeros((4, KI_I), dtype=np.int32); ir = np.zeros((4, KI_R))
        cen = [(x0 - HW - thick / 2, bx), (x0 + HW + thick / 2, bx)]
        ii[0] = (bx, -1); ir[0, 0:3] = (x0 - HW - thick / 2, 0.5 * NY, x0)
        ii[1] = (bx, -1); ir[1, 0:3] = (x0 + HW + thick / 2, 0.5 * NY, x0)
        ii[2] = (bz, -1); ir[2, 0:3] = (x0, 0.5 * NY, x0 - HW - thick / 2)
        ii[3] = (bz, -1); ir[3, 0:3] = (x0, 0.5 * NY, x0 + HW + thick / 2)
        for i in range(4):
            ir[i, 6] = 1.0; ir[i, 7] = 1.0
        ninst = 4
    s = solver()
    s.set_scene(np.asarray(ni, np.int32), np.asarray(nr, float), ii.ravel(), ir.ravel(),
                periodic=True)
    if moving:
        for i in (MOVE if kind in ("sphere0+duct1", "duct0+sphere1") else range(ninst)):
            s.set_instance_motion(i, lin_vel=[0.0, U, 0.0])
    s.set_solid_from_scene(True)
    s.set_velocity(1, np.full((L, NY, L), U, dtype=np.float64, order="F"))
    v0 = float(np.asarray(s.get_v()).sum())
    out = []
    for k in range(STEPS):
        s.step()
        if k in (0, 9, 49, STEPS - 1):
            out.append((k + 1, float(np.asarray(s.get_v()).sum()) / v0))
    return out


import sys
CASES = (("sphere0+duct1", True), ("duct0+sphere1", True)) if len(sys.argv) > 1 else (("difference", True), ("difference", False), ("four-box", True))
for kind, moving in CASES:
    r = run(kind, moving)
    print("  %-11s walls %-6s : momentum fraction  " % (kind, "moving" if moving else "AT REST")
          + "  ".join("k=%d:%.4f" % t for t in r))
