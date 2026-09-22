"""Periodic images are a UNION -- the oversized-leaf trap, gated.

A container wall built as slab-minus-cavity in a periodic box: if the slab is wider than the box,
its periodic images overlap and refill the cavity wherever a neighbouring image's slab covers it
(the query takes the min over images). set_solid_from_scene must (a) DETECT it exactly --
periodic_image_overlap_cells() > 0 plus a stderr warning -- for a 0.7 L slab, and (b) report 0
for a slab of half the box plus the wall thickness, whose cavity must then be the full 53 cells.

Usage: PYTHONPATH=<flow build> python periodic_image_gate.py
"""
import numpy as np
from peclet import flow as sdflow
from peclet.core import geom

L, NY, U = 64, 32, 0.02
KI_I, KI_R = 2, 17
OFF, WX = 0.3, 5.335
HW = (L - 2 * WX) / 2
x0 = 0.5 * L + OFF


def cavity_width(slab_half):
    b = geom.SceneBuilder()
    slab = b.add_leaf("box", [slab_half, NY * 2.0, slab_half])
    cav = b.add_leaf("box", [HW, NY * 2.0, HW])
    node = b.add_difference(slab, cav)
    ni, nr, _, _ = b.encode()
    ii = np.zeros((1, KI_I), dtype=np.int32); ir = np.zeros((1, KI_R))
    ii[0] = (node, -1); ir[0, 0:3] = (x0, 0.5 * NY, x0); ir[0, 6] = 1.0; ir[0, 7] = 1.0
    s = sdflow.Solver(L, NY, L)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(1.0)
    s.set_scene(np.asarray(ni, np.int32), np.asarray(nr, float), ii.ravel(), ir.ravel(),
                periodic=True)
    s.set_solid_from_scene(True)
    s.set_velocity(1, np.full((L, NY, L), U, dtype=np.float64, order="F"))
    v = np.asarray(s.get_v()).reshape((L, NY, L), order="F")
    cols = np.where((np.abs(v) > 0).sum(axis=(1, 2)) > 0.5 * NY * L)[0]
    return len(cols), s.periodic_image_overlap_cells()


ok = True
for tag, half, want_overlap, want_w in (("0.70 L slab (WIDER than the box)", 0.70 * L, True, 38),
                                        ("L/2 + 1 slab (half box + wall)", L / 2 + 1.0, False, 53)):
    w, n = cavity_width(half)
    good = (n > 0) == want_overlap and w == want_w
    ok &= good
    print("  %-34s cavity %2d cols (expect %2d)  overlap cells %7d  %s"
          % (tag, w, want_w, n, "ok" if good else "FAIL"))
print("PERIODIC-IMAGE GATE %s" % ("PASS" if ok else "FAIL"))
