"""Regression for the contact-buffer capacity bug (found via peclet-examples/pall-ring-packing).

TWO independent symptoms of one cause -- every maxContacts-sized collision buffer was sized from
the particle capacity AT THE TIME a shape or wall was registered, while `demStep` grows the
particle capacity every step (ghost headroom = numReal + estGhosts + 4096). Buffers frozen at the
smaller size are an out-of-bounds write, and the narrow phase's boundary contacts -- appended AFTER
the body-body ones -- are the first thing to fall off the end.

  A. add_scene_shape never called ensureContactCapacity() at all (every sibling adder does), so a
     composed-tree particle's contacts against an add_plane floor were dropped and the grains fell
     straight through it.
  B. Even with that call, the buffer did not follow the per-step capacity growth.

Claim: composed-tree particles dropped onto a plane come to rest ON it. Run at OMP_NUM_THREADS=1.
"""
import os
import numpy as np

assert os.environ.get("OMP_NUM_THREADS") == "1", "run with OMP_NUM_THREADS=1"

from peclet.core import geom
from peclet.dem import scene_particle
from peclet import dem as pdem

R1, R2, OFF = 0.30, 0.22, 0.35
b = geom.SceneBuilder()
db = b.add_union(b.add_leaf("sphere", [R1], translation=[-OFF, 0, 0]),
                 b.add_leaf("sphere", [R2], translation=[+OFF, 0, 0]))
sp = scene_particle.build(b, db, bounds=([-0.8, -0.5, -0.5], [0.8, 0.5, 0.5]),
                          n=40, shell_resolution=96, target_shell_points=400)

N = 6
s = pdem.Simulation(64)                      # deliberately a SMALL construction capacity
s.set_gravity(0.0, -9.81, 0.0)
s.set_global_scale(1.0)
sid = sp.register(s)
s.add_plane(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)    # floor at y = 0, normal +y
s.set_material_params(0.2, 0.0, 0.5)
pos = np.array([[1.4 * i - 3.5, 2.0 + 0.3 * i, 0.0] for i in range(N)], dtype=np.float32)
s.set_positions(pos)
s.set_shape_ids(np.full(N, sid, dtype=np.int32))     # AFTER set_positions
s.set_inv_mass(np.full(N, 1.0 / sp.mass, dtype=np.float32))
s.set_inv_inertia(np.tile(np.asarray(sp.inv_inertia_unit, dtype=np.float32), (N, 1)))
s.set_dt(2e-3)
for _ in range(2500):
    s.step(2e-3)
p = np.asarray(s.get_positions())
ymin = float(p[:, 1].min())
lo = float(sp.shell[:, 1].min())     # the shell's lowest point in the body frame
print("shell lowest point (body frame) %.4f   -> a resting centre cannot go below ~%.3f"
      % (lo, -lo))
print("lowest particle centre after 2500 steps: %.4f" % ymin)
print("max overlap %.4f   contacts %d" % (s.max_overlap(), s.num_contacts()))
ok = ymin > 0.5 * (-lo)
print("GATE %s  [composed-tree particles rest ON the plane, not through it]"
      % ("PASS" if ok else "FAIL"))
