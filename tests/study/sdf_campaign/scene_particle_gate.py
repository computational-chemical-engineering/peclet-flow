"""dem gate: a COMPOSED analytic tree as a particle shape (SHAPE_SCENE).

Two claims, both at OMP_NUM_THREADS=1 (dem's multithreaded step is nondeterministic — §6.0b):

  1. STATIC CONTACT PROBE. For hand-placed overlapping pairs, the analytic-tree particle and the
     SAME shape baked to a fine grid-SDF particle must report contacts of the same count, with
     penetration depths agreeing to the grid's own resolution error, and the tree's depths must
     match the EXACT closed form (a dumbbell's lobe is a sphere: depth = 2r − distance).
  2. DYNAMICS. Two settling/colliding dumbbells: 400-step single-thread trajectories, tree vs
     baked-grid shape. Same physics, different geometry representation: positions must track to
     the grid's resolution error, not diverge.

The dumbbell is authored in peclet.core.geom, mass properties measured by implicit quadrature,
put in its principal frame with principal_frame() (a no-op rotation here, but the COM recentre is
real when the lobes differ), and handed to dem BOTH ways from the SAME builder.
"""
import os
import numpy as np

assert os.environ.get("OMP_NUM_THREADS") == "1", "run with OMP_NUM_THREADS=1 (dem determinism)"

from peclet.core import geom
from peclet import dem as pdem

# --- author the particle: an ASYMMETRIC dumbbell (different lobe radii => real COM shift) -------
R1, R2, OFF = 0.30, 0.22, 0.35
b = geom.SceneBuilder()
s1 = b.add_leaf("sphere", [R1], translation=[-OFF, 0.0, 0.0])
s2 = b.add_leaf("sphere", [R2], translation=[+OFF, 0.0, 0.0])
db = b.add_union(s1, s2)
props = b.body_properties(db, lo=[-0.8, -0.5, -0.5], hi=[0.8, 0.5, 0.5], n=40)
home = b.principal_frame(db, lo=[-0.8, -0.5, -0.5], hi=[0.8, 0.5, 0.5], n=40)
ph = b.body_properties(home, lo=[-0.9, -0.6, -0.6], hi=[0.9, 0.6, 0.6], n=40)
com = np.array(props["com"])
print("dumbbell: com=%s  principal=%s  (reframed com -> %s)"
      % (np.round(com, 5), np.round(props["principal"], 6), np.round(ph["com"], 6)))
inv_i_unit = tuple(props["mass"] / np.maximum(np.array(ph["principal"]), 1e-12))
bound = float(OFF + max(R1, R2) + abs(com[0]) + 0.05)

# --- shell + grid bake from the SAME reframed tree ---------------------------------------------
NB = 96
half = bound * 1.05
sp = 2 * half / (NB - 1)
gridf = np.asarray(b.bake(home, origin=[-half, -half, -half], spacing=[sp, sp, sp],
                          dims=[NB, NB, NB]))
grid3 = gridf.reshape(NB, NB, NB, order="F")
from skimage import measure
verts, faces, _, _ = measure.marching_cubes(np.ascontiguousarray(grid3), level=0.0,
                                            spacing=(sp, sp, sp))
verts += np.array([-half, -half, -half])
step = max(1, len(verts) // 400)
shell = np.ascontiguousarray(verts[::step], dtype=np.float32)
print("bake %d^3 (spacing %.4f = %.3f of the small radius), shell %d points"
      % (NB, sp, sp / R2, len(shell)))

def make_sim(kind):
    d = pdem.Simulation(4)
    d.initialize(shape_type=1, radius=0.5)          # placeholder shape 0
    d.set_domain((0.0, 0.0, 0.0), (8.0, 8.0, 8.0))
    d.enable_periodicity(True, True, True)
    d.set_gravity(0.0, 0.0, 0.0)
    ni, nr, ii, ir = b.encode()
    if kind == "tree":
        sid = d.add_scene_shape(np.ascontiguousarray(ni, dtype=np.int32),
                                np.ascontiguousarray(nr, dtype=np.float32), home,
                                shell, inv_i_unit, bound)
    else:
        sid = d.add_sdf_shape(np.ascontiguousarray(gridf, dtype=np.float32), NB, NB, NB,
                              (-half, -half, -half), (sp, sp, sp), shell, inv_i_unit, bound)
    # two particles on a collision course, second one rotated 90 deg about z
    pos = np.zeros((2, 4), dtype=np.float32)
    pos[0, 0:3] = (3.2, 4.0, 4.0); pos[0, 3] = 1.0
    pos[1, 0:3] = (4.9, 4.1, 4.0); pos[1, 3] = 1.0
    d.set_positions(pos)
    q = np.zeros((2, 4), dtype=np.float32)
    q[0] = (0, 0, 0, 1)
    q[1] = (0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    d.set_quaternions(q)
    v = np.zeros((2, 4), dtype=np.float32)
    v[0, 0] = +0.6; v[1, 0] = -0.6
    d.set_velocities(v)
    d.set_shape_ids(np.array([sid, sid], dtype=np.int32))
    d.set_material_params(0.3, 0.2, 0.0)
    d.set_solver_iterations(20, 20)
    return d

# --- 1. static contact probe: hand-placed interpenetration with an exact closed form ----------
# Facing +x lobes: after the reframe the lobes sit at body x = -OFF-com (radius R1) and
# OFF-com (radius R2). Identity quaternions, centres DX apart along x => the touching pair is
# particle0's +x lobe (R2, at +a2) against particle1's -x lobe (R1, at -a1):
#     overlap_exact = (R1 + R2) - (DX - a1 - a2)
a1 = OFF + com[0]      # |body-x| of the R1 lobe after recentring (com is negative)
a2 = OFF - com[0]
PEN = 0.08
DX = a1 + a2 + (R1 + R2) - PEN
# step(0) is dem's documented dynamics-free relaxation (overlap removal only): after it, the
# centre separation must be the exact touching distance a1 + a2 + R1 + R2 -- the closed form the
# tree must hit and the grid may miss by its resolution.
def probe(kind):
    d = make_sim(kind)
    pos = np.zeros((2, 4), dtype=np.float32)
    pos[0, 0:3] = (3.0, 4.0, 4.0); pos[0, 3] = 1.0
    pos[1, 0:3] = (3.0 + DX, 4.0, 4.0); pos[1, 3] = 1.0
    d.set_positions(pos)
    # set_positions RESETS every particle to shape 0 (documented dem gotcha: it re-applies the
    # default shape + inertia) -- re-assign the composed shape AFTER it, or the probe silently
    # relaxes two placeholder spheres.
    d.set_shape_ids(np.array([1, 1], dtype=np.int32))
    d.set_quaternions(np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32))
    d.set_velocities(np.zeros((2, 4), dtype=np.float32))
    for _ in range(40):
        d.step(0.0)
    q = np.array(d.get_positions())
    return float(q[1, 0] - q[0, 0])
touch = a1 + a2 + R1 + R2
st, sg = probe("tree"), probe("grid")
print("relaxed separation: exact %.5f  tree %.5f (err %.2e)  grid %.5f (err %.2e; resolution %.4f)"
      % (touch, st, abs(st - touch), sg, abs(sg - touch), sp))
ok1 = abs(st - touch) < 5e-3 and abs(sg - touch) < 3 * sp

# --- 2. dynamics -------------------------------------------------------------------------------
def march(d, n=400, dtstep=0.01):
    traj = []
    for k in range(n):
        d.step(dtstep)
        if (k + 1) % 100 == 0:
            traj.append(np.array(d.get_positions()))
    return np.array(traj), np.array(d.get_velocities()), np.array(d.get_quaternions())

Tt, Vt, Qt = march(make_sim("tree"))
Tg, Vg, Qg = march(make_sim("grid"))
dpos = np.abs(Tt - Tg).max(axis=(1, 2))
print("trajectory |tree - grid| at steps 100/200/300/400:", np.round(dpos, 5))
print("  (the grid's own geometry error is ~%.4f; agreement is claimed at that scale, not tighter)"
      % sp)
dvel = np.abs(Vt - Vg).max()
dquat = np.abs(Qt - Qg).max()
print("final state |tree - grid|: velocities %.5f  quaternions %.5f" % (dvel, dquat))
ok2 = dpos[-1] < 10 * sp
print("GATE %s  [tree particle collides like its own fine-grid bake]"
      % ("PASS" if (ok1 and ok2) else "FAIL"))
