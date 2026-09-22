"""Layer 2: the in-solver analytic scene must reproduce the geometry flow has always been handed
as a sampled array -- on a CONVERGED periodic sphere packing, not a diverged one."""
import numpy as np
from peclet import flow as sdflow

N = 32
R = 0.18 * N
CEN = [(0.25*N, 0.25*N, 0.25*N), (0.75*N, 0.75*N, 0.25*N),
       (0.75*N, 0.25*N, 0.75*N), (0.25*N, 0.75*N, 0.75*N)]

# Reference: exactly the SDF the verify script uses (min-image periodic).
g = np.arange(N) + 0.5
X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
ref = np.full((N, N, N), 1e30)
for (cx, cy, cz) in CEN:
    dx = np.abs(X-cx); dx = np.minimum(dx, N-dx)
    dy = np.abs(Y-cy); dy = np.minimum(dy, N-dy)
    dz = np.abs(Z-cz); dz = np.minimum(dz, N-dz)
    ref = np.minimum(ref, np.sqrt(dx*dx+dy*dy+dz*dz) - R)

# Scene: PERIODICITY IS THE CALLER'S -- instantiate the 27 images of each sphere so the analytic
# min over instances reproduces the min-image distance. Cell centre (i,j,k) sits at (i,j,k), and
# the numpy grid puts it at i+0.5, so instance centres are shifted by -0.5.
KN_R, KI_I, KI_R = 16, 2, 17
node_ints  = np.array([1, -1, -1], dtype=np.int32)          # kSphere
node_reals = np.zeros(KN_R); node_reals[0] = R
node_reals[14] = 1.0   # quaternion w
node_reals[15] = 1.0   # scale
inst = []
for (cx, cy, cz) in CEN:
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                inst.append((cx - 0.5 + ix*N, cy - 0.5 + iy*N, cz - 0.5 + iz*N))
inst_ints  = np.zeros((len(inst), KI_I), dtype=np.int32)
inst_reals = np.zeros((len(inst), KI_R))
for m, (cx, cy, cz) in enumerate(inst):
    inst_ints[m] = (0, -1)
    inst_reals[m, 0:3] = (cx, cy, cz)
    inst_reals[m, 6] = 1.0
    inst_reals[m, 7] = 1.0
print("  scene: 1 node, %d instances (4 spheres x 27 images)" % len(inst))

def run(setup):
    s = sdflow.Solver(N, N, N)
    # rho/mu/dt BEFORE the geometry: set_solid builds the momentum operator from them. Setting
    # them afterwards leaves it built on defaults and the run diverges (~1e35).
    s.set_rho(1.0); s.set_mu(1.0); s.set_dt(50.0)
    setup(s)
    s.set_body_force(1.0, 0.0, 0.0)
    for _ in range(40): s.step()
    return s, np.asarray(s.get_u())

s1, u1 = run(lambda s: (s.set_scene(node_ints, node_reals, inst_ints.ravel(), inst_reals.ravel()),
                        s.set_solid_from_scene(True)))
s2, u2 = run(lambda s: s.set_solid(np.asfortranarray(ref), cutcell_pressure=True))

# NATIVE periodic scene: 4 instances, no images, periodic=True — must agree with both
base_ints  = np.zeros((len(CEN), KI_I), dtype=np.int32)
base_reals = np.zeros((len(CEN), KI_R))
for m, (cx, cy, cz) in enumerate(CEN):
    base_ints[m] = (0, -1)
    base_reals[m, 0:3] = (cx - 0.5, cy - 0.5, cz - 0.5)
    base_reals[m, 6] = 1.0
    base_reals[m, 7] = 1.0
s4, u4 = run(lambda s: (s.set_scene(node_ints, node_reals, base_ints.ravel(), base_reals.ravel(),
                                    periodic=True),
                        s.set_solid_from_scene(True)))

den = max(np.abs(u2).max(), 1e-300)
rel = np.abs(u1-u2).max()/den
print("  CONVERGED? u_absmax array = %.6e  (a diverged run is ~1e35)" % np.abs(u2).max())
print("  108-image scene vs sampled array: max rel diff = %.3e" % rel)
rel4 = np.abs(u4-u2).max()/den
print("  NATIVE periodic (4 inst)  vs array: max rel diff = %.3e" % rel4)
ok = rel < 1e-12 and rel4 < 1e-12 and np.abs(u2).max() < 1e3

s3, u3 = run(lambda s: (s.set_scene(node_ints, node_reals, base_ints.ravel(), base_reals.ravel(),
                                    periodic=True),
                        s.set_exact_crossings_from_scene(),
                        s.set_solid_from_scene(True)))
print("  with EXACT scene crossings: u_absmax %.6e (vs %.6e)" % (np.abs(u3).max(), np.abs(u1).max()))
print("    changed the solution: %s   finite: %s" %
      (not np.array_equal(u3, u1), bool(np.isfinite(u3).all())))
print("SCENE OK" if ok and np.isfinite(u3).all() and np.abs(u3).max() < 1e3 else "SCENE FAIL")
