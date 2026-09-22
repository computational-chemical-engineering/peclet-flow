"""Layer 3 rung 1 gate (flow side): every CUT cell's owner is the brute-force nearest sphere.

A cut cell is a fluid cell (sdf>0) with at least one solid face-neighbour -- the cells whose
stencils the IBM overlay modifies, and therefore the only cells whose ownership a moving wall
velocity (R2) or a CFD-DEM force integration (L4-R2) ever reads.

Coordinates: the solver samples the scene at INTEGER cell coordinates (cell (i,j,k) at (i,j,k)),
so the reference lattice here is arange(N) -- not the i+0.5 convention scene_check.py uses for
its own sampled-array reference.
"""
import numpy as np
from peclet import flow as sdflow

N = 32
R = 0.18 * N
CEN = [(0.25*N, 0.25*N, 0.25*N), (0.75*N, 0.75*N, 0.25*N),
       (0.75*N, 0.25*N, 0.75*N), (0.25*N, 0.75*N, 0.75*N)]

KN_R, KI_I, KI_R = 16, 2, 17
node_ints  = np.array([1, -1, -1], dtype=np.int32)          # kSphere
node_reals = np.zeros(KN_R); node_reals[0] = R
node_reals[14] = 1.0; node_reals[15] = 1.0                  # quat w, scale
inst_ints  = np.zeros((len(CEN), KI_I), dtype=np.int32)
inst_reals = np.zeros((len(CEN), KI_R))
for m, c in enumerate(CEN):
    inst_ints[m] = (0, -1)
    inst_reals[m, 0:3] = c
    inst_reals[m, 6] = 1.0   # quat w
    inst_reals[m, 7] = 1.0   # scale

s = sdflow.Solver(N, N, N)
s.set_rho(1.0); s.set_mu(1.0); s.set_dt(50.0)
s.set_scene(node_ints, node_reals, inst_ints.ravel(), inst_reals.ravel(), periodic=True)
s.set_solid_from_scene(True)
owner = np.asarray(s.get_cut_owner())

g = np.arange(N).astype(float)
X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
d = np.empty((len(CEN), N, N, N))
for k, (cx, cy, cz) in enumerate(CEN):
    dx = np.abs(X-cx); dx = np.minimum(dx, N-dx)
    dy = np.abs(Y-cy); dy = np.minimum(dy, N-dy)
    dz = np.abs(Z-cz); dz = np.minimum(dz, N-dz)
    d[k] = np.sqrt(dx*dx+dy*dy+dz*dz) - R
ref_owner = np.argmin(d, axis=0)          # numpy argmin: lowest index wins ties, same rule
ref_sdf = d.min(axis=0)

fluid = ref_sdf > 0.0
solid = ~fluid
nbr_solid = np.zeros_like(solid)
for ax in (0, 1, 2):
    for sh in (+1, -1):
        nbr_solid |= np.roll(solid, sh, axis=ax)
cut = fluid & nbr_solid

srt = np.sort(d, axis=0)
tied = int(((srt[1] - srt[0]) == 0.0).sum())
bad_cut = int((owner[cut] != ref_owner[cut]).sum())
bad_all = int((owner != ref_owner).sum())
print("cut cells            %d of %d" % (int(cut.sum()), owner.size))
print("exact ties on lattice %d" % tied)
print("owner mismatches     cut=%d  all-cells=%d" % (bad_cut, bad_all))
print("GATE %s" % ("PASS" if bad_cut == 0 and bad_all == 0 else "FAIL"))
