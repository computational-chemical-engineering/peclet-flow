# Full-precision physics fingerprint of the flow solver, for a before/after comparison of the
# set_solid refactor. Prints enough digits that a 1-ULP change is visible.
import numpy as np, sys
from peclet import flow as sdflow

def packing_sdf(N, radius_frac=0.18):
    R = radius_frac * N
    c = [(0.25*N, 0.25*N, 0.25*N), (0.75*N, 0.75*N, 0.25*N),
         (0.75*N, 0.25*N, 0.75*N), (0.25*N, 0.75*N, 0.75*N)]
    g = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    sdf = np.full((N, N, N), 1e30)
    for (cx, cy, cz) in c:
        dx = np.abs(X-cx); dx = np.minimum(dx, N-dx)
        dy = np.abs(Y-cy); dy = np.minimum(dy, N-dy)
        dz = np.abs(Z-cz); dz = np.minimum(dz, N-dz)
        sdf = np.minimum(sdf, np.sqrt(dx*dx+dy*dy+dz*dz) - R)
    return sdf

N = 32
s = sdflow.Solver(N, N, N)
s.set_rho(1.0); s.set_mu(1.0); s.set_dt(50.0)
s.set_solid(np.asfortranarray(packing_sdf(N)), cutcell_pressure=True)
s.set_body_force(1.0, 0.0, 0.0)
for _ in range(40): s.step()
u = np.asarray(s.get_u()); p = np.asarray(s.get_p())
print("u_sum  %.17e" % float(u.sum()))
print("u_absmax %.17e" % float(np.abs(u).max()))
print("p_sum  %.17e" % float(p.sum()))
print("u_l2   %.17e" % float(np.sqrt((u*u).sum())))
