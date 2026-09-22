"""set_solid cost: sampled-array vs scene path, and the per-call breakdown that Layer 3's
incremental-rebuild decision needs. Also the per-step moving-geometry cost TODAY."""
import numpy as np, time, sys
from peclet import flow as sdflow

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
R = 0.18 * N
CEN = [(0.25*N,0.25*N,0.25*N),(0.75*N,0.75*N,0.25*N),(0.75*N,0.25*N,0.75*N),(0.25*N,0.75*N,0.75*N)]
KN_R, KI_I, KI_R = 16, 2, 17
node_ints = np.array([1,-1,-1], dtype=np.int32)
node_reals = np.zeros(KN_R); node_reals[0]=R; node_reals[14]=1.0; node_reals[15]=1.0
bi = np.zeros((4,KI_I),dtype=np.int32); br = np.zeros((4,KI_R))
for m,(cx,cy,cz) in enumerate(CEN):
    bi[m]=(0,-1); br[m,0:3]=(cx-0.5,cy-0.5,cz-0.5); br[m,6]=1.0; br[m,7]=1.0

g = np.arange(N)+0.5
X,Y,Z = np.meshgrid(g,g,g,indexing="ij")
ref = np.full((N,N,N),1e30)
for (cx,cy,cz) in CEN:
    dx=np.abs(X-cx); dx=np.minimum(dx,N-dx); dy=np.abs(Y-cy); dy=np.minimum(dy,N-dy)
    dz=np.abs(Z-cz); dz=np.minimum(dz,N-dz)
    ref=np.minimum(ref,np.sqrt(dx*dx+dy*dy+dz*dz)-R)
refF = np.asfortranarray(ref)

s = sdflow.Solver(N,N,N)
s.set_rho(1.0); s.set_mu(1.0); s.set_dt(50.0)
s.set_scene(node_ints,node_reals,bi.ravel(),br.ravel(),periodic=True)
s.set_solid_from_scene(True)  # warm-up (allocations, MG first build)
for name, fn in [("set_solid(array)  incl. numpy-free upload", lambda: s.set_solid(refF, cutcell_pressure=True)),
                 ("set_solid_from_scene FULL (pressure on)   ", lambda: s.set_solid_from_scene(True)),
                 ("set_solid_from_scene MOMENTUM-ONLY        ", lambda: s.set_solid_from_scene(False))]:
    ts=[]
    for _ in range(5):
        t0=time.time(); fn(); ts.append(time.time()-t0)
    ts=sorted(ts)
    print("  N=%d  %-42s median %.1f ms  (min %.1f)" % (N, name, 1e3*ts[2], 1e3*ts[0]))
print("  -> a moving body today costs the full set_solid_from_scene per step")
