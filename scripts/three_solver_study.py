"""sdflow side of the three-solver study: staggered vs collocated, Z&H SC sphere + random packing,
Re=0 (Stokes) and Re=100 (advection on). Prints K (Z&H) / k* (random), cells, pressure iters, wall.
Geometry + metrics identical to tests/regression/sdflow_regression.py so AMR can be compared directly.
Pair with transport-core/python/amr_drag_study.py for the AMR (graded cut-cell) side.

Findings (2026-06-25):
  * Z&H phi=0.216 (ref K=7.442), Re=0: staggered converges 2nd-order from below
    (N=16 -1.78% -> N=48 -0.07%); collocated carries the intrinsic +~1% gap (N=32 +1.16%).
  * random pack k*: staggered -> ~0.00622 from above, collocated -> ~0.00618 from below (same k_inf).
  * Re~100 (F=2.6e-3, N=32): staggered K=8.90 (Re=101.6), collocated K=9.05 (Re=99.9), ~+20% over
    Stokes; both converge cleanly (div~1e-12). Same F at N=48 gives Re~300 (R scales with N) -- only
    N=32 is a true Re=100 comparison.
  * Staggered is the accuracy default for permeability/drag; collocated trades ~1%/grid for
    cell-centered storage. (Kokkos/OpenMP: ~1-30 s per case.)
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build_omp"))))
import sdflow

ZH_PHI = [0.000125,0.001,0.008,0.027,0.064,0.125,0.216,0.343,0.45,0.5236]
ZH_K   = [1.096,1.212,1.525,2.008,2.810,4.292,7.442,15.4,28.1,42.1]
zh_ref = lambda phi: float(np.interp(phi, ZH_PHI, ZH_K))

def _grid(N):
    g=np.arange(N)+0.5; return np.meshgrid(g,g,g,indexing="ij")
def _mm(d,N): return d-N*np.round(d/N)

def sdf_zh(N, phi=0.216):
    R=(phi*3/(4*np.pi))**(1/3)*N; X,Y,Z=_grid(N); c=N/2
    return np.asfortranarray(np.sqrt((X-c)**2+(Y-c)**2+(Z-c)**2)-R), R

def sdf_random(N, n=8, r_frac=0.18, jit=0.06, seed=12345):
    rng=np.random.default_rng(seed); R=r_frac*N
    base=np.array([[(i+.5)/2,(j+.5)/2,(k+.5)/2] for i in range(2) for j in range(2) for k in range(2)])
    ctr=((base+jit*rng.standard_normal(base.shape))%1.0)*N
    X,Y,Z=_grid(N); sdf=np.full((N,N,N),1e30)
    for cx,cy,cz in ctr:
        sdf=np.minimum(sdf,np.sqrt(_mm(X-cx,N)**2+_mm(Y-cy,N)**2+_mm(Z-cz,N)**2)-R)
    return np.asfortranarray(sdf), R

def run(case, N, solver="staggered", re=0.0, mu=0.1, F=1e-3, dt=60.0, max_steps=500, tol=1e-6):
    if case=="zh": sdf,R=sdf_zh(N); metric="K"
    else:          sdf,R=sdf_random(N); metric="k*"
    lv=max(2,int(np.floor(np.log2(N)))-1)
    Cls=sdflow.SolverColocated if solver=="colocated" else sdflow.Solver
    s=Cls(N,N,N); s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(F,0,0)
    s.set_advection(re>0.0)
    if re>0: s.set_implicit_advection(True)
    s.set_velocity_solver_params(80)
    s.set_pressure_multigrid(True,levels=lv); s.set_pressure_pcg(True,300,1e-8)
    s.set_solid(sdf,cutcell_pressure=True,pressure_coarse="rediscretized")
    t0=time.time(); prev=0.0; steps=0; piters=[]
    for it in range(max_steps):
        s.step(); steps+=1; piters.append(s.last_pressure_iterations())
        if it%5==4:
            m=float(s.get_u().mean())
            if it>=15 and abs(m-prev)<tol*(abs(m)+1e-30): break
            prev=m
    wall=time.time()-t0; u=s.get_u(); um=float(u.mean())
    K=F*N**3/(6*np.pi*mu*R*um); kstar=mu*um/(F*N**2)
    Re=1.0*abs(um)*2*R/mu
    val=K if metric=="K" else kstar
    rr={"N":N,"cells":N**3,"metric":metric,"val":val,"K":K,"kstar":kstar,"umean":um,
        "Re":Re,"piters":int(np.median(piters[len(piters)//2:])),"steps":steps,
        "div":float(s.max_open_divergence()),"wall":wall}
    del s
    return rr

if __name__=="__main__":
    import gc
    cases = [("zh",[16,24,32,48]), ("random",[24,32,48])]
    print("=== sdflow: staggered vs collocated, Re=0 (Stokes) ===",flush=True)
    for case,grids in cases:
        ref = zh_ref(0.216) if case=="zh" else None
        print(f"\n[{case}] ref={ref}",flush=True)
        for solver in ("staggered","colocated"):
            for N in grids:
                r=run(case,N,solver,re=0.0); gc.collect()
                err = f"{100*(r['val']-ref)/ref:+.2f}%" if ref else ""
                print(f"  {solver:10s} N={N:3d} cells={r['cells']:6d} {r['metric']}={r['val']:.5g} {err:>8} "
                      f"piter={r['piters']:3d} steps={r['steps']:3d} div={r['div']:.1e} {r['wall']:.1f}s",flush=True)
    print("\n=== sdflow: Re~100 (advection on), Z&H phi=0.216 ===",flush=True)
    for solver in ("staggered","colocated"):
        for N in [32,48]:
            r=run("zh",N,solver,re=100.0,F=5e-2,dt=20.0,max_steps=800); gc.collect()
            print(f"  {solver:10s} N={N:3d} K={r['K']:.4f} Re={r['Re']:.1f} steps={r['steps']} "
                  f"div={r['div']:.1e} {r['wall']:.1f}s",flush=True)
