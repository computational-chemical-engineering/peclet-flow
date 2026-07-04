import os, sys, time
sys.path.insert(0, os.path.abspath("build_cuda2"))
import numpy as np
from peclet import flow

def lattice_sdf(N, phi=0.125):
    R=(3*phi/(4*np.pi))**(1/3)*N
    g=np.arange(N)+0.5; X,Y,Z=np.meshgrid(g,g,g,indexing="ij")
    dx=X-0.5*N; dx-=N*np.round(dx/N)
    dy=Y-0.5*N; dy-=N*np.round(dy/N)
    dz=Z-0.5*N; dz-=N*np.round(dz/N)
    return np.sqrt(dx*dx+dy*dy+dz*dz)-R, R

def drag(N, mode, mu=0.1, F=1e-3, dt=80.0, warm_tol=1e-7, tail=40, max_steps=4000):
    sdf,R=lattice_sdf(N); lv=max(2,int(np.log2(N))-1)
    s=flow.SolverColocated(N,N,N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(F,0,0); s.set_advection(False)
    s.set_velocity_solver_params(200); s.set_pressure_multigrid(True,levels=lv)
    s.set_pressure_pcg(True,400,1e-10); s.set_face_interp(mode)
    s.set_solid(sdf,cutcell_pressure=True,pressure_coarse="rediscretized")
    prev,warm,um,t0=0.0,None,[],time.time()
    for it in range(max_steps):
        s.step(); m=float(s.get_u().mean()); um.append(m)
        if warm is None:
            if it%10==9:
                if it>10 and abs(m-prev)<warm_tol*(abs(m)+1e-30): warm=it
                prev=m
        elif it-warm>=tail: break
    return F*N**3/(6*np.pi*mu*R*np.mean(um[-tail:])), it+1, time.time()-t0

kref=4.2920
M0={32:+0.99,48:+0.68,64:+0.598,96:+0.397,128:+0.299}   # mode-0 baselines (prior runs)
print(f"Z&H K={kref}. Collocated: mode0 = plain avg (baseline), mode3 = wall-aware @ open-centroid + transpose.",flush=True)
print(f"{'N':>4} | {'mode0 err%':>10} | {'mode3 err%':>10} | steps | secs",flush=True)
prev_e=None
for N in (32,48,64,96,128):
    K,steps,secs=drag(N,3)
    e=100*(K-kref)/kref
    o=f"  (order {np.log(abs(prev_e)/abs(e))/np.log(N/prev_N):+.2f})" if prev_e else ""
    print(f"{N:>4} | {M0[N]:>+10.3f} | {e:>+10.3f} | {steps:>5d} | {secs:>4.0f}{o}",flush=True)
    prev_e,prev_N=e,N
