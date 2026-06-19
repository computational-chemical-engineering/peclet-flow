#!/usr/bin/env python3
# Parity test: Kokkos sdflow backward-facing step (non-uniform inlet PROFILE via set_domain_bc_profile,
# the step realized purely as the inlet condition) vs CUDA sdflow, at the steady fixed point. A faithful
# port matches the CUDA velocity field to ~machine precision. (cfd CUDA sdflow is Kokkos-free; co-import.)
import sys, gc
import numpy as np
sys.path.insert(0, "build"); sys.path.insert(0, "build_module")
import sdflow as cu
import sdflow_kokkos as kk
print("execution space:", kk.execution_space)
S=8; H=2*S; L=24*S; nz=4; U=1.0; nu=U*S/100.0; dt=50.0
def inlet(H,S,nz,U):
    p=np.zeros((H,nz,3)); yc=np.arange(H)+0.5; eta=(yc-S)/S; up=yc>S
    p[up,:,0]=(6.0*U*eta*(1-eta))[up,None]; return p
def setup(mod):
    s=mod.Solver(L,H,nz); s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(False)
    s.set_domain_bc_profile(0, inlet(H,S,nz,U)); s.set_domain_bc(1,3)
    s.set_domain_bc(2,1); s.set_domain_bc(3,1)
    s.set_velocity_solver_params(60); s.set_pressure_multigrid(True,levels=4); s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((L,H,nz),1e30)); return s
sc,sk=setup(cu),setup(kk)
for _ in range(150): sc.step(); sk.step()
uc,uk=sc.get_u(),sk.get_u()
du=np.abs(uc-uk).max()/(np.abs(uc).max()+1e-30)
print(f"  BFS vs CUDA: max|du|/maxu={du:.2e}  (CUDA u.max={uc.max():.3f}, Kokkos u.max={uk.max():.3f})")
ok = du<1e-10 and np.isfinite(uk).all()
print("PASS" if ok else "FAIL"); del sc,sk; gc.collect(); sys.exit(0 if ok else 1)
