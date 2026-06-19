#!/usr/bin/env python3
# Parity test: the Kokkos sdflow developing-channel (uniform inflow -> parabolic outflow) vs CUDA sdflow,
# at the steady fixed point. Domain BCs: -x uniform inflow, +x outflow, +-y no-slip walls. A faithful port
# matches the CUDA velocity field to ~machine precision and recovers the developed parabola u_max/U_mean->1.5.
# (cfd's CUDA sdflow is Kokkos-free, so both modules co-import.)
import sys, gc
import numpy as np
sys.path.insert(0, "build")          # CUDA sdflow
sys.path.insert(0, "build_module")   # Kokkos sdflow_kokkos
import sdflow as cu
import sdflow_kokkos as kk

print("execution space:", kk.execution_space)
L, H, nz, U = 48, 24, 4, 1.0
nu = U * H / 100.0          # Re = 100
dt = 100.0                  # large dt -> steady developing channel


def setup(mod):
    s = mod.Solver(L, H, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(False)
    s.set_domain_bc(0, 2, U, 0.0, 0.0)   # -x inflow (uniform stream)
    s.set_domain_bc(1, 3)                # +x outflow
    s.set_domain_bc(2, 1); s.set_domain_bc(3, 1)   # +-y no-slip walls
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=3); s.set_pressure_solver_params(60)
    s.set_pressure_geometry(np.full((L, H, nz), 1e30))
    return s


sc, sk = setup(cu), setup(kk)
for _ in range(150):
    sc.step(); sk.step()
uc, uk = sc.get_u(), sk.get_u()
du = np.abs(uc - uk).max() / (np.abs(uc).max() + 1e-30)
prof = uk[L - 2, :, nz // 2]               # outlet profile
ratio = float(prof.max() / (prof.mean() + 1e-30))   # developed parabola -> 1.5
print(f"  channel vs CUDA: max|du|/maxu={du:.2e}   outlet u_max/U_mean={ratio:.4f}")
ok = du < 1e-10 and abs(ratio - 1.5) < 0.05
print("PASS" if ok else "FAIL")
del sc, sk; gc.collect()
sys.exit(0 if ok else 1)
