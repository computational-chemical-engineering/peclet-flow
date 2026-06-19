#!/usr/bin/env python3
# Parity test: the Kokkos sdflow domain-BC lid-driven cavity vs the CUDA sdflow, at the steady-state
# fixed point (large dt, backward Euler). No immersed solid -- the cavity is set up with set_domain_bc
# (3 no-slip walls + a moving lid) + set_pressure_geometry(all-fluid). A faithful port must match the
# CUDA velocity field to ~machine precision (a steady linear Stokes solve has a unique fixed point).
# (cfd's CUDA sdflow is Kokkos-free, so both modules co-import in one process.)
import sys, gc
import numpy as np
sys.path.insert(0, "build")          # CUDA sdflow
sys.path.insert(0, "build_module")   # Kokkos sdflow_kokkos
import sdflow as cu
import sdflow_kokkos as kk

print("execution space:", kk.execution_space)
N, nz, U = 32, 4, 1.0
nu = U * N / 100.0          # Re = 100
dt = 200.0                  # large dt -> backward Euler reaches the steady Stokes cavity in a few steps


def setup(mod):
    s = mod.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt)
    s.set_advection(False)                          # Stokes -> unique steady fixed point
    s.set_domain_bc(0, 1); s.set_domain_bc(1, 1); s.set_domain_bc(2, 1)   # -x, +x, -y no-slip walls
    s.set_domain_bc(3, 2, U, 0.0, 0.0)                                    # +y lid moving in +x
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=3); s.set_pressure_solver_params(50)
    s.set_pressure_geometry(np.full((N, N, nz), 1e30))
    return s


sc, sk = setup(cu), setup(kk)
for _ in range(120):
    sc.step(); sk.step()
uc, vc = sc.get_u(), sc.get_v()
uk, vk = kk.get_u() if False else sk.get_u(), sk.get_v()
du = np.abs(uc - uk).max() / (np.abs(uc).max() + 1e-30)
dv = np.abs(vc - vk).max() / (np.abs(vc).max() + 1e-30)
# physical sanity: lid drives a recirculation -> centreline u is negative in the lower half
prof = uk[N // 2, :, nz // 2]
recirc = prof[: N // 4].mean() < 0.0 < prof[3 * N // 4:].mean()
print(f"  steady cavity vs CUDA: max|du|/maxu={du:.2e}  max|dv|/maxv={dv:.2e}")
print(f"  recirculation (Kokkos): {recirc}")
ok = du < 1e-10 and dv < 1e-10 and recirc
print("PASS" if ok else "FAIL")
del sc, sk; gc.collect()
sys.exit(0 if ok else 1)
