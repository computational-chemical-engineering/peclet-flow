#!/usr/bin/env python3
# Python validation of the Kokkos sdflow drop-in: plane Poiseuille through an SDF channel with
# cut-cell IBM no-slip walls, body-force driven, physical units. Mirrors verify_poiseuille_sdflow.
# U_max must match the analytic parabola F*H^2/(8*mu).
import sys, gc
import numpy as np
import sdflow_kokkos as sdflow

print("execution space:", sdflow.execution_space)

nx, ny, nz = 4, 48, 4
ylo, yhi = 6.5, ny - 6.5            # half-integer walls -> cut cells form
H = yhi - ylo
rho, mu, dt, F = 1.0, 0.1, 50.0, 0.01

gy = np.arange(ny, dtype=np.float64)
sdf = np.empty((nx, ny, nz))
sdf[:, :, :] = np.minimum(gy - ylo, yhi - gy)[None, :, None]   # <0 in the walls

s = sdflow.Solver(nx, ny, nz)
s.set_rho(rho); s.set_mu(mu); s.set_dt(dt)
s.set_body_force(F, 0.0, 0.0)
s.set_velocity_solver_params(200)
s.set_solid(np.asfortranarray(sdf))
for _ in range(600):
    s.step()

u = s.get_u()                       # [x, y, z]
prof = u[0, :, 0]
U_ana = F * H * H / (8.0 * mu)
umax = float(prof.max())

# profile L2 error vs the analytic parabola over the fluid interior
ys = np.arange(ny, dtype=np.float64)
fluid = (ys > ylo) & (ys < yhi)
ue = (F / (2.0 * mu)) * (ys - ylo) * (yhi - ys)
l2 = float(np.sqrt(np.sum((prof[fluid] - ue[fluid]) ** 2) / np.sum(ue[fluid] ** 2)))
umax_err = abs(umax - U_ana) / U_ana

print(f"U_max={umax:.5f}  analytic={U_ana:.5f}  (err {umax_err:.2e});  profile L2 err={l2:.2e}")
ok = (umax_err < 3e-2) and (l2 < 3e-2)
print("PASS" if ok else "FAIL")

del s
gc.collect()
sys.exit(0 if ok else 1)
