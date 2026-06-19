#!/usr/bin/env python3
# Python validation of the Kokkos sdflow drop-in: creeping (Stokes) flow through a periodic 2x2x2 sphere
# packing -- the porous-media target. Body force drives flow through the pore space; the cut-cell IBM
# enforces no-slip on the spheres and the cut-cell pressure projection enforces incompressibility. Mirrors
# verify_periodic_spheres_sdflow.py: checks incompressible (small flux divergence), exact no-slip in deep
# solid, and a finite positive permeability k = mu*<u>/F that rises as the spheres resolve.
import sys, gc
import numpy as np
import sdflow_kokkos as sdflow

print("execution space:", sdflow.execution_space)


def packing_sdf(N, radius_frac=0.18):
    R = N * radius_frac
    gx = np.arange(N)
    cs = [(c + 0.5) * N / 2.0 for c in (0, 1)]
    X, Y, Z = np.meshgrid(gx, gx, gx, indexing="ij")
    best = np.full((N, N, N), 1e30)
    for sx in cs:
        for sy in cs:
            for sz in cs:
                dx = X - sx; dx -= N * np.round(dx / N)
                dy = Y - sy; dy -= N * np.round(dy / N)
                dz = Z - sz; dz -= N * np.round(dz / N)
                best = np.minimum(best, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return best, R


def run(N, rho=1.0, mu=0.1, dt=60.0, F=1e-3, max_steps=200):
    sdf, R = packing_sdf(N)
    porosity = 1.0 - 8.0 * (4.0 / 3.0 * np.pi * R**3) / N**3
    s = sdflow.Solver(N, N, N)
    s.set_rho(rho); s.set_mu(mu); s.set_dt(dt)
    s.set_body_force(F, 0.0, 0.0)
    s.set_advection(False)
    s.set_velocity_solver_params(80)
    s.set_pressure_solver_params(20)
    s.set_pressure_multigrid(True, levels=1)
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True, pressure_coarse="const")
    deep_solid = sdf < -2.0
    prev = 0.0
    for it in range(max_steps):
        s.step()
        u = s.get_u()
        umean = float(u.mean())
        if it > 8 and abs(umean - prev) < 3e-4 * (abs(umean) + 1e-15):
            prev = umean
            break
        prev = umean
    u = s.get_u()
    div = s.max_open_divergence()
    k = mu * float(u.mean()) / F
    u_solid = float(np.abs(u[deep_solid]).max())
    res = (N, porosity, k, u_solid, float(u.max()), div)
    del s; gc.collect()
    return res


def main():
    results = [run(N) for N in (32, 64)]
    print("=== sdflow_kokkos: Stokes flow through a periodic sphere packing (rho/mu units) ===")
    print(f"{'N':>4} {'porosity':>9} {'permeability k':>15} {'max|u|solid':>12} {'max fluxdiv':>12}")
    ks = []
    for N_, phi, k, us, umax, div in results:
        print(f"{N_:4d} {phi:9.3f} {k:15.6e} {us:12.3e} {div:12.3e}")
        ks.append((k, us, umax, div))
    k32, k64 = ks[0][0], ks[1][0]
    incompressible = all(d < 1e-6 * umax for (_, _, umax, d) in ks)
    no_slip = all(us == 0.0 for (_, us, _, _) in ks)
    sensible = 0.0 < k32 < k64 and np.isfinite(k64)
    ok = incompressible and no_slip and sensible
    print(f"  k(N=32)={k32:.3e}  k(N=64)={k64:.3e}")
    print(f"  incompressible={incompressible}  no-slip(exact in solid)={no_slip}  sensible-k={sensible}")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
