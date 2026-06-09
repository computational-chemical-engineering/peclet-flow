#!/usr/bin/env python3
"""Verification (sdflow): creeping (Stokes) flow through a periodic sphere packing -- the porous-media
target case. A body force drives flow through the pore space; the Robust-Scaled cut-cell IBM enforces
no-slip on the spheres and the cut-cell pressure operator (Galerkin multigrid + CG) enforces
incompressibility. We report the Darcy permeability k = mu*<u>/F and check:
  * the flow is incompressible (small cut-cell flux divergence),
  * no-slip holds (velocity ~ 0 inside the solid),
  * the permeability is finite/positive and increases as the spheres resolve.

Uses the canonical `sdflow` module (one GPU as plain `python`, or multi-rank under `mpirun -np N python`).
Physical units: set_rho/set_mu, body force F is a force per unit volume (= -dp/dx). Grid units (dx = 1).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import sdflow  # noqa: E402


def packing_sdf(N, radius_frac=0.18):
    """2x2x2 sphere packing as a 3-D array sdf[x,y,z]; negative inside the spheres (min-image)."""
    R = N * radius_frac
    gx = np.arange(N)
    cs = [(cx + 0.5) * N / 2.0 for cx in (0, 1)]  # centres of the 2x2x2 lattice
    X, Y, Z = np.meshgrid(gx, gx, gx, indexing="ij")  # (N,N,N) indexed [x,y,z]
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
    sdf, R = packing_sdf(N)  # 3-D sdf[x,y,z]
    porosity = 1.0 - 8.0 * (4.0 / 3.0 * np.pi * R**3) / N**3

    s = sdflow.Solver(N, N, N)
    s.set_rho(rho)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0.0, 0.0)
    s.set_advection(False)  # creeping (Stokes) flow
    s.set_pressure_pcg(True, max_iter=150, rtol=1e-9)            # CG-accelerated pressure solve
    s.set_velocity_multigrid(True, levels=4, v_cycles=12)
    s.set_solid(sdf, cutcell_pressure=True, galerkin=True)       # no-slip + cut-cell pressure operator

    deep_solid = sdf < -2.0  # cells whose every velocity face is solid -> must be exactly no-slip
    prev = 0.0
    for it in range(max_steps):
        s.step()  # large dt -> backward Euler approaches the steady Stokes solve
        u = s.get_u()                          # collective gather: ALL ranks must call it
        converged = False
        if s.rank() == 0:
            umean = float(u.mean())
            converged = it > 8 and abs(umean - prev) < 3e-4 * (abs(umean) + 1e-15)
            prev = umean
        if s.bcast_from_root(converged):
            break

    u = s.get_u()                              # collective
    div = s.max_open_divergence()              # collective (Allreduce) -- all ranks must call it
    if s.rank() != 0:
        return None
    k = mu * float(u.mean()) / F               # Darcy permeability (grid units)
    u_solid = float(np.abs(u[deep_solid]).max())  # no-slip check (deep solid)
    return N, porosity, k, u_solid, float(u.max()), div


def main():
    # All ranks must call run() for every N (collective steps/gathers); only root has results.
    results = [run(N) for N in (32, 64)]
    if results[0] is None:  # non-root rank
        return
    print("=== sdflow: Stokes flow through a periodic sphere packing (rho/mu units) ===")
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
    print(f"  permeability k(N=32)={k32:.3e}  k(N=64)={k64:.3e}  (rises as the spheres resolve)")
    print(f"  incompressible={incompressible}  no-slip(exact in solid)={no_slip}  sensible-k={sensible}")
    print(f"  result: {'PASS' if ok else 'FAIL'}  (incompressible Stokes flow, exact no-slip on SDF spheres)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
