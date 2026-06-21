#!/usr/bin/env python3
"""Phase-4 verification (collocated grid): Stokes flow through a periodic 2x2x2 sphere packing with the
Robust-Scaled cut-cell IBM — the first immersed-solid test of the collocated solver. A body force drives
flow through the pores; the cut-cell overlay enforces no-slip at the cell-centered velocities and the
shared cut-cell pressure operator + approximate (MAC) projection enforce incompressibility.

This is the §4 gate of doc/sdflow_colocated_plan.md: the collocated permeability must (a) be incompressible
(projected FACE field divergence-free to machine precision), (b) satisfy exact no-slip in the deep solid,
and (c) converge — across resolution AND toward the staggered solver, which is itself validated against the
Zick & Homsy sphere-array drag. The plain area-weighted face averaging (Option A) + central-difference cell
correction are tested here; escalation to the open-centroid reconstruction (Option B) / openness-aware cell
gradient is only warranted if this gate fails.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", os.environ.get("SDFLOW_BUILD", "build"))))
import sdflow  # noqa: E402


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
    return np.asfortranarray(best), R


def run(SolverCls, N, mu=0.1, dt=60.0, F=1e-3, steps=400):
    sdf, R = packing_sdf(N)
    s = SolverCls(N, N, N)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0.0, 0.0)
    s.set_advection(False)                      # creeping (Stokes) flow
    s.set_velocity_solver_params(80)
    s.set_pressure_pcg(True, 500, 1e-10)
    s.set_solid(sdf, cutcell_pressure=True)

    prev = 0.0
    for it in range(steps):
        s.step()
        um = float(s.get_u().mean())
        if it > 8 and abs(um - prev) < 3e-4 * (abs(um) + 1e-15):
            break
        prev = um

    u = s.get_u()
    k = mu * float(u.mean()) / F
    u_solid = float(np.abs(u[sdf < -2.0]).max())
    return k, u_solid, float(s.max_open_divergence()), float(u.max())


def main():
    Ns = (32, 48, 64)
    print("=== sdflow phase-4: collocated cut-cell IBM — Stokes permeability through a sphere packing ===")
    print(f"{'N':>4} {'k_stag':>12} {'k_coloc':>12} {'rel.diff':>9} {'maxdiv_co':>11} {'u_solid_co':>11}")
    rows = []
    for N in Ns:
        ks, _, _, _ = run(sdflow.Solver, N)
        kc, usc, dc, umc = run(sdflow.SolverColocated, N)
        rel = 100.0 * abs(kc - ks) / ks
        print(f"{N:4d} {ks:12.5e} {kc:12.5e} {rel:8.2f}% {dc:11.2e} {usc:11.2e}")
        rows.append((ks, kc, rel, dc, usc))

    incompressible = all(r[3] < 1e-7 for r in rows)
    no_slip = all(r[4] == 0.0 for r in rows)
    k_converges = rows[0][1] < rows[1][1] < rows[2][1]          # collocated k rises as spheres resolve
    rel_shrinks = rows[2][2] < rows[0][2]                        # collocated -> staggered with resolution
    ok = incompressible and no_slip and k_converges and rel_shrinks
    print(f"  incompressible (face div <1e-7): {incompressible}")
    print(f"  exact no-slip in deep solid: {no_slip}")
    print(f"  collocated k grid-converges: {k_converges}  ({rows[0][1]:.4e} -> {rows[2][1]:.4e})")
    print(f"  collocated -> staggered with N: {rel_shrinks}  ({rows[0][2]:.2f}% -> {rows[2][2]:.2f}%)")
    print(f"  result: {'PASS' if ok else 'FAIL'}  (Option A area-weighting + central-diff cell correction)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
