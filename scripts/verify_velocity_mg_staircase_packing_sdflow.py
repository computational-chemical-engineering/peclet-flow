#!/usr/bin/env python3
"""Staircase velocity-MG on a REAL packed bed (random periodic sphere packing, thin pore necks).

The SC single sphere is one connected solid blob; a packed bed has MANY solid spheres with thin fluid necks
between near-contacts -- the geometry that stresses the coarse operator (the const-coarse couples fluid
pockets through resolved walls; the staircase classification should disconnect them). This checks that the
staircase velocity-MG (a) gives the EXACT RB-GS mean velocity / permeability on a packed bed, (b) stays stable
at large dt, and (c) how the coarsening-level cap matters (deep coarsening dissolves the thin necks).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import sdflow  # noqa: E402


def packing_sdf(N, R, target_phi, gap=2.0, seed=0):
    """Random periodic sphere packing: greedily place non-overlapping spheres (min surface gap `gap` cells)
    until the solid fraction reaches target_phi. SDF = min over spheres of (dist - R), periodic, <0 in solid."""
    rng = np.random.default_rng(seed)
    g = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    sdf = np.full((N, N, N), 1e30)
    centers = []
    vsph = 4.0 / 3.0 * np.pi * R ** 3
    while (len(centers) * vsph) / N ** 3 < target_phi:
        placed = False
        for _ in range(2000):
            c = rng.random(3) * N
            ok = True
            for cc in centers:
                d = c - cc; d -= N * np.round(d / N)
                if np.linalg.norm(d) < 2 * R + gap:
                    ok = False; break
            if ok:
                centers.append(c); placed = True; break
        if not placed:
            break
    for c in centers:
        dx = X - c[0]; dx -= N * np.round(dx / N)
        dy = Y - c[1]; dy -= N * np.round(dy / N)
        dz = Z - c[2]; dz -= N * np.round(dz / N)
        sdf = np.minimum(sdf, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return sdf, len(centers), float((sdf < 0).mean())


def run(sdf, mode, dt, vcyc=12, levels=4, mu=0.1, f=1e-3, steps=500, tol=1e-7):
    N = sdf.shape[0]
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(f, 0, 0); s.set_advection(False)
    if mode == "rbgs":
        s.set_velocity_solver_params(400)
    else:
        s.set_velocity_multigrid(True, levels, vcyc)
        if mode == "const":
            pass                                              # geometry-blind const-coeff coarse (no volfrac)
        elif mode == "area":
            s.set_velocity_mg_volfrac(True, 0.1, True, False)  # area-fraction operator
        elif mode == "stair":
            s.set_velocity_mg_volfrac(True, 0.1, True, True)   # staircase operator
    plv = max(2, int(np.log2(N)) - 1)
    s.set_pressure_multigrid(True, levels=plv); s.set_pressure_pcg(True, max_iter=200, rtol=1e-8)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev = 0.0
    for it in range(steps):
        s.step()
        if not np.isfinite(s.get_u()).all():
            return None
        if it % 5 == 4:
            m = float(s.get_u().mean())
            if it > 10 and abs(m - prev) < tol * (abs(m) + 1e-30):
                break
            prev = m
    return float(s.get_u().mean())   # superficial velocity ~ permeability (compared across methods)


def main():
    N = int(os.environ.get("VMG_N", 64))
    R = float(os.environ.get("VMG_R", 9.0))
    phi = float(os.environ.get("VMG_PHI", 0.35))
    sdf, nsph, phi_act = packing_sdf(N, R, phi, seed=int(os.environ.get("VMG_SEED", 1)))
    print(f"=== packed bed: {nsph} spheres R={R:.0f} in {N}^3, solid phi={phi_act:.3f} ===")

    uref = run(sdf, "rbgs", dt=60.0)
    print(f"  RB-GS reference: U_sup={uref:.6e}")
    print("--- correctness (vs RB-GS) + large-dt stability, levels=4 ---")
    for mode in ["const", "area", "stair"]:
        for dt in [60.0, 800.0]:
            u = run(sdf, mode, dt=dt, levels=4)
            e = "NaN" if u is None else f"{abs(u - uref) / abs(uref) * 100:.3f}%"
            print(f"  {mode:6s} dt={dt:5.0f}: {'BLEW UP' if u is None else f'U_sup={u:.6e}  err {e}'}")

    print("--- staircase: coarsening-level cap (deep levels dissolve thin necks), dt=60 ---")
    for lv in [2, 3, 4, 5]:
        u = run(sdf, "stair", dt=60.0, levels=lv)
        e = "NaN" if u is None else f"{abs(u - uref) / abs(uref) * 100:.3f}%"
        print(f"  levels={lv}: {'BLEW UP' if u is None else f'err {e}'}")


if __name__ == "__main__":
    main()
