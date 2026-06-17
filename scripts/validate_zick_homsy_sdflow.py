#!/usr/bin/env python3
"""GROUND TRUTH: simple-cubic (SC) array of spheres, Stokes drag factor K vs Zick & Homsy (1982).

A single sphere centred in a periodic cube is the classic SC lattice. Z&H give the semi-analytic Stokes
drag factor K(c) (c = sphere solid fraction) to high accuracy -- the external reference for the sdflow
cut-cell IBM. (Historically this also re-ran the retired pnm_backend reference solver; sdflow was validated
bit-identical to it before its retirement, so only the external Z&H comparison remains.)

  K = F_total / (6*pi*mu*R*U_sup),   F_total = f * V_cell,   U_sup = mean(u) over the whole cell.

K is dimensionless and unit-invariant.

Usage:  python scripts/validate_zick_homsy_sdflow.py [N1,N2,...] [phi1,phi2,...]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))

# Zick & Homsy (1982), simple cubic: solid fraction c -> drag factor K.
ZH_PHI = [0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236]
ZH_K = [1.096, 1.212, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1]


def zh_ref(phi):
    return float(np.interp(phi, ZH_PHI, ZH_K))


def sphere_radius(phi, N):
    """Sphere radius (grid units) for solid fraction phi in an N^3 cube."""
    return (phi * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * N


def sc_sdf_xyz(N, phi):
    """SC single sphere, SDF as 3-D array sdf[x,y,z] in grid units (dx=1); negative inside the sphere."""
    R = sphere_radius(phi, N)
    g = np.arange(N) + 0.5  # cell centres
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    c = N / 2.0
    return np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R, R


def drag_K(umean, R, N, f, mu):
    """K = F_total/(6 pi mu R U_sup), F_total = f*N^3 (grid units), U_sup = mean(u)."""
    return f * N ** 3 / (6.0 * np.pi * mu * R * umean)


def _upsample2(a):
    """2x nearest upsample x4 (self-similar packing: u_2N(2x) ~ 4*u_N(x)) -> near-steady seed."""
    return np.repeat(np.repeat(np.repeat(a, 2, 0), 2, 1), 2, 2) * 4.0


def run_sdflow(N, phi, mu=0.1, f=1e-3, dt=None, max_steps=600, tol=1e-6, coarse="rediscretized",
               seed=None):
    """Multilevel MG-PCG pressure solve; `coarse` selects the coarse-operator mode
    ('rediscretized' = the geometric per-level cut-cell operator, the recommended default;
    'galerkin' = the inconsistent aggregation path; 'const' = geometry-blind coarse).
    `seed` = (u,v,w) from the next-coarser N, upsampled here, to start the march near steady."""
    import sdflow
    if dt is None:
        dt = 60.0 if N <= 64 else 120.0
    lv = max(2, int(np.log2(N)) - 1)                          # coarsen to ~4^3
    sdf, R = sc_sdf_xyz(N, phi)
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(f, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(200)                         # velocity RB-GS (the velocity MG under-
    #                                                           converges the IBM diffusion -- separate bug)
    s.set_pressure_multigrid(True, levels=lv)
    s.set_pressure_pcg(True, max_iter=200, rtol=1e-8)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse=coarse)
    if seed is not None:
        u, v, w = (np.asfortranarray(_upsample2(c)) for c in seed)
        s.set_state(u, v, w)
    prev = 0.0; t0 = time.time()
    for it in range(max_steps):
        s.step()
        if it % 5 == 4:
            m = float(s.get_u().mean())
            if it > 10 and abs(m - prev) < tol * (abs(m) + 1e-30):
                break
            prev = m
    m = float(s.get_u().mean())
    return drag_K(m, R, N, f, mu), (s.get_u(), s.get_v(), s.get_w()), time.time() - t0


def main():
    Ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [32, 64, 128]
    phis = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.064, 0.125, 0.216]
    print("=== Zick & Homsy (1982) SC drag factor K: sdflow ===")
    print(f"{'phi':>7} {'N':>4} {'K_ZH':>8} {'K_sdflow':>9} {'err%':>7}")
    for phi in phis:
        kref = zh_ref(phi)
        seed = None  # coarse->fine continuation per phi
        for N in sorted(Ns):
            ks, fld, ts = run_sdflow(N, phi, seed=seed)
            seed = fld
            print(f"{phi:7.4f} {N:4d} {kref:8.3f} {ks:9.3f} {100*(ks-kref)/kref:7.2f}", flush=True)


if __name__ == "__main__":
    main()
