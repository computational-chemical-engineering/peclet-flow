#!/usr/bin/env python3
"""GROUND TRUTH: simple-cubic (SC) array of spheres, Stokes drag factor K vs Zick & Homsy (1982).

A single sphere centred in a periodic cube is the classic SC lattice. Z&H give the semi-analytic Stokes
drag factor K(c) (c = sphere solid fraction) to high accuracy -- the external reference that grid
refinement of our two cut-cell codes against *each other* could not provide. pnm_backend was previously
validated against this to <0.1% (output/drag_dimensionless_sc.csv); this redoes it for **sdflow** (and
re-runs pnm_backend to confirm), on the *identical* geometry.

  K = F_total / (6*pi*mu*R*U_sup),   F_total = f * V_cell,   U_sup = mean(u) over the whole cell.

K is dimensionless and unit-invariant, so each solver runs in whatever units suit it; only K is compared.

Usage:  python scripts/validate_zick_homsy_sdflow_vs_pnm.py [N1,N2,...] [phi1,phi2,...] [both|sdflow|pnm]
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


# ---------------------------------------------------------------- sdflow
def run_sdflow(N, phi, mu=0.1, f=1e-3, dt=None, max_steps=600, tol=1e-6, galerkin=False, seed=None):
    """Default = the CORRECT path: direct cut-cell RB-GS pressure operator (galerkin=False).
    galerkin=True uses the Galerkin-MG + PCG pressure path (known to drift -- demonstrably buggy).
    `seed` = (u,v,w) from the next-coarser N, upsampled here, to start the march near steady."""
    import sdflow
    if dt is None:
        dt = 60.0 if N <= 64 else 120.0
    sdf, R = sc_sdf_xyz(N, phi)
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(f, 0, 0); s.set_advection(False)
    if galerkin:
        lv = max(2, int(np.log2(N)) - 2)
        s.set_velocity_multigrid(True, levels=lv, v_cycles=10)
        s.set_pressure_multigrid(True, levels=lv)
        s.set_pressure_pcg(True, max_iter=100, rtol=1e-6)
    else:
        s.set_velocity_solver_params(150); s.set_pressure_solver_params(60)
        s.set_pressure_multigrid(True, levels=1)              # 1 level == pure RB-GS on the cut-cell op
    s.set_solid(sdf, cutcell_pressure=True, galerkin=galerkin)
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


# ---------------------------------------------------------------- pnm_backend (the proven SIMPLE config)
def run_pnm(N, phi, mu=1.0, f=1.0, dt=1.0, max_steps=200, tol=1e-5):
    import pnm_backend
    R = sphere_radius(phi, N)
    sdf, _ = sc_sdf_xyz(N, phi)                              # sdf[x,y,z]
    s = pnm_backend.CFDSolver([N, N, N], [1.0, 1.0, 1.0])
    # current API: initialize takes a float32 array that ravels x-fastest -> shape (nz,ny,nx)
    s.initialize(np.ascontiguousarray(np.transpose(sdf, (2, 1, 0))).astype(np.float32), [0.] * 3, [1.] * 3)
    s.set_rho(0.0); s.set_mu(mu); s.set_body_force(pnm_backend.float3(f, 0, 0))
    s.set_pressure_solver_params(50); s.set_velocity_solver_params(2)
    s.set_outer_iterations(800); s.set_outer_tolerance(0.0)
    prev = 0.0; t0 = time.time()
    for it in range(max_steps):
        s.step(dt)
        if it % 5 == 0:
            m = float(np.asarray(s.get_u()).mean())
            if it > 10 and abs(m - prev) < tol * (abs(m) + 1e-30):
                break
            prev = m
    m = float(np.asarray(s.get_u()).mean())
    return drag_K(m, R, N, f, mu), time.time() - t0


def main():
    Ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [32, 64, 128]
    phis = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.064, 0.125, 0.216]
    which = sys.argv[3] if len(sys.argv) > 3 else "both"
    print("=== Zick & Homsy (1982) SC drag factor K: sdflow vs pnm_backend ===")
    hdr = f"{'phi':>7} {'N':>4} {'K_ZH':>8}"
    if which in ("both", "sdflow"): hdr += f" {'K_sdflow':>9} {'err%':>7}"
    if which in ("both", "pnm"):    hdr += f" {'K_pnm':>9} {'err%':>7}"
    print(hdr)
    for phi in phis:
        kref = zh_ref(phi)
        seed = None  # coarse->fine continuation per phi (sdflow path)
        for N in sorted(Ns):
            row = f"{phi:7.4f} {N:4d} {kref:8.3f}"
            if which in ("both", "sdflow"):
                ks, fld, ts = run_sdflow(N, phi, seed=seed)
                seed = fld
                row += f" {ks:9.3f} {100*(ks-kref)/kref:7.2f}"
            if which in ("both", "pnm"):
                kp, tp = run_pnm(N, phi)
                row += f" {kp:9.3f} {100*(kp-kref)/kref:7.2f}"
            print(row, flush=True)


if __name__ == "__main__":
    main()
