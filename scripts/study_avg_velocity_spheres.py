#!/usr/bin/env python3
"""Grid-convergence of Stokes permeability through an SC sphere array (Zick & Homsy ground truth),
comparing how the SUPERFICIAL VELOCITY U_sup = <u_x> is computed across discretizations:

  * staggered      — sdflow.Solver: u lives on faces (divergence-free), <u_x> over the face field.
  * collocated/cell — sdflow.SolverColocated: <u_x> over the CELL-centered velocity (the default).
  * collocated/face — sdflow.SolverColocated: <u_x> over the projected, divergence-free MAC FACE field
                      (get_uf). Same solve as collocated/cell — only the averaging location differs.
  * amr/uniform     — transport-core tpx_amr.Flow (collocated cut-cell) on a UNIFORM octree (lmax=0).

Motivation: the staggered solver is ~1% more accurate per grid than collocated on permeability. The cell
average is biased by the openness-aware central-difference pressure correction (projectCorrectCenter) at cut
cells, which has a non-zero mean there; the face field's correction is a plain gradient (zero periodic mean),
so its mean is the clean momentum-balance superficial velocity. This script measures whether averaging the
divergence-free face field recovers the staggered accuracy.

K = f N^3 / (6 pi mu R U_sup) (Z&H drag); err% vs the Z&H table. Run from sdflow/ with the OpenMP build:
  SDFLOW_BUILD=build_omp PYTHONPATH=build_omp:../transport-core/python/build python scripts/study_avg_velocity_spheres.py
"""
import os
import sys
import time

import numpy as np

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", os.environ.get("SDFLOW_BUILD", "build_omp"))))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "..", "transport-core", "python", "build")))

import sdflow  # noqa: E402

ZH_PHI = [0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236]
ZH_K = [1.096, 1.212, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1]


def zh_ref(phi):
    return float(np.interp(phi, ZH_PHI, ZH_K))


def sphere_radius(phi, N):
    return (phi * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * N


def sc_sdf(N, phi):
    """Single SC sphere centered in the N^3 cube; suite sign (<0 solid)."""
    R = sphere_radius(phi, N)
    g = np.arange(N)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    c = N / 2.0
    d = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R
    return np.asfortranarray(d), R, c


def drag_K(umean, R, N, f, mu):
    return f * N ** 3 / (6.0 * np.pi * mu * R * umean)


def run_sdflow(SolverCls, N, phi, mu=0.1, f=1e-3, dt=60.0, max_steps=600, tol=1e-6):
    """Returns (U_cell, U_face): the cell-mean and the divergence-free face-mean of u_x at steady state."""
    sdf, R, _ = sc_sdf(N, phi)
    lv = max(2, int(np.log2(N)) - 1)
    s = SolverCls(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(f, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=lv)
    s.set_pressure_pcg(True, max_iter=200, rtol=1e-8)
    s.set_solid(sdf, cutcell_pressure=True)
    prev = 0.0
    for it in range(max_steps):
        s.step()
        if it % 5 == 4:
            m = float(s.get_u().mean())
            if it > 10 and abs(m - prev) < tol * (abs(m) + 1e-30):
                break
            prev = m
    u, uf = s.get_u(), s.get_uf()
    if os.environ.get("STUDY_DIAG") and SolverCls is sdflow.SolverColocated:
        # prove get_uf() is the genuinely different face field, not an alias of the cell field
        print(f"      [diag N={N}] max|uf-u_cell| = {np.abs(uf - u).max():.3e}  "
              f"(field differs pointwise; means: cell {float(u.mean()):.8e} face {float(uf.mean()):.8e})",
              flush=True)
    return float(u.mean()), float(uf.mean())


def run_amr(N, phi, mu=0.1, f=1e-3, dt=60.0, steps=100, mom=120, pres=6, psw=2):
    import tpx_amr
    _, R, c = sc_sdf(N, phi)
    oct = tpx_amr.Octree([N, N, N], 0, [0.0, 0.0, 0.0], 1.0)
    fl = tpx_amr.Flow(oct, 1.0, mu, dt)
    fl.set_body_force(f, 0, 0); fl.set_advection(False)
    fl.set_solid(lambda x, y, z: ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) ** 0.5 - R)
    for _ in range(steps):
        fl.step(mom, pres, psw)
    return float(fl.velocity(0).mean())


def main():
    phi = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
    Ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [16, 24, 32, 48]
    amr_max = int(os.environ.get("AMR_MAX_N", "32"))  # AMR host path is serial; cap its N
    kref = zh_ref(phi)
    mu, f = 0.1, 1e-3

    print(f"=== Stokes permeability through an SC sphere (phi={phi}, Z&H K_ref={kref:.4f}) ===")
    print(f"    U_sup averaging: staggered(face) | collocated(cell) | collocated(FACE) | amr/uniform(cell)")
    print(f"{'N':>4} | {'K_stag':>8} {'e%':>6} | {'K_co_cell':>9} {'e%':>6} | {'K_co_FACE':>9} {'e%':>6}"
          f" | {'K_amr':>8} {'e%':>6} | {'face-cell':>9}")
    for N in Ns:
        t0 = time.time()
        u_stag, _ = run_sdflow(sdflow.Solver, N, phi)
        uc, uf = run_sdflow(sdflow.SolverColocated, N, phi)
        Ks = drag_K(u_stag, sphere_radius(phi, N), N, f, mu)
        Kc = drag_K(uc, sphere_radius(phi, N), N, f, mu)
        Kf = drag_K(uf, sphere_radius(phi, N), N, f, mu)
        es = 100 * (Ks - kref) / kref
        ec = 100 * (Kc - kref) / kref
        ef = 100 * (Kf - kref) / kref
        fc = (uf - uc) / uc  # relative shift in U_sup from face vs cell averaging
        if N <= amr_max:
            ua = run_amr(N, phi)
            Ka = drag_K(ua, sphere_radius(phi, N), N, f, mu)
            ea = 100 * (Ka - kref) / kref
            amr = f"{Ka:8.4f} {ea:+6.2f}"
        else:
            amr = f"{'--':>8} {'--':>6}"
        print(f"{N:4d} | {Ks:8.4f} {es:+6.2f} | {Kc:9.4f} {ec:+6.2f} | {Kf:9.4f} {ef:+6.2f}"
              f" | {amr} | {fc:+.2e} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
