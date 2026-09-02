#!/usr/bin/env python3
"""WO-V6b gate 1 + the float floor — the Navier slip length in the cut-cell IBM wall closure.

Two analytically exact steady profiles over FLAT SDF walls, both driven without moving geometry:

  `poiseuille`  a slit of width H between two SDF walls, driven by a uniform body force G.
                With Navier slip lambda at BOTH walls the exact solution is
                    u(z) = (G / 2mu) * ( (z - z0)(z1 - z) + lambda H ),   H = z1 - z0,
                a QUADRATIC that satisfies the Robin condition exactly, so the Robust-Scaled
                closure (which fits a quadratic through u_m, u_c, u_g and imposes the Robin
                condition at the crossing) must reproduce it to the storage floor.

  `couette`     the work order's own gate: an SDF wall with slip at z0 and the +z domain face
                driven at U (as an INFLOW face -- a type-1 domain WALL ignores its tangential
                velocity, measured), giving the linear profile
                    u(z) = U (z - z0 + lambda) / (H + lambda),   H = nz - z0.

  `floor`       the poiseuille case over a lambda ladder down to 1e-10, to MEASURE the float
                storage floor below which the Robin datum is indistinguishable from no-slip
                (the closure is stored in float: D = theta(1+theta) + lambda(1+2 theta)).

Wall placement is QUARTER-INTEGER throughout (WO-S finding 5 / WO-V7 finding 1).

Usage: PYTHONPATH=$PWD/build_omp python tests/study/wall_slip.py [poiseuille|couette|floor|all]
"""
import math
import sys

import numpy as np

import peclet.flow as pf

NX, NY = 8, 8
MU = 1.0
G = 1e-3          # body force per unit volume


def slab_sdf(nz, z0, z1):
    """Exact signed distance to the slab boundary; > 0 in the fluid (z0 < z < z1)."""
    z = (np.arange(nz) + 0.5)[None, None, :]
    d = np.minimum(z - z0, z1 - z)
    return np.asfortranarray(np.broadcast_to(d, (NX, NY, nz)).astype(float).copy())


def _solve(s, steps, tol=1e-13):
    prev = None
    for i in range(steps):
        s.step()
        u = np.asarray(s.get_u())[NX // 2, NY // 2, :].copy()
        if prev is not None:
            d = np.abs(u - prev).max() / max(np.abs(u).max(), 1e-300)
            if d < tol:
                return i + 1, d
        prev = u
    return steps, float("nan")


def poiseuille(nz, z0, z1, lam, steps=4000, verbose=True):
    H = z1 - z0
    s = pf.Solver(NX, NY, nz)
    s.set_rho(1.0)
    s.set_mu(MU)
    s.set_dt(50.0)
    s.set_advection(False)          # Stokes: the exact profile is a steady Stokes solution
    s.set_body_force(G, 0.0, 0.0)
    s.set_pressure_multigrid(True, levels=3)
    s.set_pressure_solver_params(400)
    s.set_velocity_solver_params(400)
    s.set_solid(slab_sdf(nz, z0, z1), cutcell_pressure=True)
    if lam > 0:
        s.set_wall_slip_length(lam)
    n, d = _solve(s, steps)
    u = np.asarray(s.get_u())[NX // 2, NY // 2, :]
    zc = np.arange(nz) + 0.5
    fluid = (zc > z0) & (zc < z1)
    ue = (G / (2 * MU)) * ((zc - z0) * (z1 - zc) + lam * H)
    err = np.abs(u[fluid] - ue[fluid]).max()
    scale = np.abs(ue[fluid]).max()
    # the analytic slip velocity AT the wall, and the one the discrete profile extrapolates to
    us = (G / (2 * MU)) * lam * H
    if verbose:
        print(f"  lambda {lam:<10.4g} H {H:g}  steps {n:5d} (dU {d:.2e})  "
              f"max|u - u_exact| {err:.4e}  rel {err/scale:.4e}   u_slip(exact) {us:.6e}   "
              f"u_centre {u[fluid][len(u[fluid])//2]:.8e}")
    return dict(lam=lam, err=err, rel=err / scale, u=u.copy(), ue=ue, fluid=fluid,
                iters=s.last_pressure_iterations(), sandwich=s.wall_slip_sandwich_cells(),
                div=s.max_open_divergence_projected(), u_slip=us, steps=n)


def couette(nz, z0, lam, U=1.0, steps=200):
    """The work order's gate 1: plane COUETTE over a flat SDF wall carrying the Navier slip.

    The bottom boundary is an SDF wall at `z0` with slip lambda; the top is the +z DOMAIN face
    driven at U.  A type-1 domain wall IGNORES its tangential velocity (measured: a plain channel
    with `set_domain_bc(5, 1, U, 0, 0)` stays identically 0 after 400 steps) -- the moving lid has
    to be an INFLOW face (type 2) whose velocity is purely tangential, which is what
    `tests/kokkos_mpi/test_varmu_mpi.cpp` uses.  Exact steady Stokes profile:

        u(z) = U (z - z0 + lambda) / (H + lambda),    H = nz - z0

    (no-slip at the lid, Navier slip at the SDF wall), a LINEAR profile, so the closure has to be
    exact on it.
    """
    H = float(nz) - z0
    s = pf.Solver(NX, NY, nz)
    s.set_rho(1.0)
    s.set_mu(MU)
    s.set_dt(100.0)
    s.set_advection(False)
    s.set_domain_bc(4, 1, 0.0, 0.0, 0.0)          # -z: wall, buried in the solid
    s.set_domain_bc(5, 2, U, 0.0, 0.0)            # +z: the moving lid
    s.set_pressure_pcg(True, 400, 1e-12)
    s.set_velocity_solver_params(400)
    z = (np.arange(nz) + 0.5)[None, None, :]
    d = np.broadcast_to(z - z0, (NX, NY, nz)).astype(float).copy()
    s.set_solid(np.asfortranarray(d), cutcell_pressure=True)
    if lam > 0:
        s.set_wall_slip_length(lam)
    n, dd = _solve(s, steps)
    u = np.asarray(s.get_u())[NX // 2, NY // 2, :]
    zc = np.arange(nz) + 0.5
    fluid = zc > z0
    ue = U * (zc - z0 + lam) / (H + lam)
    err = np.abs(u[fluid] - ue[fluid]).max()
    print(f"  lambda {lam:<10.4g} H {H:g}  steps {n:5d} (dU {dd:.2e})  "
          f"max|u - u_exact| {err:.4e}  rel {err/U:.4e}   "
          f"u(wall,exact) {U*lam/(H+lam):.6e}  u(first fluid cell) {u[fluid][0]:.8e} "
          f"vs exact {ue[fluid][0]:.8e}   press {s.last_pressure_iterations()}")
    return dict(lam=lam, err=err, rel=err / U)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nz, z0, z1 = 40, 8.25, 32.25
    if which in ("poiseuille", "all"):
        print("=" * 96)
        print(f"GATE 1a  slip POISEUILLE, {NX}x{NY}x{nz}, SDF walls at z = {z0} / {z1} "
              f"(quarter-integer), H = {z1-z0:g}, mu = {MU:g}, G = {G:g}")
        print("  exact: u(z) = (G/2mu)((z-z0)(z1-z) + lambda H); theta = 0.25 (low wall) / "
              "0.75 (high wall)")
        print("=" * 96)
        base = None
        for lam in (0.0, 0.02, 0.05, 0.1, 0.3, 0.5, 1.0):
            r = poiseuille(nz, z0, z1, lam)
            if lam == 0.0:
                base = r
        print(f"  sandwich cells (must be 0,0,0): {base['sandwich']}, "
              f"max|div| {base['div']:.2e}, pressure iterations {base['iters']}")
    if which in ("couette", "all"):
        print("\n" + "=" * 96)
        print(f"GATE 1b  slip COUETTE, SDF wall at z = {z0}, +z domain wall moving at U")
        print("=" * 96)
        for lam in (0.0, 0.02, 0.05, 0.1, 0.3, 0.5):
            couette(nz, z0, lam)
    if which in ("floor", "all"):
        print("\n" + "=" * 96)
        print("THE FLOAT FLOOR: the closure is stored in FLOAT, so lambda vanishes from")
        print("D = theta(1+theta) + lambda(1+2 theta) once lambda(1+2 theta) < eps_f32 * D.")
        print("Reported as the measured slip velocity u(z0) - the analytic no-slip profile,")
        print("i.e. how much of the imposed slip actually survives the storage.")
        print("=" * 96)
        r0 = poiseuille(nz, z0, z1, 0.0, verbose=False)
        for lam in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
            r = poiseuille(nz, z0, z1, lam, verbose=False)
            d = np.abs(r["u"][r["fluid"]] - r0["u"][r0["fluid"]]).max()
            expect = (G / (2 * MU)) * lam * (z1 - z0)
            print(f"  lambda {lam:<9.1e}  max|u(lambda) - u(0)| {d:.6e}   "
                  f"expected slip {expect:.6e}   ratio {d/expect if expect else float('nan'):8.4f}")


if __name__ == "__main__":
    main()
