#!/usr/bin/env python
"""Byte gate for the peclet.flow public surface: SHA-256 of the final state of one fixed-seed run
per public entry path, at one thread.

Run:  OMP_NUM_THREADS=1 OMP_PROC_BIND=false PYTHONPATH=<build> python tests/regression/state_hash.py
      OMP_NUM_THREADS=1 PYTHONPATH=<build> mpirun -np 2 python tests/regression/state_hash.py mpi

Entry paths: the staggered periodic sphere bed (set_solid, cut-cell pressure), the collocated
solver under each of its four schemes ('ghost', 'gauge-exact', 'plain', 'embed'), an
inflow/wall/outflow channel with advection, a VoF droplet under surface tension, Boussinesq
scalar transport, the porous (volume-averaged) continuity with implicit drag, an analytic scene
with a moving instance (set_instance_motion + set_instance_transform + rebuild_geometry), and the
distributed step at np = 2 (flow.mpi_block + init_mpi). Each prints one line: the case, the
SHA-256 over the per-field hashes, and the first 12 hex digits of each field's hash (get_u/v/w/p,
plus get_vof and the scalar where present), taken over the contiguous float64 bytes.

Every run here is reproducible run-to-run at one thread (verified 2026-09-10 on host-openmp); a
refactor of the bindings or the solver that is not meant to change numerics must reproduce every
line bitwise (QUALITY_PLAN F gate). The hashes recorded at each package-F milestone are in the
commit messages.
"""
import hashlib
import sys

import numpy as np

import peclet.flow as pf

CASES = ("staggered_bed", "colocated_ghost", "colocated_gauge_exact", "colocated_plain",
         "colocated_embed", "channel", "vof_droplet", "scalar", "porous", "scene_moving")


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a), dtype=np.float64).tobytes()).hexdigest()


def packing_sdf(N, rfrac=0.18):
    R = rfrac * N
    cs = np.array([0.25 * N, 0.75 * N])
    xs = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    best = np.full((N, N, N), 1e30)
    for sx in cs:
        for sy in cs:
            for sz in cs:
                dx = X - sx; dx -= N * np.round(dx / N)
                dy = Y - sy; dy -= N * np.round(dy / N)
                dz = Z - sz; dz -= N * np.round(dz / N)
                best = np.minimum(best, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return np.asfortranarray(best)


def sphere_fractions(shape, R, c, sub=4):
    n = np.array(shape)
    off = (np.arange(sub) + 0.5) / sub
    out = np.zeros(shape)
    for ox in off:
        for oy in off:
            for oz in off:
                X, Y, Z = np.meshgrid(np.arange(n[0]) + ox, np.arange(n[1]) + oy,
                                      np.arange(n[2]) + oz, indexing="ij")
                out += ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= R * R)
    return np.asfortranarray(out / sub ** 3)


def stokes_bed(cls, N=16, steps=5, scheme=None):
    s = cls(N, N, N)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(60.0); s.set_body_force((1e-3, 0.0, 0.0))
    s.set_advection(False)
    s.diagnostics.set_velocity_solver_params(80)
    s.set_pressure_multigrid(True, 3)
    s.set_pressure_pcg(True, 200, 1e-9)
    if scheme is not None:
        s.set_collocated_scheme(scheme)
    s.set_solid(packing_sdf(N), cutcell_pressure=True)
    for _ in range(steps):
        s.step()
    return s


def case_staggered_bed():
    return stokes_bed(pf.Solver)


def case_colocated_ghost():
    return stokes_bed(pf.SolverColocated, scheme="ghost")


def case_colocated_gauge_exact():
    return stokes_bed(pf.SolverColocated, scheme="gauge-exact")


def case_colocated_plain():
    return stokes_bed(pf.SolverColocated, scheme="plain")


def case_colocated_embed():
    return stokes_bed(pf.SolverColocated, scheme="embed")


def case_channel():
    nx, ny, nz = 24, 12, 8
    s = pf.Solver(nx, ny, nz)
    s.set_rho(1.0); s.set_mu(0.05); s.set_dt(0.5)
    s.set_advection(True); s.set_advection_scheme("koren")
    s.set_domain_bc("-x", "inflow", (1.0, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow")
    s.set_domain_bc("-y", "wall"); s.set_domain_bc("+y", "wall")
    s.set_pressure_geometry(np.asfortranarray(np.full((nx, ny, nz), 10.0)))
    s.set_pressure_pcg(True, 200, 1e-9)
    for _ in range(5):
        s.step()
    return s


def case_vof_droplet():
    n, R = 16, 4.0
    s = pf.Solver(n, n, n)
    s.set_rho(1.0); s.set_mu(0.1)
    s.set_pressure_geometry(np.asfortranarray(np.full((n, n, n), 10.0)))
    s.set_pressure_pcg(True, 500, 1e-12)
    s.enable_vof()
    s.set_vof(sphere_fractions((n, n, n), R, (n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11)))
    s.set_property_model("rho", "linear", "C", [1.0, 0.0])
    s.set_surface_tension(1.0)
    s.set_dt(0.25 * s.capillary_dt())
    for _ in range(5):
        s.step()
    return s


def case_scalar():
    N, nz = 16, 4
    s = pf.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(0.05); s.set_dt(8.0)
    s.set_implicit_advection(True); s.diagnostics.set_outer_iterations(2)
    for f in ("-x", "+x", "-y", "+y"):
        s.set_domain_bc(f, "wall", (0.0, 0.0, 0.0))
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, nz), 10.0)))
    alpha = 0.05 / 0.71
    s.add_scalar("T", diffusivity=alpha, scheme="koren", iters=50)
    s.set_scalar_bc("T", "-x", "dirichlet", 1.0); s.set_scalar_bc("T", "+x", "dirichlet", 0.0)
    s.set_scalar_bc("T", "-y", "neumann", 0.0); s.set_scalar_bc("T", "+y", "neumann", 0.0)
    coeff = 1e4 * 0.05 ** 2 / (0.71 * N ** 3)
    s.set_property_model("force_y", "boussinesq", "T", [1.0, coeff, 1.0, 0.5])
    x = np.arange(N)
    T0 = np.repeat((1.0 - (x + 0.5) / N)[:, None, None], N, 1).repeat(nz, 2)
    s.set_field("T", np.asfortranarray(T0.astype(np.float64)))
    for _ in range(5):
        s.step()
    return s


def case_porous():
    N = 16
    s = pf.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(1.0); s.set_dt(0.5)
    s.set_advection(False)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, N), 10.0)))
    s.enable_drag()
    s.set_porous_continuity(True)
    s.set_field("eps", np.asfortranarray(np.full((N, N, N), 0.6)))
    s.set_field("drag_beta", np.asfortranarray(np.full((N, N, N), 4.0)))
    s.diagnostics.exchange_field("eps"); s.diagnostics.exchange_field("drag_beta")
    s.sync_porous_prev()
    s.set_body_force((0.0, 0.0, 0.2))
    for _ in range(5):
        s.step()
    return s


def case_scene_moving():
    N = 16
    s = pf.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(2.0); s.set_body_force((1e-3, 0.0, 0.0))
    s.set_advection(False)
    s.set_pressure_pcg(True, 200, 1e-9)
    node_ints = np.array([1, -1, -1], dtype=np.int32)      # kSphere
    node_reals = np.zeros(16); node_reals[0] = 3.0; node_reals[14] = 1.0; node_reals[15] = 1.0
    ii = np.array([[0, -1]], dtype=np.int32)
    ir = np.zeros((1, 18)); ir[0, 0:3] = (N / 2, N / 2, N / 2); ir[0, 6] = 1.0; ir[0, 7] = 1.0
    s.set_scene(node_ints, node_reals, ii.ravel(), ir.ravel(), periodic=True)
    s.set_instance_transform(0, [N / 2, N / 2, N / 2], [0.0, 0.0, 0.0, 1.0])
    s.set_instance_motion(0, lin_vel=[0.02, 0.0, 0.0], ang_vel=[0.0, 0.0, 0.005])
    s.set_solid_from_scene(True)
    for k in range(5):
        if k == 3:
            s.set_instance_transform(0, [N / 2 + 0.1, N / 2, N / 2], [0.0, 0.0, 0.0, 1.0])
            s.rebuild_geometry()
        s.step()
    return s


def fields(s):
    d = {"u": s.get_u(), "v": s.get_v(), "w": s.get_w(), "p": s.get_p()}
    if s.has_field("C"):
        d["C"] = s.get_vof()
    if s.has_field("T"):
        d["T"] = s.get_field("T")
    return d


def report(name, d):
    h = hashlib.sha256()
    parts = []
    for k in sorted(d):
        hk = sha(d[k]); h.update(hk.encode()); parts.append(f"{k}={hk[:12]}")
    print(f"{name:24s} {h.hexdigest()}  " + " ".join(parts), flush=True)


def case_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    N = 16
    gsdf = packing_sdf(N)
    origin, bsize = pf.mpi_block(N, N, N)
    ox, oy, oz = origin; lnx, lny, lnz = bsize
    lsdf = np.asfortranarray(gsdf[ox:ox + lnx, oy:oy + lny, oz:oz + lnz])
    s = pf.Solver(lnx, lny, lnz)
    s.init_mpi(N, N, N)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(60.0); s.set_body_force((1e-3, 0.0, 0.0))
    s.set_advection(False)
    s.diagnostics.set_velocity_solver_params(80)
    s.set_pressure_multigrid(True, 3)
    s.set_pressure_pcg(True, 200, 1e-9)
    s.set_solid(lsdf, cutcell_pressure=True)
    for _ in range(5):
        s.step()
    loc = {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in fields(s).items()}
    allf = comm.gather(loc, root=0)
    if comm.Get_rank() == 0:
        d = {k: np.concatenate([np.asarray(r[k]).ravel() for r in allf]) for k in loc}
        report(f"mpi_np{comm.Get_size()}", d)


if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(CASES)
    for c in want:
        if c == "mpi":
            case_mpi()
        else:
            report(c, fields(globals()["case_" + c]()))
