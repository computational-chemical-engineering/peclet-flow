#!/usr/bin/env python3
"""MG-depth measurement driver (doc/mg_depth_policy.md §6 WO-0; reused by the WO-2 gates).

One process = one case at one pressure-multigrid depth. Every case takes

    --levels L   L >= 1: set_pressure_multigrid(True, L) BEFORE the geometry (an exact depth);
                 L = 0 (the default): do NOT call set_pressure_multigrid -- the solver default
                 (depth 4 on the pre-WO-2 module, AUTO after WO-2).
    --steps N    timed steps (default 30); --warm W untimed warm-up steps (default 5)
    --dump F.npz u, v, w, p (and C where present) at the end; per rank under MPI (keys u_r<k>, ...,
                 origin_r<k>)
    --json F     the per-row metrics below plus the per-step lists
    --rtol R     the case's Krylov tolerance, where the case has a Krylov driver (see each case)
    --line-relaxation on|off
                 diagnostics.set_pressure_line_relaxation (doc/mg_depth_policy.md A2.4) before the
                 geometry; absent: no call (the solver default, on since WO-7)
    --vcycle-symmetric on|off
                 diagnostics.set_pressure_vcycle_symmetric (doc/mg_depth_policy.md A3) before the
                 geometry; absent: no call (the solver default)

Cases (--case):
    swarm      periodic bubble swarm, case A groups, single rank: the --single path of
               tests/study/vof_perf/weak_swarm.py, code for code (--cells n, default 192;
               --d-over-h 20, --bubbles-per-axis 2). --driver pcg (default; MG-PCG cap 800 at
               --rtol, default 1e-8, as weak_swarm) or --driver default (no driver call: the
               variable-density default, Chebyshev).
               --dims NX NY NZ: an anisotropic box instead of --cells^3 (row M10: an axis that
               stops coarsening early, doc/mg_depth_policy_log.md); the bubble lattice then has
               max(1, N_axis // 64) bubbles along each axis (measurement choice) unless
               --bubbles-per-axis is given, which stays one count for every axis.
    bc         the peclet-examples bubble column (run_peclet.build(), MG-PCG rtol 1e-8 as built)
               at --cells-per-d S (default 32: 256 x 192 x 128), FRESH START from the seeded
               spheres at rest (the 128 x 96 x 64 checkpoint does not exist at other
               resolutions). --rtol re-selects MG-PCG cap 800 at R.
    zh_sphere / random_spheres
               the cut-cell Stokes geometries of tests/regression/sdflow_regression.py at --n N
               with its CFG (MG-PCG cap 300 rtol 1e-8; --rtol overrides), marched to steady with
               its stop test (--max-steps, default 400; --steps is ignored, --warm 0). Records K / k*.
               --solver colocated: SolverColocated with NO scheme call, i.e. the AUTO default
               ('ghost': BiCGStab preconditioned by the V-cycle, at the MG-PCG cap / rtol above);
               --dump then also writes the fluid-cell mask `fluid` (sdf > 0 at the cell centres,
               where every colocated field lives). Row M9 (doc/mg_depth_policy_log.md); its exit
               trace is the solver's PECLET_FLOW_MG_DEBUG=2 line (debug output only).
    channel    single-phase all-fluid channel --nx x --ny x --nz (default 2048 x 64 x 64):
               periodic x, walls on +-y and +-z, body force along x, from rest plus a fixed-seed
               non-solenoidal low-mode perturbation (--perturb, default 1e-2). Single rank:
               the default driver (MG-PCG, auto-enabled); --rtol selects MG-PCG cap 800 at R.
               Under mpirun (np > 1): mpi_block + init_mpi, the multi-rank default (the
               standalone V-cycle, no Krylov driver selected; --rtol is refused).
    anchored   inflow/outflow channel --nx x --ny x --nz (default 128 x 32 x 32): uniform inflow
               U at -x, outflow at +x, walls on +-y, +-z, all fluid; MG-PCG cap --maxit
               (default 300) at --rtol (default 1e-12) -- the divergence floor at an identical
               budget at every depth.
    porous     CFD-DEM-style fixed porous bed --n^3 (default 128): add 'eps' via
               set_porous_continuity(True) with implicit drag (enable_drag), drag_beta from the
               viscous Ergun term, advection + implicit advection ON (as peclet_coupling's
               CfdDem), periodic, body force along x; the driver is the solver default for that
               configuration (no driver call). --no-drag drops enable_drag.

Per step: wall ms of step() alone, last_pressure_iterations(), max_open_divergence_projected()
(read after the step, outside the timer), pressure_vcycle_precision(), pressure_solve_failed(),
and for the porous case max_porous_residual() (the continuity the porous projection solves).
At the end: the level ratios, the bottom's global dims, the load average before/after. The bottom
ENGINE is not visible through the API; run_matrix.sh reads it from a short probe run with
PECLET_FLOW_MG_DEBUG=1 PECLET_FLOW_AGMG_DEBUG=1 (debug output only).
Rank 0 prints one `RESULT key=value ...` line.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

A = list(sys.argv)


def arg(name, default, cast=str):
    return cast(A[A.index(name) + 1]) if name in A else default


CASE = arg("--case", None)
if CASE is None:
    sys.exit("mgd_case: --case swarm|bc|zh_sphere|random_spheres|channel|anchored|porous")
LEVELS = arg("--levels", 0, int)
STEPS = arg("--steps", 30, int)
WARM = arg("--warm", 5, int)
DUMP = arg("--dump", None)
JSON = arg("--json", None)
RTOL = arg("--rtol", None, float)
TAG = arg("--tag", "")
LINE = arg("--line-relaxation", None)
if LINE not in (None, "on", "off"):
    sys.exit("mgd_case: --line-relaxation on|off")
SYM = arg("--vcycle-symmetric", None)
if SYM not in (None, "on", "off"):
    sys.exit("mgd_case: --vcycle-symmetric on|off")
load0 = os.getloadavg()

# MPI only when launched under mpirun (the channel at np > 1)
NP_ENV = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
if NP_ENV > 1:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
else:
    comm, rank, size = None, 0, 1

info = {}            # case-specific end-of-run numbers
FLUID_MASK = None    # colocated IBM: dumped beside the fields (G-CONV (b) on fluid cells)
fields_C = False     # dump C too


def depth(s):
    """--levels L >= 1: an exact depth, before the geometry; 0: leave the solver default.
    --line-relaxation on|off: the A2.4 line smoother switch (absent: the default).
    --vcycle-symmetric on|off: the A3 symmetric V-cycle switch (absent: the default)."""
    if LEVELS >= 1:
        s.set_pressure_multigrid(True, LEVELS)
    if LINE is not None:
        s.diagnostics.set_pressure_line_relaxation(LINE == "on")
    if SYM is not None:
        s.diagnostics.set_pressure_vcycle_symmetric(SYM == "on")


# ------------------------------------------------------------------------------------- swarm
def build_swarm():
    """weak_swarm.py --single, code for code (case A groups, cell units)."""
    global fields_C
    import peclet.flow as pf
    AR, BO, RHO_RATIO, MU_RATIO = 29.9, 2.0, 0.1, 0.1
    RHO_L, G, D = 1.0, 1.0, 1.0
    RHO_G = RHO_RATIO * RHO_L
    MU_L = np.sqrt(RHO_L * abs(RHO_G - RHO_L) * G * D ** 3) / AR
    MU_G = MU_RATIO * MU_L
    SIGMA = abs(RHO_G - RHO_L) * G * D ** 2 / BO
    SEED, PERTURB = 20261024, 0.25
    CELLS = arg("--cells", 192, int)
    S = arg("--d-over-h", 20.0, float)
    DRIVER = arg("--driver", "pcg")
    rtol = RTOL if RTOL is not None else 1e-8
    if "--dims" in A:   # anisotropic box (M10); default lattice max(1, N // 64) per axis
        k0 = A.index("--dims")
        NX, NY, NZ = (int(v) for v in A[k0 + 1:k0 + 4])
        Bv = np.array([arg("--bubbles-per-axis", max(1, n // 64), int) for n in (NX, NY, NZ)])
    else:               # the cube, code for code as before
        NX = NY = NZ = CELLS
        Bv = np.full(3, arg("--bubbles-per-axis", 2, int))
    rng = np.random.default_rng(SEED)
    spv = np.array([NX, NY, NZ]) / Bv
    gapv = spv - S
    if (gapv <= 0).any():
        sys.exit(f"swarm: bubbles (d/h {S}) do not fit a lattice spacing of {spv} cells")
    seeds = []
    for k in range(Bv[2]):
        for j in range(Bv[1]):
            for i in range(Bv[0]):
                c = (np.array([i, j, k]) + 0.5) * spv + (rng.random(3) - 0.5) * PERTURB * gapv
                seeds.append((float(c[0]), float(c[1]), float(c[2]), 0.5 * S))
    siz = (NX, NY, NZ)
    s = pf.Solver(*siz)
    depth(s)
    s.set_rho(RHO_L)
    s.set_mu(S * S * MU_L)
    s.set_pressure_geometry(np.full(siz, 10.0, order="F"))
    s.enable_vof()
    s.set_vof(np.zeros(siz, order="F"))
    s.set_property_model("rho", "linear", "C", [RHO_L, RHO_G - RHO_L])
    s.set_property_model("mu", "linear", "C", [S * S * MU_L, S * S * (MU_G - MU_L)])
    s.set_surface_tension(S ** 3 * SIGMA)
    s.enable_vof_blocks(seeds)
    s.enable_vof_block_csf()
    vol = sum(b["volume"] for b in s.diagnostics.vof_block_stats())
    alpha = vol / (NX * NY * NZ)
    rho_av = RHO_L + (RHO_G - RHO_L) * alpha
    s.set_property_model("force_x", "linear", "C",
                         [-S * (RHO_L - rho_av) * G, -S * (RHO_G - RHO_L) * G])
    s.set_superficial_velocity(True, "x", 0.0)
    if DRIVER == "pcg":
        s.set_pressure_pcg(True, 800, rtol)   # last (the rho closure re-selects); case A's driver
    elif DRIVER != "default":
        sys.exit("swarm: --driver pcg|default")
    s.update_properties()
    fields_C = True
    info.update(cells=CELLS if "--dims" not in A else [NX, NY, NZ], d_over_h=S, bubbles=len(seeds),
                alpha=alpha, driver=DRIVER,
                rtol=rtol if DRIVER == "pcg" else "solver default")

    def dt_of():
        lim = s.vof_step_limits()
        return 0.25 * min(lim["cfl_dt"], lim["capillary_dt"])
    return s, (NX, NY, NZ), dt_of, None


# ------------------------------------------------------------------------------- bubble column
def build_bc():
    global fields_C
    spd = arg("--cells-per-d", 32, int)
    # the example's own resolution parameter; read by case.py at import (an example-script input,
    # not a solver setting)
    os.environ["BUBBLE_COLUMN_CELLS_PER_D"] = str(spd)
    sys.path.insert(0, arg("--case-dir", os.path.expanduser(
        "~/Codes/peclet-examples/benchmarks/bubble-column/scripts")))
    import run_peclet as rp
    if LEVELS >= 1:
        orig = rp.pf.Solver

        class _Shim:  # set_pressure_multigrid right after construction, as prof.py --levels
            def __new__(cls, *a, **k):
                o = orig(*a, **k)
                o.set_pressure_multigrid(True, LEVELS)
                return o
        rp.pf.Solver = _Shim
    s = rp.build(None)
    if SYM is not None:  # the A3 switch is live (read at V-cycle time): after the build
        s.diagnostics.set_pressure_vcycle_symmetric(SYM == "on")
    if RTOL is not None:
        s.set_pressure_pcg(True, 800, RTOL)
    fields_C = True
    info.update(cells_per_d=spd, start="fresh (seeded spheres at rest)",
                rtol=RTOL if RTOL is not None else 1e-8)

    def dt_of():
        lim = s.vof_step_limits()
        return min(0.25 * lim["cfl_dt"], 0.25 * lim["capillary_dt"])
    return s, (rp.NX, rp.NY, rp.NZ), dt_of, None


# ------------------------------------------------------------------------------------- IBM
def _ibm_sdf(name, N):
    g = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    if name == "zh_sphere":   # sdflow_regression.sdf_zh_sphere (phi 0.216)
        phi = 0.216
        R = (phi * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * N
        c = N / 2.0
        sdf = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R
        return sdf, R
    # sdflow_regression.sdf_random_spheres (n 8, r_frac 0.18, jit 0.06, seed 12345)
    rng = np.random.default_rng(12345)
    R = 0.18 * N
    base = np.array([[(i + 0.5) / 2.0, (j + 0.5) / 2.0, (k + 0.5) / 2.0]
                     for i in range(2) for j in range(2) for k in range(2)])
    centres = ((base + 0.06 * rng.standard_normal(base.shape)) % 1.0) * N
    sdf = np.full((N, N, N), 1e30)
    mi = lambda d: d - N * np.round(d / N)
    for cx, cy, cz in centres:
        dx, dy, dz = mi(X - cx), mi(Y - cy), mi(Z - cz)
        sdf = np.minimum(sdf, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return sdf, R


def build_ibm():
    import peclet.flow as pf
    N = arg("--n", 128, int)
    CFG = dict(rho=1.0, mu=0.1, dt=60.0, F=1e-3, vel_sweeps=80, pcg_maxit=300, pcg_rtol=1e-8)
    rtol = RTOL if RTOL is not None else CFG["pcg_rtol"]
    sdf, R = _ibm_sdf(CASE, N)
    solver = arg("--solver", "staggered")
    if solver not in ("staggered", "colocated"):
        sys.exit("mgd_case: --solver staggered|colocated")
    s = (pf.SolverColocated if solver == "colocated" else pf.Solver)(N, N, N)
    s.set_rho(CFG["rho"]); s.set_mu(CFG["mu"]); s.set_dt(CFG["dt"])
    s.set_body_force((CFG["F"], 0.0, 0.0))
    s.set_advection(False)  # creeping Stokes
    s.diagnostics.set_velocity_solver_params(CFG["vel_sweeps"])
    depth(s)
    s.set_pressure_pcg(True, CFG["pcg_maxit"], rtol)
    s.set_solid(sdf, cutcell_pressure=True)
    info.update(n=N, R=R, rtol=rtol, cfg=CFG, solver=solver, _sdf=sdf)
    return s, (N, N, N), None, CFG


# ------------------------------------------------------------------------------------- channel
def build_channel():
    import peclet.flow as pf
    G = (arg("--nx", 2048, int), arg("--ny", 64, int), arg("--nz", 64, int))
    mu, dt, F = arg("--mu", 0.05, float), arg("--dt", 1.0, float), arg("--force", 1e-5, float)
    if size > 1:
        if RTOL is not None:
            sys.exit("channel np > 1: the standalone V-cycle default has no Krylov driver (--rtol)")
        org, siz = pf.mpi_block(*G)
    else:
        org, siz = (0, 0, 0), G
    s = pf.Solver(*siz)
    if size > 1:
        s.init_mpi(*G)
    depth(s)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    for f in ("-y", "+y", "-z", "+z"):
        s.set_domain_bc(f, "wall", (0.0, 0.0, 0.0))
    s.set_pressure_geometry(np.full(tuple(siz), 10.0, order="F"))
    s.set_body_force((F, 0.0, 0.0))
    if RTOL is not None:
        s.set_pressure_pcg(True, 800, RTOL)
    # a fixed-seed, non-solenoidal low-mode initial perturbation (8 modes per component, amplitude
    # --perturb), evaluated at GLOBAL cell coordinates so every np sees the same field: without it
    # the body-force channel is unidirectional and the projection has nothing to do
    amp = arg("--perturb", 1e-2, float)
    rng = np.random.default_rng(20261024)
    xs = [np.arange(org[d], org[d] + siz[d]) + 0.5 for d in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    for c in range(3):
        a = np.zeros(tuple(siz))
        for _ in range(8):
            n = (rng.integers(1, 17), rng.integers(1, 5), rng.integers(1, 5))
            ph = rng.random() * 2 * np.pi
            a += np.sin(2 * np.pi * (n[0] * X / G[0] + n[1] * Y / G[1] + n[2] * Z / G[2]) + ph)
        s.set_velocity(c, np.asfortranarray(amp / 8.0 * a))
    del X, Y, Z
    info.update(mu=mu, dt=dt, force=F, origin=list(org), block=list(siz), perturb=amp,
                driver=("pcg rtol %g" % RTOL) if RTOL is not None else
                ("default (standalone V-cycle)" if size > 1 else "default (MG-PCG)"))
    return s, G, None, None


def build_anchored():
    import peclet.flow as pf
    G = (arg("--nx", 128, int), arg("--ny", 32, int), arg("--nz", 32, int))
    mu, dt, U = arg("--mu", 0.1, float), arg("--dt", 1.0, float), arg("--u-in", 0.05, float)
    maxit = arg("--maxit", 300, int)
    rtol = RTOL if RTOL is not None else 1e-12
    s = pf.Solver(*G)
    depth(s)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    s.set_domain_bc("-x", "inflow", (U, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow", (0.0, 0.0, 0.0))
    for f in ("-y", "+y", "-z", "+z"):
        s.set_domain_bc(f, "wall", (0.0, 0.0, 0.0))
    s.set_pressure_geometry(np.full(G, 10.0, order="F"))
    s.set_pressure_pcg(True, maxit, rtol)
    info.update(mu=mu, dt=dt, u_in=U, maxit=maxit, rtol=rtol)
    return s, G, None, None


def build_porous():
    import peclet.flow as pf
    N = arg("--n", 128, int)
    mu, dt, F, dp = arg("--mu", 0.05, float), arg("--dt", 1.0, float), 1e-4, 4.0
    drag = "--no-drag" not in A
    G = (N, N, N)
    # a fixed bed: eps 0.45 +- 0.1 (smoothed fixed-seed noise) for x < 0.6 N, 1 above, a smooth
    # 4-cell transition; drag_beta = the viscous Ergun term 150 mu (1-eps)^2 / (eps dp^2)
    rng = np.random.default_rng(20261024)
    noise = rng.standard_normal(G)
    k = np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    noise = np.real(np.fft.ifftn(np.fft.fftn(noise) * np.exp(-(KX**2 + KY**2 + KZ**2) * (N / 8.0) ** 2)))
    noise /= np.abs(noise).max()
    x = np.arange(N) + 0.5
    w = 0.5 * (1.0 - np.tanh((x - 0.6 * N) / 4.0))[:, None, None]
    eps = np.asfortranarray(1.0 - w * (0.55 - 0.1 * noise))
    beta = np.asfortranarray(150.0 * mu * (1.0 - eps) ** 2 / (eps * dp * dp))
    s = pf.Solver(*G)
    depth(s)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    s.set_advection(True)
    s.set_implicit_advection(True)
    s.set_pressure_geometry(np.full(G, 10.0, order="F"))
    if drag:
        s.enable_drag()
    s.set_porous_continuity(True)
    s.set_field("eps", eps)
    s.diagnostics.exchange_field("eps")
    if drag:
        s.set_field("drag_beta", beta)
        s.diagnostics.exchange_field("drag_beta")
    s.sync_porous_prev()
    s.set_body_force((F, 0.0, 0.0))
    if RTOL is not None:
        sys.exit("porous: the default driver only (no --rtol)")
    info.update(n=N, mu=mu, dt=dt, force=F, dp=dp, drag=drag, eps_min=float(eps.min()),
                beta_max=float(beta.max()) if drag else 0.0)
    return s, G, None, None


BUILD = {"swarm": build_swarm, "bc": build_bc, "zh_sphere": build_ibm, "random_spheres": build_ibm,
         "channel": build_channel, "anchored": build_anchored, "porous": build_porous}
if CASE not in BUILD:
    sys.exit(f"mgd_case: unknown case {CASE}")
t_setup0 = time.perf_counter()
s, GRID, dt_of, IBM = BUILD[CASE]()
if comm is not None:
    comm.Barrier()
t_setup = time.perf_counter() - t_setup0
import peclet.flow as pf

step_ms, iters, divs, prec, failed, pres = [], [], [], [], [], []
POROUS = CASE == "porous"


def one(timed):
    if dt_of is not None:
        s.set_dt(dt_of())
    t0 = time.perf_counter()
    s.step()
    t1 = time.perf_counter()
    if timed:
        step_ms.append(1000.0 * (t1 - t0))
        iters.append(int(s.diagnostics.last_pressure_iterations()))
        divs.append(float(s.max_open_divergence_projected()))
        prec.append(s.diagnostics.pressure_vcycle_precision())
        failed.append(bool(s.diagnostics.pressure_solve_failed()))
        if POROUS:  # the porous continuity residual (div(open u) is -d(eps)/dt-shaped there)
            pres.append(float(s.max_porous_residual()))


if IBM is None:
    for _ in range(WARM):
        one(False)
    if comm is not None:
        comm.Barrier()
    w0 = time.perf_counter()
    for _ in range(STEPS):
        one(True)
    if comm is not None:
        comm.Barrier()
    wall = time.perf_counter() - w0
else:   # sdflow_regression.run_case's march and stop test (check every 5, min 15, tol 1e-5, max 400)
    prev = 0.0
    w0 = time.perf_counter()
    for it in range(arg("--max-steps", 400, int)):
        one(True)
        if it % 5 == 4:
            m = float(s.get_u().mean())
            if it >= 15 and abs(m - prev) < 1e-5 * (abs(m) + 1e-30):
                break
            prev = m
    wall = time.perf_counter() - w0
    STEPS, WARM = len(step_ms), 0
    u = s.get_u()
    umean = float(u.mean())
    sdf = info.pop("_sdf")
    if info.get("solver") == "colocated":
        FLUID_MASK = sdf > 0.0
    deep = sdf < -2.0
    if CASE == "zh_sphere":
        info["K"] = IBM["F"] * GRID[0] ** 3 / (6.0 * np.pi * IBM["mu"] * info["R"] * umean)
    else:
        info["k_star"] = IBM["mu"] * umean / (IBM["F"] * GRID[0] ** 2)
    info["divergence_end"] = float(s.max_open_divergence())
    info["max_u_solid"] = float(np.abs(u[deep]).max()) if deep.any() else 0.0
    info["iters_median_2nd_half"] = float(np.median(iters[len(iters) // 2:]))
    info["iters_total"] = int(sum(iters))

# the hierarchy: per-level ratios; the bottom's global dims = grid / prod(ratios of the levels above)
ratios = [tuple(int(v) for v in r) for r in s.diagnostics.pressure_mg_level_ratios()]
g = list(GRID)
dims = [tuple(g)]
for r in ratios[:-1]:
    g = [g[0] // r[0], g[1] // r[1], g[2] // r[2]]
    dims.append(tuple(g))
bottom = dims[-1]

h = hashlib.sha256()
fx = {"u": s.get_u, "v": s.get_v, "w": s.get_w, "p": lambda: s.get_field("p")}
if fields_C:
    fx["C"] = lambda: s.get_field("C")
arrs = {k: np.ascontiguousarray(f()) for k, f in fx.items()}
for k in arrs:
    h.update(arrs[k].tobytes())
if DUMP:
    if comm is None:
        np.savez(DUMP, **arrs, **({} if FLUID_MASK is None else {"fluid": FLUID_MASK}))
    else:
        allv = comm.gather((list(info.get("origin", [0, 0, 0])), arrs), root=0)
        if rank == 0:
            out = {}
            for r, (o, a) in enumerate(allv):
                out[f"origin_r{r}"] = np.asarray(o)
                for k, v in a.items():
                    out[f"{k}_r{r}"] = v
            np.savez(DUMP, **out)
digest = h.hexdigest()
if comm is not None:
    ds = comm.gather(digest, root=0)
    if rank == 0:
        gg = hashlib.sha256()
        for d in ds:
            gg.update(d.encode())
        digest = gg.hexdigest()
load1 = os.getloadavg()

if rank == 0:
    sm = np.asarray(step_ms)
    prec_flips = sum(1 for a, b in zip(prec, prec[1:]) if a == "fp32" and b == "fp64")
    res = {
        "case": CASE, "tag": TAG, "np": size, "grid": list(GRID), "levels_req": LEVELS,
        "line_relaxation": LINE or "default", "vcycle_symmetric": SYM or "default",
        "levels": len(ratios), "ratios": [list(r) for r in ratios], "bottom": list(bottom),
        "bottom_cells": int(np.prod(bottom)), "dims": [list(d) for d in dims],
        "space": str(getattr(pf, "execution_space", "?")),
        "flow_module": os.path.dirname(pf.__file__),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "warm": WARM, "steps": STEPS, "setup_s": t_setup, "wall_ms": 1000.0 * wall / max(STEPS, 1),
        "step_ms_mean": float(sm.mean()), "step_ms_median": float(np.median(sm)),
        "iters_mean": float(np.mean(iters)), "iters_min": int(min(iters)),
        "iters_max": int(max(iters)), "div_max": float(max(divs)), "div_last": divs[-1],
        "prec_first": prec[0], "prec_last": prec[-1], "fp32_restart_flips": prec_flips,
        "solve_failed": int(sum(failed)),
        "cheb_restarts": int(getattr(s.diagnostics, "num_pressure_chebyshev_restarts", -1)),
        "load_before": list(load0), "load_after": list(load1), "hash": digest,
    }
    res.update({k: v for k, v in info.items()})
    if POROUS:
        res.update(porous_res_max=max(pres), porous_res_last=pres[-1])
    keep = ("case", "tag", "np", "levels_req", "levels", "bottom", "steps", "wall_ms",
            "step_ms_median", "iters_mean", "iters_min", "iters_max", "div_max", "prec_last",
            "fp32_restart_flips", "solve_failed", "cheb_restarts", "porous_res_max", "K", "k_star")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return "x".join(map(str, v)) if isinstance(v, list) else str(v)
    print("RESULT " + " ".join(f"{k}={fmt(res[k])}" for k in keep if k in res)
          + f" load={load0[0]:.1f}/{load1[0]:.1f}", flush=True)
    if JSON:
        res.update(step_ms_list=step_ms, iters_list=iters, div_list=divs, prec_list=prec,
                   failed_list=failed, porous_res_list=pres)
        with open(JSON, "w") as fo:
            json.dump(res, fo, indent=1, default=str)
