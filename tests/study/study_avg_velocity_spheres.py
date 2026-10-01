#!/usr/bin/env python3
"""Grid convergence of the Stokes permeability of a simple-cubic sphere array against Zick & Homsy,
comparing WHERE the superficial velocity U_sup = <u_x> is averaged:

  * staggered        -- peclet.flow.Solver: u lives on the MAC faces; <u_x> over that face field.
  * collocated/cell  -- peclet.flow.SolverColocated: <u_x> over the cell-centred velocity.
  * collocated/FACE  -- the same SolverColocated solve, <u_x> over its projected, divergence-free MAC
                        face field (get_uf). Only the averaging location differs from the cell column.
  * amr/cell, amr/FACE -- peclet.amr.Flow on a UNIFORM octree (lmax = 0), cell mean and the mean of
                        its divergence-free face field. At lmax = 0 the octree is flow's brick, so
                        this is a cross-check of the collocated columns (amr's uniform-parity gate,
                        amr/tests/study/flow_parity, holds the two engines to 1e-5 on a ghost-
                        projected sphere). The two modules cannot share an interpreter (each
                        finalizes Kokkos under the other's Views), so the amr runs go to a subprocess.

Each solver runs its DEFAULT immersed-boundary scheme: staggered = the cut-cell (Robust-Scaled)
overlay; collocated and amr = AUTO, i.e. the fluid-only 'ghost' projection.

One sphere of volume fraction phi at the centre of a periodic unit cube (on a grid vertex for even N, as in
collocated_zh_schemes.py and amr's parity cases), rho = mu = 1, body force f = 1 along x, marched
to a steady state with the pseudo-time step at viscous number nu dt / h^2 = 6 (the cell-unit dt 60
at nu 0.1 the study was first run with), at most 1500 steps (a run that hits the cap is flagged
'*'). Drag: K = f L^3 / (6 pi mu R U_sup), error in % against the Z&H table.

Steady state (march(), the same test for every column): every 5 steps, d = the change of <u_x> over
the block and R = d / the previous block's d; a block passes when 0 < R < 1 and the geometric
remainder |d| / (1 - max(R, 0.997^5)) is below 1e-4 |<u_x>|, and three consecutive passes stop the
march. So the stopped K is within 1e-4 (relative; the resolution of the e% columns) of the
converged one, provided no mode slower than max(R, 0.997 per step) carries the remainder. Checked
against the geometric limit of 800-1200-step marches, every column at N = 14..48: the largest
|K_stop / K_inf - 1| is 7.4e-5 (collocated N = 16; gauge-exact there 8.1e-5), all others <= 6.1e-5.
Why a remainder test and not an increment test: the collocated march carries a slowly relaxing
(pi,0,0) pressure checkerboard (doc/collocated_invisible_subspace.md §11; instrument
collocated_checkerboard_tail.py) -- about 0.996 per step at N = 14..24, independent of dt, with an
N-dependent amplitude. The old test (|d| < 1e-6 |<u_x>| over 5 steps) waited 490 steps on it at
N = 14 and 16, and stopped falsely where two modes of opposite sign cross: collocated N = 18 at
step 45 with K off by 2.4e-4, staggered N = 24 and 32 at 4.6e-5 and 4.8e-5. The 0.997 floor keeps
a fast, cleanly contracting transient from hiding the slow tail beneath it.

Run (from flow/, OpenMP pool bounded; the amr columns need an amr build, otherwise they print --):

    OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=<flow build> \\
    PECLET_AMR_PYTHONPATH=<amr build> python tests/study/study_avg_velocity_spheres.py [phi] [N,N,..]

Defaults phi = 0.125, N = 16,24,32,48. AMR_MAX_N (default 48) caps the N the amr column runs at.

Result, 2026-10-01 (flow 5bcf353 + this stop test, amr 7bf183a, host-openmp -march=native,
8 threads), phi = 0.125, K_ZH = 4.292, N = 16,18,24,32,48, 3.6 min wall:

       N |   K_stag     e% | K_co_cel     e% | K_co_FAC     e% |    K_amr     e% | K_amrFAC     e% | steps s/c/a
      16 |   4.2120  -1.86 |   4.2663  -0.60 |   4.2663  -0.60 |   4.2663  -0.60 |   4.2663  -0.60 | 75/395/395
      18 |   4.2350  -1.33 |   4.2698  -0.52 |   4.2698  -0.52 |   4.2698  -0.52 |   4.2698  -0.52 | 175/335/335
      24 |   4.2573  -0.81 |   4.2791  -0.30 |   4.2791  -0.30 |   4.2791  -0.30 |   4.2791  -0.30 | 135/90/90
      32 |   4.2771  -0.35 |   4.2845  -0.18 |   4.2845  -0.18 |   4.2845  -0.18 |   4.2845  -0.18 | 200/145/145
      48 |   4.2880  -0.09 |   4.2884  -0.08 |   4.2884  -0.08 |   4.2884  -0.08 |   4.2884  -0.08 | 280/280/280

  * Every column under-predicts K and its error falls monotonically with N. Observed orders: staggered
    2.4 (N 16 -> 32) and 3.1 (24 -> 48), collocated 1.8 and 1.8 -- about 2, as flow's ghost-projection
    notes have it. The 24 -> 48 staggered figure is inflated: at -0.09 % the error is within ~8x of the
    Z&H table's own last digit (4.292 +/- 0.0005, i.e. +/- 0.012 %).
  * Face vs cell averaging no longer matters: under the ghost projection the two collocated means
    agree to <= 1e-8 relative at every N (STUDY_DIAG prints them), although the fields differ
    pointwise by up to 6e-3 of the mean. The question the study was written for -- the OLD collocated
    scheme's cell average was biased at cut cells by its openness-aware centre correction and lost
    ~1 % per grid to staggered; does the face average recover it? -- is moot for the current default,
    and on this case collocated is now the more accurate of the two at every N.
  * amr at lmax = 0 reproduces flow collocated to all printed digits (the uniform-parity result).
  * Against the old stop test (5bcf353's table) K moved by at most 1e-4: collocated N = 16
    4.2664 -> 4.2663, staggered N = 24 4.2571 -> 4.2573 and N = 32 4.2770 -> 4.2771; the steps
    went 70/490/490 -> 75/395/395 at N = 16, and 70/85 -> 135/90 at N = 24 (staggered had
    stopped early on a sign change there).

History: written against the sdflow-era module names and the retired tpx_amr (later
peclet.core.amr) module; ported to the physical-domain API of peclet.flow and peclet.amr and moved
from scripts/ to tests/study/ on 2026-10-01. Two things changed in the port beyond the names: the
sphere sits at the box centre sampled at the solver's cell_centers() (the old cell-index sampling put
it half a cell off the vertex), and the amr column stops on the same steady-state test as flow
instead of after a fixed 100 steps.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ZH_PHI = [0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236]
ZH_K = [1.096, 1.212, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1]

L = 1.0                  # periodic unit cube
RHO, MU, F = 1.0, 1.0, 1.0
BETA = 6.0               # viscous number nu dt / h^2 of the pseudo-time march
# Steady-state stop (march()): the change still to come in <u_x>, hence in K, below TOL_K relative
TOL_K, CHECK_EVERY, N_PASS, MAX_STEPS = 1e-4, 5, 3, 1500
RHO_SLOW = 0.997        # slowest per-step contraction the march has (the collocated (pi,0,0) tail)
ROUNDOFF = 1e-11        # a block change this small (relative) is converged whatever its ratio


def zh_ref(phi):
    return float(np.interp(phi, ZH_PHI, ZH_K))


def sphere_radius(phi):
    return (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0) * L


def pseudo_dt(N):
    h = L / N
    return BETA * RHO * h * h / MU


def drag_K(umean, phi):
    return F * L ** 3 / (6.0 * math.pi * MU * sphere_radius(phi) * umean)


def march(step, umean):
    """Step until the change of <u_x> still to come is below TOL_K (relative); see the docstring.

    Every CHECK_EVERY steps, d = the change of <u_x> over the block and R = d / (the previous
    block's d). A block PASSES when the increments contract without changing sign (0 < R < 1) and
    the geometric remainder |d| / (1 - max(R, RHO_SLOW**CHECK_EVERY)) is below TOL_K |<u_x>| -- or
    when |d| is at round-off. N_PASS consecutive passes stop the march. The RHO_SLOW floor keeps a
    fast, cleanly contracting transient from hiding the slow tail beneath it; the sign and
    consecutive-pass conditions reject the dip where two modes of opposite sign cross.
    """
    prev = dprev = None
    passes = 0
    for it in range(MAX_STEPS):
        step()
        if it % CHECK_EVERY != CHECK_EVERY - 1:
            continue
        m = umean()
        if prev is not None:
            d = m - prev
            ok = abs(d) <= ROUNDOFF * abs(m)
            if not ok and dprev:
                R = d / dprev
                ok = 0.0 < R < 1.0 and \
                    abs(d) / (1.0 - max(R, RHO_SLOW ** CHECK_EVERY)) < TOL_K * abs(m)
            passes = passes + 1 if ok else 0
            if passes >= N_PASS:
                return it + 1, True
            dprev = d
        prev = m
    return MAX_STEPS, False


def run_flow(flow, SolverCls, N, phi):
    """(U_cell, U_face, steps, converged) of one flow solve at steady state."""
    s = SolverCls((N, N, N), extent=(L, L, L))
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(pseudo_dt(N))
    s.set_body_force((F, 0.0, 0.0)); s.set_advection(False)
    s.diagnostics.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=max(2, int(np.log2(N)) - 1))
    s.set_pressure_pcg(True, max_iter=200, rtol=1e-8)
    cx, cy, cz = s.cell_centers()
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    c = 0.5 * L
    sdf = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - sphere_radius(phi)
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True)
    steps, ok = march(s.step, lambda: float(s.get_u().mean()))
    u, uf = s.get_u(), s.get_uf()
    if os.environ.get("STUDY_DIAG") and SolverCls is flow.SolverColocated:
        # get_uf() is the genuinely different face field, not an alias of the cell field
        print(f"      [diag N={N}] max|uf-u_cell| = {np.abs(uf - u).max():.3e}  "
              f"(means: cell {float(u.mean()):.8e} face {float(uf.mean()):.8e})", flush=True)
    return float(u.mean()), float(uf.mean()), steps, ok


def amr_worker(N, phi):
    """Subprocess body: one amr solve; prints a JSON line {u_cell, u_face, steps, converged}."""
    from peclet import amr
    oct_ = amr.Octree(cells=[N, N, N], lmax=0, origin=[0.0, 0.0, 0.0], extent=[L, L, L])
    fl = amr.Flow(oct_, density=RHO, viscosity=MU, dt=pseudo_dt(N))
    fl.set_body_force(F, 0.0, 0.0); fl.set_advection(False)
    fl.set_pressure_tolerance(1e-8)  # flow's set_pressure_pcg rtol above
    c = 0.5 * L
    fl.set_solid_spheres(np.array([[c, c, c]]), np.array([sphere_radius(phi)]), periodic=True)
    steps, ok = march(lambda: fl.step(mom_iters=200, pres_iters=200),
                      lambda: float(fl.velocity(0).mean()))
    # face_field() is one value per CSR face, six per leaf on a uniform octree; [0::6] is each
    # leaf's x-face (amr/tests/study/flow_parity/run_amr.py reads uf the same way)
    ff = fl.face_field()
    assert ff.size == 6 * fl.num_leaves, "expected six faces per leaf at lmax = 0"
    uf = ff[0::6]
    print(json.dumps({"u_cell": float(fl.velocity(0).mean()), "u_face": float(np.mean(uf)),
                      "steps": steps, "converged": ok}))


def run_amr(N, phi):
    pp = os.environ.get("PECLET_AMR_PYTHONPATH", "")
    if not pp:
        return None
    env = dict(os.environ)
    env["PYTHONPATH"] = pp + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--amr-worker", str(N), str(phi)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  amr worker failed at N={N} (exit {r.returncode}):\n{r.stdout}\n{r.stderr}")
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "scripts"))
    from _bootstrap import ensure_flow
    flow = ensure_flow()

    phi = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
    Ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [16, 24, 32, 48]
    amr_max = int(os.environ.get("AMR_MAX_N", "48"))
    kref = zh_ref(phi)

    def col(u):
        K = drag_K(u, phi)
        return f"{K:8.4f} {100 * (K - kref) / kref:+6.2f}"

    def flag(ok):
        return "" if ok else "*"

    print(f"=== Stokes permeability, SC sphere array (phi={phi}, Z&H K_ref={kref:.4f}) ===")
    print(f"{'N':>4} | {'K_stag':>8} {'e%':>6} | {'K_co_cel':>8} {'e%':>6} | {'K_co_FAC':>8} {'e%':>6}"
          f" | {'K_amr':>8} {'e%':>6} | {'K_amrFAC':>8} {'e%':>6} | steps s/c/a")
    for N in Ns:
        t0 = time.time()
        us, _, ns, oks = run_flow(flow, flow.Solver, N, phi)
        uc, uf, nc, okc = run_flow(flow, flow.SolverColocated, N, phi)
        a = run_amr(N, phi) if N <= amr_max else None
        if a is None:
            amr_cols, na = f"{'--':>8} {'--':>6} | {'--':>8} {'--':>6}", "--"
        else:
            amr_cols = f"{col(a['u_cell'])} | {col(a['u_face'])}"
            na = f"{a['steps']}{flag(a['converged'])}"
        print(f"{N:4d} | {col(us)} | {col(uc)} | {col(uf)} | {amr_cols} | "
              f"{ns}{flag(oks)}/{nc}{flag(okc)}/{na} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--amr-worker":
        amr_worker(int(sys.argv[2]), float(sys.argv[3]))
    else:
        main()
