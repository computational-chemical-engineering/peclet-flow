"""THE ghost-projection experiment: staggered Zick & Homsy drag, cut-cell vs directional
ghost-cell projection (set_ghost_projection). Success = ghost-mode order ~2 with K converging to
the same Z&H limit; also logs BiCGStab vs PCG iteration counts (solver health / compatibility
floor). Pattern: tests/study/collocated_zh_ab.py."""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.environ.get("SDFLOW_BUILD", "build_cuda2")))
import numpy as np
from peclet import flow


def lattice_sdf(N, phi=0.125):
    R = (3 * phi / (4 * np.pi)) ** (1 / 3) * N
    g = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    dx = X - 0.5 * N
    dx -= N * np.round(dx / N)
    dy = Y - 0.5 * N
    dy -= N * np.round(dy / N)
    dz = Z - 0.5 * N
    dz -= N * np.round(dz / N)
    return np.sqrt(dx * dx + dy * dy + dz * dz) - R, R


def drag(N, ghost, mu=0.1, F=1e-3, dt=80.0, warm_tol=1e-7, tail=40, max_steps=4000):
    sdf, R = lattice_sdf(N)
    lv = max(2, int(np.log2(N)) - 1)
    s = flow.Solver(N, N, N)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0, 0)
    s.set_advection(False)
    s.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=lv)
    s.set_pressure_pcg(True, 400, 1e-10)
    if ghost:
        s.set_ghost_projection(True)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev, warm, um, t0 = 0.0, None, [], time.time()
    for it in range(max_steps):
        s.step()
        m = float(s.get_u().mean())
        um.append(m)
        if warm is None:
            if it % 10 == 9:
                if it > 10 and abs(m - prev) < warm_tol * (abs(m) + 1e-30):
                    warm = it
                prev = m
        elif it - warm >= tail:
            break
    K = F * N**3 / (6 * np.pi * mu * R * np.mean(um[-tail:]))
    return K, it + 1, s.last_pressure_iterations(), s.max_open_divergence(), time.time() - t0


kref = 4.2920
print(f"Z&H K={kref}. Staggered: cut-cell projection (baseline) vs directional ghost-cell "
      f"projection (set_ghost_projection).", flush=True)
print(f"{'N':>4} | {'cutcell err%':>12} {'ord':>6} | {'ghost err%':>11} {'ord':>6} | "
      f"{'it_p':>4} {'it_g':>4} | {'div_g':>9} | secs", flush=True)
prev = {}
for N in (32, 48, 64, 96, 128):
    Kc, sc, ipc, dvc, tc = drag(N, ghost=False)
    Kg, sg, ipg, dvg, tg = drag(N, ghost=True)
    ec = 100 * (Kc - kref) / kref
    eg = 100 * (Kg - kref) / kref
    oc = np.log(abs(prev["c"]) / abs(ec)) / np.log(N / prev["N"]) if prev else float("nan")
    og = np.log(abs(prev["g"]) / abs(eg)) / np.log(N / prev["N"]) if prev else float("nan")
    print(f"{N:>4} | {ec:>+12.3f} {oc:>6.2f} | {eg:>+11.3f} {og:>6.2f} | "
          f"{ipc:>4d} {ipg:>4d} | {dvg:>9.2e} | {tc + tg:>4.0f}", flush=True)
    prev = {"c": ec, "g": eg, "N": N}
