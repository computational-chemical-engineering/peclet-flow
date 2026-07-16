"""THE collocated ghost-projection experiment: Zick & Homsy drag, SolverColocated mode-0
(plain average + central grad(P), 1st order) vs the directional ghost-cell projection
(set_ghost_projection: ghost-closed divergence of the face-averaged field + gpCenterGrad
predictor/correction). Success = ghost order ~2 converging to the same Z&H limit — closing the
open problem of doc/collocated_second_order_open_problem.md. Pattern:
tests/study/staggered_zh_ghostproj.py (same warm-detector + tail protocol as the staggered
experiment and the mode-0 baselines)."""
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


def drag(N, ghost, orders=(1, 2), mu=0.1, F=1e-3, dt=80.0, warm_tol=1e-7, tail=40,
         max_steps=4000):
    sdf, R = lattice_sdf(N)
    lv = max(2, int(np.log2(N)) - 1)
    s = flow.SolverColocated(N, N, N)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0, 0)
    s.set_advection(False)
    s.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=lv)
    s.set_pressure_pcg(True, 400, 1e-10)
    if ghost:
        s.set_ghost_projection(True, *orders)
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


if __name__ == "__main__":
    kref = 4.2920
    print(f"Z&H K={kref}. Collocated: mode-0 (baseline, 1st order) vs directional ghost-cell "
          f"projection (1,2).", flush=True)
    print(f"{'N':>4} | {'mode0 err%':>10} {'ord':>6} | {'ghost err%':>11} {'ord':>6} | "
          f"{'it_0':>4} {'it_g':>4} | {'div_g':>9} | secs", flush=True)
    prev = {}
    for N in (32, 48, 64, 96, 128):
        Kc, sc, ipc, dvc, tc = drag(N, ghost=False)
        Kg, sg, ipg, dvg, tg = drag(N, ghost=True)
        ec = 100 * (Kc - kref) / kref
        eg = 100 * (Kg - kref) / kref
        oc = np.log(abs(prev["c"]) / abs(ec)) / np.log(N / prev["N"]) if prev else float("nan")
        og = np.log(abs(prev["g"]) / abs(eg)) / np.log(N / prev["N"]) if prev else float("nan")
        print(f"{N:>4} | {ec:>+10.3f} {oc:>6.2f} | {eg:>+11.3f} {og:>6.2f} | "
              f"{ipc:>4d} {ipg:>4d} | {dvg:>9.2e} | {tc + tg:>4.0f}", flush=True)
        prev = {"c": ec, "g": eg, "N": N}
