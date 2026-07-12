"""Ghost-projection closure-order comparison on staggered Zick & Homsy drag:
  (2,2) full quadratic  — 13-point nonsymmetric matrix (baseline ghost mode, cached below)
  (1,1) linear closure  — 7-point matrix, 1st-order closure everywhere
  (1,2) MIXED/deferred  — quadratic RHS (2nd-order steady constraint) on the 7-point linear
        matrix; the operator mismatch converges through the time stepping (rate ~0.4 measured
        a-priori). Expectation: (1,2) matches (2,2)'s accuracy with fewer BiCGStab iterations;
        (1,1) shows whether the linear closure's larger O(h^2) constant (or 1st-order term)
        degrades the drag. Cached columns from tests/study/staggered_zh_ghostproj.py runs."""
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


def drag(N, morder, rorder, mu=0.1, F=1e-3, dt=80.0, warm_tol=1e-7, tail=40, max_steps=4000):
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
    s.set_ghost_projection(True, matrix_order=morder, rhs_order=rorder)
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
# cached from staggered_zh_ghostproj.py (2026-07-12, build_cuda2): cutcell PCG + ghost (2,2)
CUT = {32: -0.314, 48: -0.082, 64: -0.018, 96: +0.009, 128: +0.013}
G22 = {32: (-0.609, 11), 48: (-0.256, 13), 64: (-0.144, 13), 96: (-0.066, 16), 128: (-0.039, 15)}

print(f"Z&H K={kref}. Ghost closure orders: (2,2) quad [cached], (1,1) linear, (1,2) mixed.",
      flush=True)
print(f"{'N':>4} | {'cutcell%':>9} | {'(2,2)%':>8} {'it':>3} | {'(1,1)%':>8} {'ord':>5} {'it':>3} "
      f"{'div':>8} | {'(1,2)%':>8} {'ord':>5} {'it':>3} {'div':>8} | secs", flush=True)
prev = {}
for N in (32, 48, 64, 96, 128):
    K11, s11, i11, d11, t11 = drag(N, 1, 1)
    K12, s12, i12, d12, t12 = drag(N, 1, 2)
    e11 = 100 * (K11 - kref) / kref
    e12 = 100 * (K12 - kref) / kref
    o11 = np.log(abs(prev["e11"]) / abs(e11)) / np.log(N / prev["N"]) if prev else float("nan")
    o12 = np.log(abs(prev["e12"]) / abs(e12)) / np.log(N / prev["N"]) if prev else float("nan")
    e22, i22 = G22[N]
    print(f"{N:>4} | {CUT[N]:>+9.3f} | {e22:>+8.3f} {i22:>3d} | {e11:>+8.3f} {o11:>5.2f} {i11:>3d} "
          f"{d11:>8.1e} | {e12:>+8.3f} {o12:>5.2f} {i12:>3d} {d12:>8.1e} | {t11 + t12:>4.0f}",
          flush=True)
    prev = {"e11": e11, "e12": e12, "N": N}
