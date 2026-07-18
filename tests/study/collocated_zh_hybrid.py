"""Z&H drag for the collocated cutcell-ghost HYBRID modes (set_face_interp 9/10): mode-0's
aperture projection with the directional gpCenterGrad -grad(P)/cell correction (9), plus the
open-centroid wall-aware constraint quadrature (10). Question: how much of the mode-0
first-order drag was the O(1/h) gradient defect alone (9), and does the a-priori-O(h^2)
open-centroid flux quadrature finally pay off once paired with a telescoping 2nd-order force
(10)? Baselines: mode-0 +1.00/+0.69/+0.60/+0.40/+0.30 %, ghost (1,2)
-0.175/-0.084/-0.056/-0.029/-0.018 % at N=32..128."""
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


def drag(N, mode, mu=0.1, F=1e-3, dt=80.0, warm_tol=1e-7, tail=40, max_steps=4000):
    sdf, R = lattice_sdf(N)
    s = flow.SolverColocated(N, N, N)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0, 0)
    s.set_advection(False)
    s.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=max(2, int(np.log2(N)) - 1))
    s.set_pressure_pcg(True, 400, 1e-10)
    if mode:
        s.set_face_interp(mode)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev, warm, um = 0.0, None, []
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
    return K, s.last_pressure_iterations()


if __name__ == "__main__":
    kref = 4.2920
    M0 = {32: +1.004, 48: +0.685, 64: +0.598, 96: +0.397, 128: +0.299}
    MG = {32: -0.175, 48: -0.084, 64: -0.056, 96: -0.029, 128: -0.018}
    print(f"Z&H K={kref}. Collocated hybrids: 9 = aperture proj + gpCenterGrad;"
          f" 10 = + open-centroid flux quadrature.", flush=True)
    print(f"{'N':>4} | {'mode0':>7} | {'ghost':>7} | {'hyb9 err%':>9} {'ord':>6} |"
          f" {'hyb10 err%':>10} {'ord':>6}", flush=True)
    prev = {}
    for N in (32, 48, 64, 96, 128):
        t0 = time.time()
        K9, i9 = drag(N, 9)
        K10, i10 = drag(N, 10)
        e9 = 100 * (K9 - kref) / kref
        e10 = 100 * (K10 - kref) / kref
        o9 = np.log(abs(prev["9"]) / abs(e9)) / np.log(N / prev["N"]) if prev else float("nan")
        o10 = np.log(abs(prev["10"]) / abs(e10)) / np.log(N / prev["N"]) if prev else float("nan")
        print(f"{N:>4} | {M0[N]:>+7.3f} | {MG[N]:>+7.3f} | {e9:>+9.3f} {o9:>6.2f} |"
              f" {e10:>+10.3f} {o10:>6.2f}   ({time.time() - t0:.0f}s)", flush=True)
        prev = {"9": e9, "10": e10, "N": N}
