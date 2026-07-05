"""Collocated EMBED momentum + PLAIN projection (mode 6, the Basilisk pairing) vs mode 0. Robust
convergence (drift<5e-5 over 200 steps past a floor). Success: order -> ~2, error below mode 0."""
import os, sys, time
sys.path.insert(0, os.path.abspath("build_cuda2"))
import numpy as np
from peclet import flow

def lattice_sdf(N, phi=0.125):
    R = (3 * phi / (4 * np.pi))**(1 / 3) * N
    g = np.arange(N) + 0.5; X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    dx = X - 0.5 * N; dx -= N * np.round(dx / N)
    dy = Y - 0.5 * N; dy -= N * np.round(dy / N)
    dz = Z - 0.5 * N; dz -= N * np.round(dz / N)
    return np.sqrt(dx * dx + dy * dy + dz * dz) - R, R

def drag(N, mode, mu=0.1, F=1e-3, dt=400.0, min_steps=500, max_steps=3500, dtol=5e-5):
    sdf, R = lattice_sdf(N); lv = max(2, int(np.log2(N)) - 1)
    s = flow.SolverColocated(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(F, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(200); s.set_pressure_multigrid(True, levels=lv)
    s.set_pressure_pcg(True, 500, 1e-12); s.set_face_interp(mode); s.set_pressure_warmstart(True)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    kfac = F * N**3 / (6 * np.pi * mu * R); hist = []; drift = 1.0; t0 = time.time()
    for it in range(max_steps):
        s.step()
        if it % 50 == 49:
            K = kfac / float(s.get_u().mean()); hist.append(K)
            if len(hist) >= 5:
                drift = abs(hist[-1] - hist[-5]) / abs(hist[-1])
                if it + 1 >= min_steps and drift < dtol: break
    return hist[-1], drift, it + 1, time.time() - t0

kref = 4.2920
print(f"Z&H K={kref}. mode 6 (EMBED momentum + PLAIN projection) vs mode 0.", flush=True)
print(f"{'N':>4} | {'m0 err%':>9} {'ord':>5} | {'m6 err%':>9} {'ord':>5} {'drift':>8} | steps6 secs", flush=True)
p0 = p6 = pN = None
for N in (32, 48, 64, 96, 128):
    K0, d0, s0, t0 = drag(N, 0); K6, d6, s6, t6 = drag(N, 6)
    e0 = 100 * (K0 - kref) / kref; e6 = 100 * (K6 - kref) / kref
    o0 = np.log(abs(p0) / abs(e0)) / np.log(N / pN) if p0 else float("nan")
    o6 = np.log(abs(p6) / abs(e6)) / np.log(N / pN) if p6 else float("nan")
    print(f"{N:>4} | {e0:>+9.4f} {o0:>+5.2f} | {e6:>+9.4f} {o6:>+5.2f} {d6:>8.1e} | {s6:>5d} {t0+t6:>4.0f}",
          flush=True)
    p0, p6, pN = e0, e6, N
