"""Z&H sphere-array drag A/B: collocated EMBED momentum (setFaceInterp(5), Basilisk embed.h true-normal
dirichlet_gradient wall drag) vs mode 0. ROBUST convergence: run past a min-step floor until the drag
drifts < 5e-5 over 200 steps (avoids false early plateaus), report K + the residual drift so convergence
is visible. Incremental-rotational pressure + warm-start. Success: order -> ~2, error below mode 0."""
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
    kfac = F * N**3 / (6 * np.pi * mu * R); hist = []; t0 = time.time(); drift = 1.0
    for it in range(max_steps):
        s.step()
        if it % 50 == 49:
            K = kfac / float(s.get_u().mean()); hist.append(K)
            if len(hist) >= 5:
                drift = abs(hist[-1] - hist[-5]) / abs(hist[-1])  # over 200 steps
                if it + 1 >= min_steps and drift < dtol: break
    return hist[-1], it + 1, drift, time.time() - t0

kref = 4.2920
print(f"Z&H K={kref}. Collocated EMBED (mode 5) vs mode 0. drift = |dK| over last 200 steps.", flush=True)
print(f"{'N':>4} | {'mode0 err%':>10} {'ord':>5} {'drift':>8} | {'embed err%':>11} {'ord':>5} {'drift':>8} | secs",
      flush=True)
p0 = p5 = pN = None
for N in (32, 48, 64, 96, 128):
    K0, st0, d0, s0 = drag(N, 0); K5, st5, d5, s5 = drag(N, 5)
    e0 = 100 * (K0 - kref) / kref; e5 = 100 * (K5 - kref) / kref
    o0 = np.log(abs(p0) / abs(e0)) / np.log(N / pN) if p0 else float("nan")
    o5 = np.log(abs(p5) / abs(e5)) / np.log(N / pN) if p5 else float("nan")
    print(f"{N:>4} | {e0:>+10.4f} {o0:>+5.2f} {d0:>8.1e} | {e5:>+11.4f} {o5:>+5.2f} {d5:>8.1e} | {s0+s5:>4.0f}",
          flush=True)
    p0, p5, pN = e0, e5, N
