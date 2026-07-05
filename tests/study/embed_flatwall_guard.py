"""Guard: a FLAT (grid-aligned normal) channel wall at a FRACTIONAL y position -> genuine axis-aligned
cut cells (fractional openness), driven Poiseuille, periodic box + cut-cell pressure. The embed wall
gradient (mode 5) must reproduce the analytic parabola as well as mode 0 (for an axis-aligned normal
the true-normal reconstruction is a pure 1-D quadratic along y = exact for the quadratic profile).
Confirms mode 5 does not degrade the flat-wall case that modes 0-3 nail."""
import os, sys
sys.path.insert(0, os.path.abspath("build_cuda2"))
import numpy as np
from peclet import flow

def run(N, mode, mu=0.1, F=0.01, dt=50.0, max_steps=1500):
    nx, nz = 8, 8; ny = N
    ylo = 0.3 * ny + 0.137          # fractional -> real cut cells (axis-aligned normal)
    yhi = 0.7 * ny - 0.137
    H = yhi - ylo
    gy = np.arange(ny, dtype=np.float64) + 0.5    # cell centres
    sdf = np.minimum(gy - ylo, yhi - gy)[None, :, None] * np.ones((nx, ny, nz))
    lv = max(2, int(np.log2(N)) - 1)
    s = flow.SolverColocated(nx, ny, nz)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(F, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(200); s.set_pressure_multigrid(True, levels=lv)
    s.set_pressure_pcg(True, 400, 1e-11); s.set_face_interp(mode)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev = 0.0
    for it in range(max_steps):
        s.step(); u = float(s.get_u().max())
        if it > 10 and abs(u - prev) < 1e-9 * (abs(u) + 1e-30): break
        prev = u
    U_sim = float(s.get_u().max()); U_ana = F * H * H / (8.0 * mu)
    return H, U_sim, U_ana, 100 * abs(U_sim - U_ana) / U_ana

print("Flat-wall (fractional, axis-aligned cut cells) Poiseuille: mode 0 vs mode 6 (embed)")
print(f"{'N':>4} | {'mode0 err%':>11} | {'mode6 err%':>11}")
for N in (32, 48, 64):
    _, _, _, e0 = run(N, 0)
    _, _, _, e6 = run(N, 6)
    print(f"{N:>4} | {e0:>+11.5f} | {e6:>+11.5f}", flush=True)
