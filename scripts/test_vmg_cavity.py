"""Velocity-MG vs RB-GS on the lid-driven cavity (Ghia Re=100). Checks (1) no regression with vmg OFF,
(2) vmg ON reaches the same Ghia centreline match + incompressibility. Run: PYTHONPATH=build_mpi python ..."""
import sys, pathlib, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sdflow
from verify_lid_cavity_sdflow import GHIA_Y, GHIA_U, GHIA_X, GHIA_V


def run(N=128, Re=100.0, U=1.0, nz=4, max_steps=4000, vmg=False, vlevels=3, vcycles=4, vel_iter=60, dt=1.0):
    nu = U * N / Re
    s = sdflow.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(nu); s.set_dt(dt); s.set_advection(True)
    s.set_domain_bc(0, 1); s.set_domain_bc(1, 1); s.set_domain_bc(2, 1)
    s.set_domain_bc(3, 2, U, 0.0, 0.0)
    s.set_velocity_solver_params(vel_iter)
    if vmg:
        s.set_velocity_multigrid(True, vlevels, vcycles)
    s.set_pressure_multigrid(True, levels=8)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((N, N, nz), 1e30))
    t0 = time.time()
    prev = None
    steps = max_steps
    for it in range(max_steps):
        s.step()
        if it % 100 == 0:
            u = np.asarray(s.get_u()).reshape((N, N, nz), order="F")
            cur = u[:, :, nz // 2].copy()
            if prev is not None:
                d = np.max(np.abs(cur - prev)) / (np.max(np.abs(cur)) + 1e-30)
                if d < 2e-5:
                    steps = it + 1; break
            prev = cur
    dt_wall = time.time() - t0
    u = np.asarray(s.get_u()).reshape((N, N, nz), order="F")
    v = np.asarray(s.get_v()).reshape((N, N, nz), order="F")
    uc = 0.5 * (u[N // 2 - 1, :, nz // 2] + u[N // 2, :, nz // 2])
    vc = 0.5 * (v[:, N // 2 - 1, nz // 2] + v[:, N // 2, nz // 2])
    yc = (np.arange(N) + 0.5) / N
    u_rms = float(np.sqrt(np.mean((np.interp(GHIA_Y, yc, uc) - GHIA_U) ** 2)))
    v_rms = float(np.sqrt(np.mean((np.interp(GHIA_X, yc, vc) - GHIA_V) ** 2)))
    try:
        div = float(s.max_flux_divergence())
    except Exception:
        div = float("nan")
    return dict(u_rms=u_rms, v_rms=v_rms, umin=float(uc.min()), div=div, steps=steps, wall=dt_wall)


if __name__ == "__main__":
    import os
    N = int(os.environ.get("VMG_N", 128))
    print(f"=== Velocity-MG (semi-coarsening) vs RB-GS: lid cavity Re=100, N={N} (quasi-2D nz=4) ===")
    # under-solved RB-GS (few sweeps) vs few-cycle deep vmg: the MG should hold accuracy where RB-GS lags.
    cfgs = [("RB-GS  60 sweeps   ", dict(vmg=False, vel_iter=60)),
            ("RB-GS  20 sweeps   ", dict(vmg=False, vel_iter=20)),
            ("vmg L=8 v=2 (semi) ", dict(vmg=True, vlevels=8, vcycles=2)),
            ("vmg L=8 v=4 (semi) ", dict(vmg=True, vlevels=8, vcycles=4))]
    for label, kw in cfgs:
        r = run(N=N, **kw)
        print(f"  {label}: u_rms={r['u_rms']:.4f} v_rms={r['v_rms']:.4f} umin={r['umin']:.4f} "
              f"(Ghia -0.2058) div={r['div']:.1e} steps={r['steps']} wall={r['wall']:.1f}s")
