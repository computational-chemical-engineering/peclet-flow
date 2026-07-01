#!/usr/bin/env python3
"""Localize the ~1% sdflow-collocated vs AMR converged-k difference: run both on the SAME
registration-matched SC sphere and compare the converged velocity fields cell-by-cell. Is the
difference at the cut cells (projection/openness) or in the bulk (base operator)?

sdflow and AMR each init Kokkos and one finalizes it, so they cannot share a process — each engine
runs in its own subprocess (mode 'sdflow'/'amr') and writes its u-field to .npy; mode 'compare'
(default) spawns both and diffs. sdflow cell (i,j,k) centre=(i,j,k); AMR origin=-0.5 puts leaf
(i,j,k) centre at (i,j,k), so the grids coincide.
"""
import os, sys, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
phi = 0.125
mu, f, dt = 0.1, 1e-3, 60.0
R = (phi * 3 / (4 * np.pi)) ** (1 / 3) * N
c = N / 2.0
OUT = f"/tmp/cmp_{{}}_{N}.npy"


def run_sdflow():
    sys.path.insert(0, os.path.join(HERE, "..", "build_omp"))
    from peclet import flow as sdflow
    g = np.arange(N); X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    sdf = np.asfortranarray(np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R)
    s = sdflow.SolverColocated(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(f, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(300); s.set_pressure_multigrid(True, levels=max(2, int(np.log2(N)) - 1))
    s.set_pressure_pcg(True, 300, 1e-10); s.set_solid(sdf, cutcell_pressure=True)
    prev = 0.0
    for it in range(800):
        s.step()
        if it % 5 == 4:
            m = float(s.get_u().mean())
            if it > 10 and abs(m - prev) < 1e-7 * (abs(m) + 1e-30):
                break
            prev = m
    np.save(OUT.format("sdflow"), np.asarray(s.get_u()))


def run_amr():
    sys.path.insert(0, os.path.join(HERE, "..", "..", "transport-core", "python", "build"))
    import tpx_amr
    def sph(x, y, z):
        return ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) ** 0.5 - R
    oct = tpx_amr.Octree([N, N, N], 0, [-0.5, -0.5, -0.5], 1.0)
    fl = tpx_amr.Flow(oct, 1.0, mu, dt); fl.set_body_force(f, 0, 0); fl.set_advection(False); fl.set_solid(sph)
    # The device Stokes MG-PCG amplifies a near-nullspace mode over MANY steps; 120 steps is the
    # converged plateau (matches the registration study), well before the blow-up.
    for _ in range(120):
        fl.step(100, 60)
    ua = np.asarray(fl.velocity(0)); cen = np.rint(np.asarray(oct.centers())).astype(int)
    u = np.zeros((N, N, N)); u[cen[:, 0], cen[:, 1], cen[:, 2]] = ua
    np.save(OUT.format("amr"), u)


def compare():
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
    for mode in ("sdflow", "amr"):
        subprocess.run([sys.executable, __file__, mode, str(N)], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    u_sd = np.load(OUT.format("sdflow")); u_amr = np.load(OUT.format("amr"))
    g = np.arange(N); X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    dist = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R
    ksd = f * N ** 3 / (6 * np.pi * mu * R * u_sd.mean())
    kam = f * N ** 3 / (6 * np.pi * mu * R * u_amr.mean())
    print(f"N={N} phi={phi}  K_sdflow={ksd:.4f} ({100*(ksd-4.292)/4.292:+.2f}%)  "
          f"K_amr={kam:.4f} ({100*(kam-4.292)/4.292:+.2f}%)")
    d = u_sd - u_amr
    print(f"  Umean: sdflow {u_sd.mean():.5e}  amr {u_amr.mean():.5e}  Δmean {d.mean():+.3e}  "
          f"max|du| {np.abs(d).max():.3e} (u scale {u_sd.max():.3e})")
    solid = dist <= 0; cut = (dist > 0) & (dist < 1.5); bulk = dist >= 1.5
    tot = d.mean()
    for name, msk in [("solid", solid), ("cut-band(<1.5h)", cut), ("bulk(>1.5h)", bulk)]:
        contrib = d[msk].sum() / d.size
        nz = np.abs(d[msk])
        print(f"  {name:16s}: ncells={int(msk.sum()):6d}  mean|du|={nz.mean():.3e}  max|du|={nz.max():.3e}  "
              f"Δmean-contrib={contrib:+.3e} ({100*contrib/tot if tot else 0:+6.1f}% of Δmean)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    {"sdflow": run_sdflow, "amr": run_amr, "compare": compare}[mode]()
