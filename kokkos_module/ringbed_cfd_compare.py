#!/usr/bin/env python3
# RingBed-CFD-Surrogate Stokes permeability: CUDA sdflow vs Kokkos sdflow_kokkos on the SAME ring-bed SDF.
# Isolates the CFD migration (the heavy solver): identical geometry + identical solver settings, so the
# permeability k must match (correctness) and the wall-time ratio is the Kokkos-vs-CUDA efficiency. The CUDA
# sdflow is Kokkos-free, so both modules co-import in one process. SDFs come from ringbed_gen_sdf.py (run
# separately, since the CUDA packing engine owns Kokkos init). Usage:
#   python ringbed_cfd_compare.py <sdf1.npy> [<sdf2.npy> ...]
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))        # CUDA sdflow
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build_module")))  # Kokkos
import sdflow as cuda
import sdflow_kokkos as kok

print("Kokkos execution space:", kok.execution_space)


def stokes_permeability(mod, sdf, axis=0, rho=1.0, mu=0.1, F=1e-3, dt=50.0, max_steps=400,
                        vel_iter=400, pres_iter=20, tol=1e-6):
    nx, ny, nz = sdf.shape
    sdf = np.ascontiguousarray(sdf, dtype=np.float64)
    s = mod.Solver(nx, ny, nz)
    s.set_rho(rho); s.set_mu(mu); s.set_dt(dt)
    s.set_body_force(*[F if i == axis else 0.0 for i in range(3)])
    s.set_advection(False)
    s.set_velocity_solver_params(vel_iter)
    s.set_pressure_solver_params(pres_iter)
    s.set_pressure_multigrid(True, levels=4)
    s.set_pressure_pcg(True, 200, 1e-8)
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True)
    getu = (s.get_u, s.get_v, s.get_w)[axis]
    prev, its = 0.0, 0
    t0 = time.perf_counter()
    for it in range(max_steps):
        s.step()
        umean = float(getu().mean()); its = it + 1
        if it > 8 and abs(umean - prev) < tol * (abs(umean) + 1e-15) and s.max_open_divergence() < 1e-5:
            break
        prev = umean
    wall = time.perf_counter() - t0
    u = getu()
    porosity = float((sdf > 0).mean())
    k = mu * float(u.mean()) / F
    return dict(porosity=porosity, k=k, U=float(u.mean()), steps=its, wall=wall,
                max_div=float(s.max_open_divergence()), umax=float(u.max()))


def main():
    sdfs = sys.argv[1:] or ["/tmp/ringbed/sdf_A_64.npy"]
    print(f"\n{'case':<22}{'res':>5}{'poros':>8}{'k_CUDA':>12}{'k_Kokkos':>12}{'rel_dk':>9}"
          f"{'t_CUDA':>9}{'t_Kok':>9}{'speedup':>9}")
    for f in sdfs:
        sdf = np.load(f)
        name = os.path.basename(f).replace(".npy", "")
        rc = stokes_permeability(cuda, sdf)
        rk = stokes_permeability(kok, sdf)
        reldk = abs(rk["k"] - rc["k"]) / (abs(rc["k"]) + 1e-30)
        speedup = rc["wall"] / rk["wall"] if rk["wall"] > 0 else float("nan")
        print(f"{name:<22}{sdf.shape[0]:>5}{rc['porosity']:>8.3f}{rc['k']:>12.5e}{rk['k']:>12.5e}"
              f"{reldk*100:>8.3f}%{rc['wall']:>9.2f}{rk['wall']:>9.2f}{speedup:>8.2f}x")
        print(f"    CUDA : steps={rc['steps']:<4} div={rc['max_div']:.1e} umax={rc['umax']:.4e}")
        print(f"    Kokk : steps={rk['steps']:<4} div={rk['max_div']:.1e} umax={rk['umax']:.4e}")


if __name__ == "__main__":
    main()
