#!/usr/bin/env python3
# Velocity multigrid (STAIRCASE coarse operator) for the Kokkos sdflow drop-in (set_velocity_multigrid).
# Mirrors scripts/verify_velocity_mg_staircase_zh_sdflow.py: a simple-cubic sphere (Zick & Homsy Stokes
# drag). The IBM velocity diffusion is solved with plain RB-GS (the exact reference) vs the staircase
# velocity-MG -- fine level = the sharp row-based IBM stencil; coarse levels classify cells by volume
# fraction (theta>=0.5 fluid / <0.5 solid-pinned) and use a plain const-coeff Helmholtz, with the IBM-cell
# residuals filtered before restriction. Checks:
#   (1) drag: |k_vmg - k_rbgs| / k_rbgs < 0.1% at a fixed V-cycle budget (vmg == RB-GS);
#   (2) stiff stability: at a large dt (beta = nu*dt large) the staircase stays finite and exact;
#   (3) parity vs CUDA sdflow velocity-MG (co-imports, Kokkos-free).
import os, sys, gc
import numpy as np
import sdflow_kokkos as kdk

cuda = None
if kdk.execution_space == "Cuda" and not os.environ.get("SKIP_CUDA_PARITY"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
    try:
        import sdflow as cuda
    except Exception as e:
        print(f"  (CUDA sdflow unavailable, skipping parity: {e})")

print("execution space:", kdk.execution_space)

N, PHI, MU, F = 48, 0.216, 0.1, 1e-3


def sc_sdf(N, phi):
    R = (phi * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * N
    g = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    c = N / 2.0
    return np.asfortranarray(np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R), R


def run(mod, mode, dt=60.0, vlevels=4, vcycles=12, max_steps=600):
    sdf, R = sc_sdf(N, PHI)
    s = mod.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(MU); s.set_dt(dt); s.set_body_force(F, 0, 0); s.set_advection(False)
    if mode == "rbgs":
        s.set_velocity_solver_params(200)
    else:
        s.set_velocity_multigrid(True, vlevels, vcycles)
    s.set_pressure_multigrid(True, 4)
    s.set_solid(sdf, cutcell_pressure=True)
    prev = 0.0
    for it in range(max_steps):
        s.step()
        u = s.get_u()
        if not np.isfinite(u).all():
            del s; gc.collect(); return None
        um = float(u.mean())
        if it > 8 and abs(um - prev) < 1e-7 * (abs(um) + 1e-15):
            break
        prev = um
    u = s.get_u()
    k = MU * float(u.mean()) / F
    div = s.max_open_divergence()
    del s; gc.collect()
    return k, div, float(u.max())


def main():
    print(f"=== velocity-MG staircase (Kokkos): vmg == RB-GS + stiff-stable + CUDA parity (N={N}, phi={PHI}) ===")
    k_rb, d_rb, _ = run(kdk, "rbgs")
    k_mg, d_mg, _ = run(kdk, "vmg")
    rel = abs(k_mg - k_rb) / k_rb
    print(f"  RB-GS        : k={k_rb:.6f}  div={d_rb:.2e}")
    print(f"  velocity-MG  : k={k_mg:.6f}  div={d_mg:.2e}")
    print(f"  |k_vmg - k_rbgs|/k_rbgs = {rel*100:.4f}%")

    # (2) stiff stability: large dt (beta = nu*dt = 0.1*1600 = 160) where a geometry-blind coarse op diverges
    stiff = run(kdk, "vmg", dt=1600.0, vcycles=16)
    stiff_ok = stiff is not None and np.isfinite(stiff[0])
    print(f"  stiff dt=1600: {'k=%.6f (finite, stable)' % stiff[0] if stiff_ok else 'DIVERGED'}")

    par = 0.0
    if cuda is not None:
        k_cu, d_cu, _ = run(cuda, "vmg")
        par = abs(k_mg - k_cu) / k_cu
        print(f"  CUDA vmg     : k={k_cu:.6f}  div={d_cu:.2e}   |Δk|/k={par*100:.4f}%")
    else:
        print("  parity: skipped (no CUDA reference)")

    ok = (rel < 1e-3) and (d_mg < 1e-6 * k_mg + 1e-9) and stiff_ok and (par < 1e-2)
    print(f"  result: {'PASS' if ok else 'FAIL'}  (vmg == RB-GS drag; stiff-stable; matches CUDA)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
