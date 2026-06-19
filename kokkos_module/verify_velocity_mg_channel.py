#!/usr/bin/env python3
# Domain-BC const-coeff velocity multigrid (Kokkos), the cavity/BFS/channel regime of set_velocity_multigrid.
# Developing plane channel (uniform inflow -> parabolic outflow; -x inflow, +x outflow, +-y no-slip walls).
# The velocity diffusion is solved with plain RB-GS (the reference) vs the const-coeff velocity-MG: every
# level is the anisotropic const-coeff Helmholtz I - nu*dt*Lap with the no-slip/inflow/outflow boundary fold
# baked into the diagonal (CUDA setDiffusionConstAllLevels + setDiffusionBoundaryFold), PLUS a held-Dirichlet
# residual exclude: the boundary-face cells are pinned by the BC re-imposition, so their residual is zeroed
# before restriction (else the coarse correction drifts the boundary ~2% -- the discrepancy CUDA's domain-BC
# vmg leaves; this port adds the exclude, analogous to the IBM clean-fluid exclude, so vel-MG == RB-GS). Checks:
#   (1) vel-MG converges to the SAME field as RB-GS (to ~float level; the MG operator is float-stored, RB-GS
#       is double) and recovers the developed parabola 1.5;
#   (2) parity vs CUDA sdflow RB-GS channel (the machine-precision reference field).
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
L, H, nz, U = 48, 24, 4, 1.0
NU = U * H / 100.0     # Re = 100
DT = 100.0


def run(mod, vmg, nsteps=150):
    s = mod.Solver(L, H, nz)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(DT); s.set_advection(False)
    s.set_domain_bc(0, 2, U, 0.0, 0.0)             # -x uniform inflow
    s.set_domain_bc(1, 3)                           # +x outflow
    s.set_domain_bc(2, 1); s.set_domain_bc(3, 1)    # +-y no-slip walls
    if vmg:
        s.set_velocity_multigrid(True, 3, 12)
    else:
        s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, 3); s.set_pressure_solver_params(60)
    s.set_pressure_geometry(np.full((L, H, nz), 1e30))
    for _ in range(nsteps):
        s.step()
    u = s.get_u()
    del s; gc.collect()
    return u


def main():
    print("=== domain-BC velocity-MG (Kokkos): const-coeff channel == RB-GS + parabola + CUDA parity ===")
    ur = run(kdk, vmg=False)
    uv = run(kdk, vmg=True)
    du = np.abs(uv - ur).max() / (np.abs(ur).max() + 1e-30)
    prof = uv[L - 2, :, nz // 2]
    ratio = float(prof.max() / (prof.mean() + 1e-30))
    print(f"  vel-MG vs RB-GS (Kokkos): max|du|/maxu={du:.2e}   outlet u_max/U_mean={ratio:.4f}")

    par = 0.0
    if cuda is not None:
        uc = run(cuda, vmg=False)  # CUDA RB-GS channel = the machine-precision reference field
        par = np.abs(uv - uc).max() / (np.abs(uc).max() + 1e-30)
        print(f"  vel-MG vs CUDA RB-GS    : max|du|/maxu={par:.2e}")
    else:
        print("  parity: skipped (no CUDA reference)")

    ok = (du < 1e-4) and (abs(ratio - 1.5) < 0.05) and (par < 1e-4)
    print(f"  result: {'PASS' if ok else 'FAIL'}  (const-coeff vel-MG == RB-GS to float level; parabola; matches CUDA)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
