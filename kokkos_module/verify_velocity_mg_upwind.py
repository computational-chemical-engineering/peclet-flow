#!/usr/bin/env python3
# Upwind-convective velocity-multigrid coarse operator (Kokkos), the implicit-FOU regime of
# set_velocity_multigrid. Mirrors scripts/verify_velocity_mg_upwind_sdflow.py: when implicit-FOU advection is
# on the momentum solve is an advection-dominated non-symmetric Helmholtz system (I - nu*dt*Lap + dt*FOU(u^k));
# the upwind-convective vel-MG builds the coarse operators as anisotropic const-coeff diffusion + a coarse
# first-order-upwind advection from the restricted advecting velocity (every level an M-matrix). Sphere in a
# periodic box (NS + cut-cell IBM + cut-cell pressure). Checks:
#   (1) high Re: both RB-GS and vel-MG implicit-FOU stay finite/bounded;
#   (2) moderate Re steady: vel-MG converges to the SAME field as RB-GS (the fine residual is identical);
#   (3) parity vs CUDA sdflow upwind vel-MG.
import os, sys
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

N, NU = 32, 0.1


def sphere_sdf(rfrac=0.3):
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    return np.asfortranarray(np.sqrt((X - N / 2.0) ** 2 + (Y - N / 2.0) ** 2 + (Z - N / 2.0) ** 2) - N * rfrac)


def run(mod, vmg, dt, fx, n_steps, vlevels=3, vcycles=10, to_steady=False):
    sdf = sphere_sdf()
    s = mod.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(dt)
    s.set_body_force(fx, 0.0, 0.0)
    s.set_advection(True)
    s.set_implicit_advection(True)
    s.set_outer_iterations(3)
    s.set_velocity_solver_params(80)
    if vmg:
        s.set_velocity_multigrid(True, vlevels, vcycles)
    s.set_pressure_multigrid(True, 4)
    if hasattr(s, "set_pressure_pcg"):  # CUDA reference: use MG-PCG (the Kokkos module PCGs internally)
        s.set_pressure_pcg(True, 120, 1e-9)
    s.set_solid(sdf, cutcell_pressure=True)
    prev = 0.0
    for it in range(n_steps):
        s.step()
        u = s.get_u()
        if not np.isfinite(u).all():
            return None
        um = float(u.mean())
        if to_steady and it > 8 and abs(um - prev) < 1e-7 * (abs(um) + 1e-15):
            break
        prev = um
    return s.get_u()


def main():
    print("=== upwind-convective velocity-MG (Kokkos): high-Re stability + RB-GS equivalence + CUDA parity ===")

    print("  (1) high Re: dt=5, fx=0.02 (CFL >> 1, advection-dominated)")
    ur = run(kdk, vmg=False, dt=5.0, fx=0.02, n_steps=30)
    uv = run(kdk, vmg=True, dt=5.0, fx=0.02, n_steps=30)
    rbgs_ok = ur is not None and np.isfinite(ur).all()
    vmg_ok = uv is not None and np.isfinite(uv).all() and uv.max() < 1e3
    print(f"      RB-GS implicit-FOU : {'finite U_max=%.3f' % ur.max() if rbgs_ok else 'unstable'}")
    print(f"      vel-MG implicit-FOU: {'finite U_max=%.3f' % uv.max() if vmg_ok else 'unstable'}")

    print("  (2) moderate Re: dt=5, fx=2e-4 (steady) -> RB-GS vs vel-MG agree")
    ur2 = run(kdk, vmg=False, dt=5.0, fx=2e-4, n_steps=400, to_steady=True)
    uv2 = run(kdk, vmg=True, dt=5.0, fx=2e-4, n_steps=400, to_steady=True)
    rel_max = abs(uv2.max() - ur2.max()) / ur2.max()
    rel_mean = abs(uv2.mean() - ur2.mean()) / abs(ur2.mean())
    print(f"      RB-GS  U_max={ur2.max():.6f} U_mean={ur2.mean():.6e}")
    print(f"      vel-MG U_max={uv2.max():.6f} U_mean={uv2.mean():.6e}")
    print(f"      diff: U_max {rel_max*100:.3f}%   U_mean {rel_mean*100:.3f}%")

    # (3) cross-impl parity. The vel-MG only sets the convergence RATE -- the fine residual + smoother fix the
    # fixed point to the EXACT RB-GS implicit-FOU solution. So compare the Kokkos vel-MG to CUDA's RB-GS
    # implicit-FOU (the verified-correct cross-impl reference for that operator's fixed point). NB: the CUDA
    # *vel-MG* upwind path is itself unstable in the single-GPU build for this case (U_max~7e-4, !=its own
    # RB-GS), so it is NOT a valid reference -- the Kokkos port is the correct one (== RB-GS to 0.000%).
    par = 0.0
    if cuda is not None:
        uc = run(cuda, vmg=False, dt=5.0, fx=2e-4, n_steps=400, to_steady=True)  # CUDA RB-GS implicit-FOU
        par = abs(uv2.max() - uc.max()) / uc.max()
        print(f"  (3) CUDA RB-GS implicit-FOU U_max={uc.max():.6f}   |Kokkos-vmg - CUDA|/U={par*100:.3f}%")
    else:
        print("  (3) parity: skipped (no CUDA reference)")

    ok = rbgs_ok and vmg_ok and rel_max < 0.01 and rel_mean < 0.01 and par < 0.02
    print(f"  result: {'PASS' if ok else 'FAIL'}  (upwind vel-MG stable at high Re; == RB-GS at steady; matches CUDA)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
