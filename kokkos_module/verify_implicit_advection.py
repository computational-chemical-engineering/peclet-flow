#!/usr/bin/env python3
# Implicit-FOU deferred-correction advection + Picard outer iterations for the Kokkos sdflow drop-in.
# Mirrors scripts/verify_implicit_advection_sdflow.py: flow around a sphere in a periodic box (full 3-D NS
# + cut-cell IBM + cut-cell pressure). Two checks, plus a parity vs the CUDA sdflow:
#   (1) high Re / large dt: EXPLICIT advection blows up; implicit-FOU stays finite and bounded;
#   (2) moderate Re (explicit stable): explicit and implicit-FOU agree -> same Koren scheme at convergence;
#   (3) parity: implicit-FOU Kokkos vs implicit-FOU CUDA agree (faithful port).
# The CUDA sdflow is Kokkos-free, so both modules co-import in one process.
import os, sys
import numpy as np
import sdflow_kokkos as kdk

# CUDA reference (Kokkos-free, co-imports). Optional: skipped on a non-CUDA backend build (e.g. OpenMP).
cuda = None
if kdk.execution_space == "Cuda" and not os.environ.get("SKIP_CUDA_PARITY"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
    try:
        import sdflow as cuda
    except Exception as e:
        print(f"  (CUDA sdflow unavailable, skipping parity: {e})")

print("execution space:", kdk.execution_space)

N = 32
NU = 0.1


def sphere_sdf(rfrac=0.3):
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    return np.sqrt((X - N / 2.0) ** 2 + (Y - N / 2.0) ** 2 + (Z - N / 2.0) ** 2) - N * rfrac


def run(mod, implicit, dt, fx, n_steps, to_steady=False):
    sdf = np.asfortranarray(sphere_sdf())
    s = mod.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(NU); s.set_dt(dt)
    s.set_body_force(fx, 0.0, 0.0)
    s.set_advection(True)
    s.set_implicit_advection(implicit)
    s.set_outer_iterations(3)
    s.set_velocity_solver_params(80)
    s.set_pressure_multigrid(True, 4)
    s.set_solid(sdf, cutcell_pressure=True)
    prev = 0.0
    for it in range(n_steps):
        s.step()
        u = s.get_u()
        if not np.isfinite(u).all():
            return None
        um = float(u.mean())
        if to_steady and it > 8 and abs(um - prev) < 1e-6 * (abs(um) + 1e-15):
            break
        prev = um
    return s.get_u()


def main():
    print("=== implicit-FOU advection (Kokkos): high-Re stability + moderate-Re correctness + CUDA parity ===")

    # (1) high Re / large dt: explicit blows up, implicit-FOU stays bounded
    print("  (1) high Re: dt=5, fx=0.02 (CFL >> 1)")
    ue = run(kdk, implicit=False, dt=5.0, fx=0.02, n_steps=30)
    ui = run(kdk, implicit=True, dt=5.0, fx=0.02, n_steps=30)
    expl_blew = ue is None
    impl_ok = ui is not None and np.isfinite(ui).all() and ui.max() < 1e3
    print(f"      explicit advection : {'BLEW UP' if expl_blew else f'finite U_max={ue.max():.3f}'}")
    print(f"      implicit-FOU       : {'finite U_max=%.3f' % ui.max() if impl_ok else 'unstable'}")

    # (2) moderate Re: explicit and implicit-FOU agree
    print("  (2) moderate Re: dt=5, fx=2e-4 (steady, both stable)")
    ue2 = run(kdk, implicit=False, dt=5.0, fx=2e-4, n_steps=300, to_steady=True)
    ui2 = run(kdk, implicit=True, dt=5.0, fx=2e-4, n_steps=300, to_steady=True)
    rel = abs(ui2.max() - ue2.max()) / ue2.max()
    print(f"      explicit U_max={ue2.max():.5f}  implicit-FOU U_max={ui2.max():.5f}  diff={rel*100:.2f}%")

    # (3) parity vs CUDA sdflow: same implicit-FOU moderate-Re steady solution
    par_max = par_field = 0.0
    if cuda is not None:
        print("  (3) parity: implicit-FOU Kokkos vs CUDA (dt=5, fx=2e-4)")
        uc = run(cuda, implicit=True, dt=5.0, fx=2e-4, n_steps=300, to_steady=True)
        par_max = abs(ui2.max() - uc.max()) / uc.max()
        par_field = float(np.abs(ui2 - uc).max()) / float(np.abs(uc).max())
        print(f"      CUDA U_max={uc.max():.5f}  Kokkos U_max={ui2.max():.5f}  "
              f"|ΔU_max|/U={par_max*100:.3f}%  max|Δu|/maxu={par_field*100:.3f}%")
    else:
        print("  (3) parity: skipped (no CUDA reference)")

    ok = expl_blew and impl_ok and rel < 0.03 and par_max < 0.02 and par_field < 0.03
    print(f"  result: {'PASS' if ok else 'FAIL'}  (implicit-FOU stable at high Re; agrees with explicit at "
          f"moderate Re; matches CUDA)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
