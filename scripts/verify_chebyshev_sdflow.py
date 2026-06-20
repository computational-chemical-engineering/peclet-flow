#!/usr/bin/env python3
# Chebyshev pressure driver (sdflow set_pressure_chebyshev): a communication-light alternative to MG-PCG
# (Chebyshev semi-iteration preconditioned by one symmetric V-cycle, spectral bounds estimated once and
# reused). Validation: it must converge to the SAME cut-cell Stokes solution as the default MG-PCG (same
# operator, both Krylov/semi-iteration -> same fixed point). Scenario = the periodic sphere packing from
# verify_periodic_spheres_sdflow.
import sys, gc
import numpy as np
import sdflow

N, MU, DT, F = 32, 0.1, 60.0, 1e-3


def packing_sdf(N, radius_frac=0.18):
    R = N * radius_frac
    gx = np.arange(N)
    cs = [(c + 0.5) * N / 2.0 for c in (0, 1)]
    X, Y, Z = np.meshgrid(gx, gx, gx, indexing="ij")
    best = np.full((N, N, N), 1e30)
    for sx in cs:
        for sy in cs:
            for sz in cs:
                dx = X - sx; dx -= N * np.round(dx / N)
                dy = Y - sy; dy -= N * np.round(dy / N)
                dz = Z - sz; dz -= N * np.round(dz / N)
                best = np.minimum(best, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return best


def run(chebyshev, max_steps=200):
    sdf = np.asfortranarray(packing_sdf(N))
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(MU); s.set_dt(DT)
    s.set_body_force(F, 0.0, 0.0)
    s.set_advection(False)
    s.set_velocity_solver_params(80)
    s.set_pressure_multigrid(True, 4)
    if chebyshev:
        s.set_pressure_chebyshev(True, 200, 1e-9)
    s.set_solid(sdf, cutcell_pressure=True)
    prev = 0.0
    for it in range(max_steps):
        s.step()
        um = float(s.get_u().mean())
        if it > 8 and abs(um - prev) < 3e-4 * (abs(um) + 1e-15):
            break
        prev = um
    u = s.get_u()
    k = MU * float(u.mean()) / F
    div = s.max_open_divergence()
    iters = s.last_pressure_iterations()
    del s; gc.collect()
    return k, div, float(u.max()), int(iters)


def main():
    print("=== Chebyshev pressure driver: converges to the MG-PCG cut-cell Stokes solution ===")
    k_pcg, d_pcg, umax_pcg, it_pcg = run(chebyshev=False)
    k_cb, d_cb, umax_cb, it_cb = run(chebyshev=True)
    rel = abs(k_cb - k_pcg) / k_pcg
    print(f"  MG-PCG    : k={k_pcg:.6e}  div={d_pcg:.2e}  last_iters={it_pcg}")
    print(f"  Chebyshev : k={k_cb:.6e}  div={d_cb:.2e}  last_iters={it_cb}")
    print(f"  |k_cheb - k_pcg|/k_pcg = {rel*100:.4f}%")
    ok = (rel < 5e-3) and (d_cb < 1e-6 * umax_cb) and (it_cb > 0)
    print(f"  result: {'PASS' if ok else 'FAIL'}  (Chebyshev == MG-PCG solution; incompressible)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
