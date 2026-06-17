#!/usr/bin/env python3
"""Z&H gate for the STAIRCASE velocity-MG coarse operator (IBM momentum solve).

Simple-cubic sphere array (the Zick & Homsy Stokes-drag ground truth). The IBM velocity diffusion is solved
with plain RB-GS (the exact reference) vs the staircase velocity multigrid (set_velocity_multigrid): the fine
level is the sharp row-based IBM stencil, the coarse levels use the volume fraction only to CLASSIFY cells
(theta>=0.5 fluid / <0.5 solid-pinned) and a plain const-coeff Helmholtz at fluid cells, with the IBM-cell
residuals filtered before restriction. See doc/velocity_mg_plan.md.

Two checks:
  (1) drag: |K_vmg - K_rbgs| / K_rbgs < 0.1% at a fixed V-cycle budget;
  (2) stiff stability: at a large dt (beta=nu*dt=80) where the old geometry-blind const coarse op diverged,
      the staircase stays finite and exact.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                 os.environ.get("SDFLOW_BUILD", "build_mpi"))))
import sdflow  # noqa: E402

ZH_PHI = [0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236]
ZH_K = [1.096, 1.212, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1]


def sphere_radius(phi, N):
    return (phi * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * N


def sc_sdf(N, phi):
    R = sphere_radius(phi, N)
    g = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    c = N / 2.0
    return np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R, R


def drag_K(umean, R, N, f, mu):
    return f * N ** 3 / (6.0 * np.pi * mu * R * umean)


def run(N, phi, mode, vel_iter=200, vlevels=4, vcycles=12, mu=0.1, f=1e-3, dt=60.0,
        max_steps=600, tol=1e-7):
    """mode: 'rbgs' (plain RB-GS reference) | 'vmg' (the staircase velocity-MG, via set_velocity_multigrid)."""
    sdf, R = sc_sdf(N, phi)
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(f, 0, 0); s.set_advection(False)
    if mode == "rbgs":
        s.set_velocity_solver_params(vel_iter)
    else:
        s.set_velocity_multigrid(True, vlevels, vcycles)      # IBM -> staircase coarse op (the default)
    lv = max(2, int(np.log2(N)) - 1)
    s.set_pressure_multigrid(True, levels=lv)
    s.set_pressure_pcg(True, max_iter=200, rtol=1e-8)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev = 0.0; t0 = time.time()
    for it in range(max_steps):
        s.step()
        u = s.get_u()
        if not np.isfinite(u).all():
            return None, None
        if it % 5 == 4:
            m = float(u.mean())
            if it > 10 and abs(m - prev) < tol * (abs(m) + 1e-30):
                break
            prev = m
    return drag_K(float(s.get_u().mean()), R, N, f, mu), time.time() - t0


def main():
    N = int(os.environ.get("VMG_N", 64))
    phi = float(os.environ.get("VMG_PHI", 0.216))
    vcyc = int(os.environ.get("VMG_VCYC", 12))
    kref = float(np.interp(phi, ZH_PHI, ZH_K))
    print(f"=== staircase velocity-MG Z&H gate: SC sphere N={N} phi={phi} (Z&H K~{kref:.3f}) ===")
    Krb, wrb = run(N, phi, "rbgs")
    Kvm, wvm = run(N, phi, "vmg", vcycles=vcyc)
    e = abs(Kvm - Krb) / Krb if (Kvm and Krb) else float("nan")
    print(f"  RB-GS         : K={Krb:.5f}  wall={wrb:.1f}s")
    print(f"  vel-MG (v={vcyc:2d}) : K={'NaN' if Kvm is None else f'{Kvm:.5f}'}  wall={wvm:.1f}s   "
          f"err vs RB-GS {e*100:.3f}%")
    ok1 = (Kvm is not None) and e < 0.001

    # stiff regime: the geometry-blind const coarse op diverged here (dt=800, beta=80); staircase stays exact.
    dt_stiff = float(os.environ.get("VMG_DT_STIFF", 800.0))
    print(f"  stiff dt={dt_stiff:.0f} (beta={mu_b(dt_stiff):.0f}; geometry-blind const coarse diverges):")
    Krb_s, _ = run(N, phi, "rbgs", dt=dt_stiff)
    Kv_s, _ = run(N, phi, "vmg", dt=dt_stiff, vcycles=vcyc)
    e_s = abs(Kv_s - Krb_s) / Krb_s if (Kv_s and Krb_s) else float("nan")
    print(f"      RB-GS K={Krb_s:.5f}   vel-MG={'BLEW UP' if Kv_s is None else f'{Kv_s:.5f} (err {e_s*100:.3f}%)'}")
    ok2 = (Kv_s is not None) and e_s < 0.005

    print(f"  result: {'PASS' if ok1 and ok2 else 'FAIL'}  (staircase vel-MG == RB-GS drag, and stable+exact "
          f"at a stiff dt)")
    sys.exit(0 if ok1 and ok2 else 1)


def mu_b(dt, mu=0.1):
    return mu * dt


if __name__ == "__main__":
    main()
