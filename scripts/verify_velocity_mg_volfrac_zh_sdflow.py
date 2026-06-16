#!/usr/bin/env python3
"""Z&H gate for the volume-fraction velocity-MG coarse operator (IBM momentum solve).

Simple-cubic sphere array (the Zick & Homsy Stokes-drag ground truth). The IBM velocity diffusion is solved
three ways at a FIXED, modest velocity-iteration budget so an under-converging solver shows up as a biased
drag K:
  - RB-GS                : the exact reference (many sweeps).
  - vel-MG, const coarse : geometry-blind coarse op -> diverges (NaN) once the diffusion is stiff (large dt).
  - vel-MG, volfrac coarse: the geometry-aware coarse op (1 + sum beta_f, beta_f = nu_dt*min(theta) /h^2) with
    the CLEAN-FLUID-INTERIOR coarse coupling -- the residual is zeroed and no correction is prolonged at the
    IBM cut cells AND solid cells, so the coarse grid couples only where its clean operator matches the fine
    one. Converges to the EXACT RB-GS drag (fine residual + As_[c] smoother own the boundary band).

Two checks:
  (1) drag: |K_volfrac - K_rbgs|/K_rbgs < 0.1% at a fixed budget (and <= the const-coarse error);
  (2) stiff stability: at a large dt where the const coarse op DIVERGES (NaN), volfrac stays exact.
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


def run(N, phi, mode, vel_iter=200, vlevels=4, vcycles=8, mu=0.1, f=1e-3, dt=60.0,
        max_steps=600, tol=1e-7):
    sdf, R = sc_sdf(N, phi)
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt); s.set_body_force(f, 0, 0); s.set_advection(False)
    if mode == "rbgs":
        s.set_velocity_solver_params(vel_iter)
    else:
        s.set_velocity_multigrid(True, vlevels, vcycles)
        if mode == "volfrac":
            s.set_velocity_mg_volfrac(True, eps=0.1)
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
    phi = float(os.environ.get("VMG_PHI", 0.216))     # dense -> many cut cells -> D_rescale departs from 1
    vcyc = int(os.environ.get("VMG_VCYC", 8))
    kref = float(np.interp(phi, ZH_PHI, ZH_K))
    print(f"=== volfrac velocity-MG Z&H gate: SC sphere N={N} phi={phi} (Z&H K~{kref:.3f}) ===")
    rows = []
    for mode, label in [("rbgs", "RB-GS (200 sweeps)   "),
                        ("const", f"vel-MG const  (v={vcyc})"),
                        ("volfrac", f"vel-MG volfrac(v={vcyc})")]:
        K, wall = run(N, phi, mode, vcycles=vcyc)
        rows.append((label, K, wall))
        print(f"  {label}: K={'NaN' if K is None else f'{K:.5f}'}  wall={wall:.1f}s" if K else
              f"  {label}: BLEW UP")

    Krb = rows[0][1]
    Kconst = rows[1][1]
    Kvol = rows[2][1]
    e_const = abs(Kconst - Krb) / Krb if (Kconst and Krb) else float("nan")
    e_vol = abs(Kvol - Krb) / Krb if (Kvol and Krb) else float("nan")
    print(f"  drag error vs RB-GS:  const coarse {e_const*100:.3f}%   volfrac coarse {e_vol*100:.3f}%")
    ok1 = (Kvol is not None) and e_vol < 0.001 and e_vol <= e_const + 1e-9

    # (2) stiff regime: a large dt where the const coarse op diverges; volfrac must stay exact.
    dt_stiff = float(os.environ.get("VMG_DT_STIFF", 200.0))
    print(f"  stiff dt={dt_stiff:.0f} (const coarse diverges):")
    Krb_s, _ = run(N, phi, "rbgs", dt=dt_stiff)
    Kc_s, _ = run(N, phi, "const", dt=dt_stiff, vcycles=vcyc)
    Kv_s, _ = run(N, phi, "volfrac", dt=dt_stiff, vcycles=vcyc)
    print(f"      RB-GS K={Krb_s:.5f}   const={'NaN' if Kc_s is None else f'{Kc_s:.5f}'}   "
          f"volfrac={'NaN' if Kv_s is None else f'{Kv_s:.5f}'}")
    e_vol_s = abs(Kv_s - Krb_s) / Krb_s if (Kv_s and Krb_s) else float("nan")
    ok2 = (Kc_s is None) and (Kv_s is not None) and e_vol_s < 0.005
    print(f"      const {'DIVERGED' if Kc_s is None else 'finite'}; volfrac error {e_vol_s*100:.3f}%")

    ok = ok1 and ok2
    print(f"  result: {'PASS' if ok else 'FAIL'}  (volfrac == RB-GS drag at a fixed budget AND stays exact "
          f"at a stiff dt where const diverges)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
