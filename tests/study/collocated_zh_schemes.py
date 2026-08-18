#!/usr/bin/env python
"""Zick & Homsy grid convergence of the COLLOCATED schemes, in the protocol the
staggered-vs-collocated example page quotes (warm_tol 1e-7, tail 40, up to 4000 steps).

Regenerates the page's collocated columns now that "gauge-exact" (the former set_face_interp(9))
is the default. Reproducing the published "plain" column is the control that the harness matches
the earlier runs.

    SDFLOW_BUILD=build_ge python tests/study/collocated_zh_schemes.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow  # noqa: E402

PHI0, K_ZH = 0.125, 4.292


def sphere_sdf(N, phi):
    R = (3 * phi / (4 * np.pi)) ** (1 / 3) * N
    g = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    d = lambda A, c: A - c * N - N * np.round((A - c * N) / N)   # noqa: E731
    return np.sqrt(d(X, .5) ** 2 + d(Y, .5) ** 2 + d(Z, .5) ** 2) - R, R


def drag(N, scheme, mu=0.1, F=1e-3, dt=80.0, warm_tol=1e-7, tail=40, max_steps=4000):
    sdf, R = sphere_sdf(N, PHI0)
    s = flow.SolverColocated(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    s.set_body_force(F, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(150)
    s.set_pressure_multigrid(True, max(2, int(np.log2(N)) - 1))
    s.set_pressure_pcg(True, 200, 1e-8)
    s.set_collocated_scheme(scheme)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev, warm, um, pit, t0 = 0.0, None, [], [], time.time()
    for it in range(max_steps):
        s.step()
        um.append(float(s.get_u().mean())); pit.append(int(s.last_pressure_iterations()))
        if warm is None:
            if it % 10 == 9:
                if it > 10 and abs(um[-1] - prev) < warm_tol * (abs(um[-1]) + 1e-30):
                    warm = it
                prev = um[-1]
        elif it - warm >= tail:
            break
    umean = float(np.mean(um[-tail:]))
    K = F * N ** 3 / (6 * np.pi * mu * R * umean)
    return K, 100 * (K - K_ZH) / K_ZH, float(np.mean(pit[-tail:])), it + 1, time.time() - t0


if __name__ == "__main__":
    Ns = [int(x) for x in (sys.argv[1:] or [32, 48, 64, 96, 128])]
    print(f"{'N':>5} {'scheme':>12} {'K':>9} {'err %':>9} {'p.iters':>8} {'steps':>7} {'secs':>7}")
    out = {}
    for scheme in ("plain", "gauge-exact"):
        for N in Ns:
            K, e, pit, st, secs = drag(N, scheme)
            out.setdefault(scheme, []).append((N, e))
            print(f"{N:>5} {scheme:>12} {K:>9.4f} {e:>+9.3f} {pit:>8.1f} {st:>7} {secs:>7.1f}",
                  flush=True)
    print("\nobserved order (|err| ratio between successive N):")
    for scheme, rows in out.items():
        ords = [f"{np.log(abs(rows[i-1][1]/rows[i][1]))/np.log(rows[i][0]/rows[i-1][0]):+.2f}"
                for i in range(1, len(rows))]
        print(f"  {scheme:>12}: " + "  ".join(ords))
    print("\nPUBLISHED col cutcell (plain) for the control: "
          "+1.004 +0.685 +0.598 +0.397 +0.299 at N=32,48,64,96,128")
