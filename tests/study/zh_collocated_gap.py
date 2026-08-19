#!/usr/bin/env python
"""Is the collocated permeability ceiling real? Measure the collocated-minus-staggered GAP on
Zick & Homsy, where both solvers run the same harness on identical geometry.

Measuring either solver against K_ZH = 4.292 is limited by the table's four figures (+-0.023 %).
The GAP between two computations on the same grid has no such limit: the benchmark value cancels,
so a ceiling of 0.3 % is resolvable far below its own size, and the question "does the gap decay
under refinement or flatten" gets a clean answer.

    SDFLOW_BUILD=build_ge python tests/study/zh_collocated_gap.py [N ...]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow  # noqa: E402

PHI0, K_ZH = 0.125, 4.292


def drag(N, kind, mu=0.1, F=1e-3, dt=80.0, warm_tol=1e-7, tail=40, max_steps=4000):
    R = (3 * PHI0 / (4 * np.pi)) ** (1 / 3) * N
    g = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    d = lambda A: A - 0.5 * N - N * np.round((A - 0.5 * N) / N)   # noqa: E731
    sdf = np.asfortranarray(np.sqrt(d(X) ** 2 + d(Y) ** 2 + d(Z) ** 2) - R)
    s = flow.Solver(N, N, N) if kind == "stag" else flow.SolverColocated(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    s.set_body_force(F, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(150)
    s.set_pressure_multigrid(True, max(2, int(np.log2(N)) - 1))
    s.set_pressure_pcg(True, 200, 1e-8)
    if kind != "stag":
        # set_collocated_scheme is new; fall back to the integer form so this runs against older
        # builds too (the Snellius module predates it).
        if hasattr(s, "set_collocated_scheme"):
            s.set_collocated_scheme(kind)
        else:
            s.set_face_interp({"gauge-exact": 9, "plain": 0}[kind])
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev, warm, um, t0 = 0.0, None, [], time.time()
    for it in range(max_steps):
        s.step()
        um.append(float(s.get_u().mean()))
        if warm is None:
            if it % 10 == 9:
                if it > 10 and abs(um[-1] - prev) < warm_tol * (abs(um[-1]) + 1e-30):
                    warm = it
                prev = um[-1]
        elif it - warm >= tail:
            break
    u = float(np.mean(um[-tail:]))
    return F * N ** 3 / (6 * np.pi * mu * R * u), it + 1, time.time() - t0


if __name__ == "__main__":
    Ns = [int(x) for x in (sys.argv[1:] or [32, 48, 64, 96, 128, 192, 256])]
    print(f"{'N':>5} {'h/R':>7} | {'K stag':>9} {'K gauge-ex':>11} | {'gap %':>8} {'order':>7} "
          f"| {'err_stag %':>10} {'steps s/g':>12} {'secs':>7}")
    prev = None
    for N in Ns:
        R = (3 * PHI0 / (4 * np.pi)) ** (1 / 3) * N
        try:
            ks, ns, ts = drag(N, "stag")
            kg, ng, tg = drag(N, "gauge-exact")
        except Exception as e:
            print(f"{N:>5}   FAILED: {type(e).__name__}: {str(e)[:60]}"); break
        gap = 100.0 * (kg - ks) / ks
        o = ""
        if prev and gap != 0:
            o = f"{np.log(abs(prev[1] / gap)) / np.log(N / prev[0]):+.2f}"
        print(f"{N:>5} {1/R:>7.4f} | {ks:>9.4f} {kg:>11.4f} | {gap:>+8.4f} {o:>7} | "
              f"{100*(ks-K_ZH)/K_ZH:>+10.3f} {f'{ns}/{ng}':>12} {ts+tg:>7.0f}", flush=True)
        prev = (N, gap)
    print("\ngap = 100*(K_collocated - K_staggered)/K_staggered on IDENTICAL geometry, so the")
    print("Zick-Homsy table's own precision cancels out. A decaying gap means no ceiling (the bed")
    print("ladders over-read two noisy rungs); a flat gap means the ceiling is real and lives in")
    print("the momentum/correction side, since the constraint operator was exonerated a-priori.")
