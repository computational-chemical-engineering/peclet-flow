#!/usr/bin/env python
"""§11 G-perf, the Python half (WO-9; an instrument — it prints, it never fails on a number).

  advance  the scalar advance against the pressure projection of the same step, on a 128^3 bed
           (G5b's simple-cubic sphere, solid fraction 0.3, one period; Stokes under G9b's body
           force) at dt D / h^2 = 1: one cut-cell scalar (koren, Dirichlet spheres c = 0, c0 = 1)
           carried by step(); per step the step's own 'projection' timer and a standalone
           advance_scalars() at the same face field (both device-fenced: the budget read after it
           reduces on the device).
  gadvb    the steady G-adv(b) closure solve at 64^3 (Neumann spheres, mean-gradient mode), Pe_h 1
           and 10: wall time of the gated solve (the one after the Pe rescale).

The other G-perf rows are tests/kokkos/bench_scalar_cutcell (triad, memory, matvec) and the
`scalar_mg` ctest (the advective V-cycle against the symmetric one).

Run: OMP_NUM_THREADS=4 PYTHONPATH=<build> python tests/study/scalar_ibm/gperf.py [advance] [gadvb]
"""
import math
import os
import sys
import time

import numpy as np
import peclet.flow as pf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))
import test_scalar_cutcell_gates as g


def advance(n=128, warm=5, steps=5):
    s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    dt = 0.01
    s.set_dt(dt)
    R = (0.3 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    X, Y, Z = g.grid(s)
    s.set_solid(np.asfortranarray(np.sqrt((X - 0.513) ** 2 + (Y - 0.479) ** 2 + (Z - 0.507) ** 2) - R),
                cutcell_pressure=True)
    s.set_body_force((30.0, 9.0, 0.0))
    for _ in range(warm):
        s.step()
    h = 1.0 / n
    s.add_scalar("c", diffusivity=h * h / dt, scheme="koren", cutcell=True)  # dt D / h^2 = 1
    s.set_scalar_wall("c", "dirichlet", 0.0)
    geo = s.diagnostics.scalar_geometry("c")
    s.set_field("c", np.asfortranarray(np.where(geo["unknown"] > 0.5, 1.0, 0.0)))
    rows = []
    for _ in range(steps):
        s.step()
        tm = s.diagnostics.last_step_timers()
        t0 = time.perf_counter()
        s.advance_scalars()
        s.diagnostics.scalar_budget("c")
        ta = time.perf_counter() - t0
        it = s.diagnostics.scalar_census("c")["krylov_iterations"]
        rows.append((tm["step"], tm["projection"], ta, it))
    print(f"advance {n}^3 bed, dt D/h^2 = 1 (koren, Dirichlet spheres), per step:")
    for st, pr, ta, it in rows:
        print(f"  step {st * 1e3:8.1f} ms  projection {pr * 1e3:8.1f} ms  advance_scalars "
              f"{ta * 1e3:8.1f} ms ({it} it)  advance/projection {ta / pr:5.2f}")
    med = np.median([r[2] / r[1] for r in rows])
    print(f"  median advance/projection {med:.2f} (red flag > 1)")


def gadvb(n=64):
    made, times = [], []

    class Timed(pf.Solver):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

        def solve_scalar_steady(self, *a, **k):
            t0 = time.perf_counter()
            r = super().solve_scalar_steady(*a, **k)
            self.diagnostics.scalar_budget("c")  # a device reduction: fences
            times.append(time.perf_counter() - t0)
            return r

    g.pf.Solver = Timed
    for pe in (1.0, 10.0):
        times.clear()
        g.gadv_b_case(n, pe, "neumann")
        it = made[-1].diagnostics.scalar_census("c")["krylov_iterations"]
        print(f"gadvb {n}^3 Pe_h {pe:g}: gated solve {times[-1]:.3f} s ({it} it); the probe solve "
              f"before the rescale (assembly + MG build) {times[0]:.3f} s")


if __name__ == "__main__":
    which = sys.argv[1:] or ["advance", "gadvb"]
    if "advance" in which:
        advance()
    if "gadvb" in which:
        gadvb()
