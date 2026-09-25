#!/usr/bin/env python
"""Cost of the balanced-force projection (doc/collocated_varrho_forces.md §4.8, gate G7) -- an
INSTRUMENT, not a gate. Two cases: MOVING (below) and STATIC (the same drop at rest, 50 steps:
with the increment solve and its full-RHS stop, WO-P5, a static interface must cost 0
pre-projection iterations after the first step).

A moving ratio-1000 drop on 32^3 (periodic, R = 8, sigma = 1, mu = 0.1, a uniform translation
U = 0.02 seeded in the velocity, dt = 0.5 capillary_dt), on both grids, stepped with the
balanced-force projection OFF and ON. Reports the median wall time per step over steps
[WARM, STEPS) and the mean pressure / balanced-force iteration counts. G7: ON <= 1.5x OFF, and
mean balanced-force iterations <= mean main pressure iterations.

The staggered grid runs with enable_vof_momentum (ratio 1000 with motion); the collocated grid is
V8 (the option is its default; OFF is set explicitly).

Run: OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=<build> python tests/study/balanced_force_cost.py
Timings on a shared host are noisy: OFF and ON are interleaved per repeat and the median taken.
"""
import sys
import time

import numpy as np

import peclet.flow as pf

N, R, SIGMA, MU, RATIO, U0 = 32, 8.0, 1.0, 0.1, 1000.0, 0.02
STEPS, WARM, REPEATS = 30, 5, 3


def sphere_c(sub=4):
    off = (np.arange(sub) + 0.5) / sub
    c = np.array([N / 2 + 0.13, N / 2 + 0.27, N / 2 + 0.11])
    out = np.zeros((N, N, N))
    for a in off:
        for b in off:
            for d in off:
                X, Y, Z = np.meshgrid(np.arange(N) + a, np.arange(N) + b, np.arange(N) + d,
                                      indexing="ij")
                out += ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 < R * R)
    return np.asfortranarray(out / sub ** 3)


def build(cls, on, u0, kappa=None):
    s = cls(N, N, N)
    s.set_rho(RATIO)
    s.set_mu(MU)
    s.set_dt(1.0)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, N), 10.0)))
    s.enable_vof()
    s.set_vof(sphere_c())
    s.set_property_model("rho", "linear", "C", [1.0, RATIO - 1.0])
    s.set_surface_tension(SIGMA)
    if kappa is not None:  # a constant curvature: the drop at rest is an EXACT equilibrium
        s.diagnostics.set_vof_kappa_constant(kappa)
    if cls is pf.Solver:
        s.enable_vof_momentum(1.0, RATIO)
    s.set_dt(0.5 * s.capillary_dt())
    if u0 != 0.0:
        s.set_field("u", np.asfortranarray(np.full((N, N, N), u0)))
    s.set_balanced_force_projection(on)
    s.set_pressure_chebyshev(True, 500, 1e-9)
    return s


def run(cls, on, u0, steps, kappa=None):
    s = build(cls, on, u0, kappa)
    t, pit, bit = [], [], []
    for k in range(steps):
        t0 = time.perf_counter()
        s.step()
        t1 = time.perf_counter()
        if k >= WARM:
            t.append(t1 - t0)
            pit.append(s.diagnostics.last_pressure_iterations())
            bit.append(s.diagnostics.last_balanced_force_iterations())
    u = max(np.max(np.abs(s.get_u())), np.max(np.abs(s.get_v())), np.max(np.abs(s.get_w())))
    if not np.isfinite(u):
        raise RuntimeError("non-finite velocity")
    return np.median(t), np.mean(pit), np.mean(bit)


def main():
    ok = True
    # static: the drop at rest, height-function curvature (parasitic currents keep the force
    # changing at a small relative level); static-k: the same with a constant curvature 1/R, an
    # exact equilibrium, where the increment solve must skip after the first step.
    for case, u0, steps, kap in (("moving", U0, STEPS, None), ("static", 0.0, 50, None),
                                 ("static-k", 0.0, 50, 1.0 / R)):
        for name, cls in (("staggered", pf.Solver), ("collocated", pf.SolverColocated)):
            res = {False: [], True: []}
            for _ in range(REPEATS):
                for on in (False, True):
                    res[on].append(run(cls, on, u0, steps, kap))
            off = np.median([r[0] for r in res[False]])
            onn = np.median([r[0] for r in res[True]])
            p_off, p_on = res[False][0][1], res[True][0][1]
            b_on = res[True][0][2]
            ratio = onn / off
            good = ratio <= 1.5 and b_on <= p_on
            ok &= good
            print(f"{case:8s} {name:10s} step OFF {1e3 * off:7.2f} ms  ON {1e3 * onn:7.2f} ms  "
                  f"ratio {ratio:.3f}   iters: main OFF {p_off:.1f}, main ON {p_on:.1f}, balanced-force ON {b_on:.1f}   "
                  f"{'ok' if good else 'MISS'}")
    print("G7 PASS" if ok else "G7 MISS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
