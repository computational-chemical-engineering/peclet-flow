#!/usr/bin/env python3
"""ctest `march_to_steady`: peclet.flow.march_to_steady and its Anderson accelerator
(doc/steady_acceleration.md rev 1, work order WO-4).

* G0(b) — `march_to_steady(accelerate=False)` IS the study's march (tests/study/
  study_avg_velocity_spheres.py, the instrument copied verbatim below as `study_march`): the §11
  case (one sphere, phi = 0.125, periodic unit cube, rho = mu = F = 1, nu dt / h^2 = 6, Stokes,
  PCG(200, 1e-8), 200 velocity sweeps), N = 16, staggered and collocated 'ghost': identical step
  count and bit-identical <u_x>.
* a small G1 — the same case at N = 12, tight inner solves (PCG(400, 1e-12), velocity residual
  tolerance 1e-12, rtol 1e-10), accelerated and plain both `converged=True` with
  |K_acc / K_plain - 1| <= 1e-8, on both grids.
* G7a — staggered N = 16 at mu = 0.0158, dt = 1.234e-2, SOU advection, whose plain march diverges:
  `converged=False` with accelerate=True and accelerate=False.
* three pure-Python tests of the driver's certification (`peclet.flow.steady._certify`) on scripted
  monitor sequences: (i) four blocks with R > 1, then a geometric tail inside the budget -> "pass";
  (ii) a clean tail at R = 0.99 that fails the remainder bound -> "slow" at the first such block;
  (iii) a sequence that never passes -> "budget" after exactly 2 (num_passes + 3) blocks.

Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import math
import sys

import numpy as np

try:
    import peclet.flow as pf
    from peclet.flow import steady
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

PHI = 0.125
L = 1.0
FAILED = []


def check(ok, what):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}", flush=True)
    if not ok:
        FAILED.append(what)


# ---- the study's instrument, verbatim (tests/study/study_avg_velocity_spheres.py, march()) ------
TOL_K, CHECK_EVERY, N_PASS, MAX_STEPS = 1e-4, 5, 3, 1500
RHO_SLOW = 0.997
ROUNDOFF = 1e-11


def study_march(step, umean):
    prev = dprev = None
    passes = 0
    for it in range(MAX_STEPS):
        step()
        if it % CHECK_EVERY != CHECK_EVERY - 1:
            continue
        m = umean()
        if prev is not None:
            d = m - prev
            ok = abs(d) <= ROUNDOFF * abs(m)
            if not ok and dprev:
                R = d / dprev
                ok = 0.0 < R < 1.0 and \
                    abs(d) / (1.0 - max(R, RHO_SLOW ** CHECK_EVERY)) < TOL_K * abs(m)
            passes = passes + 1 if ok else 0
            if passes >= N_PASS:
                return it + 1, True
            dprev = d
        prev = m
    return MAX_STEPS, False


def sphere_solver(Cls, N, *, tight, mu=1.0, dt=None, advection=None):
    s = Cls((N, N, N), extent=(L, L, L))
    s.set_rho(1.0)
    s.set_mu(mu)
    h = L / N
    s.set_dt(dt if dt is not None else 6.0 * h * h / mu)
    s.set_body_force((1.0, 0.0, 0.0))
    if advection:
        s.set_advection(True)
        s.set_advection_scheme(advection)
    else:
        s.set_advection(False)
    s.diagnostics.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=max(2, int(np.log2(N)) - 1))
    if tight:
        s.set_pressure_pcg(True, max_iter=400, rtol=1e-12)
        s.set_velocity_residual_tolerance(1e-12)
    else:
        s.set_pressure_pcg(True, max_iter=200, rtol=1e-8)
    cx, cy, cz = s.cell_centers()
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    R = (3.0 * PHI / (4.0 * math.pi)) ** (1.0 / 3.0) * L
    sdf = np.sqrt((X - 0.5 * L) ** 2 + (Y - 0.5 * L) ** 2 + (Z - 0.5 * L) ** 2) - R
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True)
    return s


def umean(s):
    return float(s.get_u().mean())


SCHEMES = (("staggered", pf.Solver), ("collocated", pf.SolverColocated))


def gate_g0b():
    print("G0(b): march_to_steady(accelerate=False) == the study's march, N = 16", flush=True)
    for name, Cls in SCHEMES:
        a = sphere_solver(Cls, 16, tight=False)
        steps_a, ok_a = study_march(a.step, lambda a=a: umean(a))
        u_a = umean(a)
        b = sphere_solver(Cls, 16, tight=False)
        res = pf.march_to_steady(b, lambda b=b: umean(b), rtol=TOL_K, max_steps=MAX_STEPS,
                                 accelerate=False)
        check(ok_a and res.converged and res.steps == steps_a and res.monitor == u_a
              and umean(b) == u_a and res.accelerated_steps == 0,
              f"{name}: study {steps_a} steps <u_x> {u_a!r}; march_to_steady {res.steps} steps "
              f"<u_x> {res.monitor!r} ({res.reason})")


def gate_g1_small():
    print("G1 (small): tight, N = 12, |K_acc / K_plain - 1| <= 1e-8", flush=True)
    for name, Cls in SCHEMES:
        out = {}
        for acc in (False, True):
            s = sphere_solver(Cls, 12, tight=True)
            res = pf.march_to_steady(s, lambda s=s: umean(s), rtol=1e-10, max_steps=20000,
                                     accelerate=acc)
            out[acc] = res
        p, a = out[False], out[True]
        rel = abs(p.monitor / a.monitor - 1.0)  # K ~ 1 / <u_x>: K_acc / K_plain = u_p / u_a
        check(p.converged and a.converged and rel <= 1e-8,
              f"{name}: plain {p.steps} steps, accelerated {a.steps} ({a.accelerated_steps} "
              f"accelerated, {a.num_restarts} restarts), |K_acc/K_plain - 1| = {rel:.2e}")


def gate_g7a():
    print("G7a: a case whose plain march diverges -> converged=False for both drivers",
          flush=True)
    for acc in (False, True):
        s = sphere_solver(pf.Solver, 16, tight=False, mu=0.0158, dt=1.234e-2, advection="sou")
        res = pf.march_to_steady(s, lambda s=s: umean(s), rtol=1e-4, max_steps=5000,
                                 accelerate=acc)
        check(not res.converged and res.reason in ("diverged", "unstable", "max_steps"),
              f"accelerate={acc}: converged={res.converged} reason={res.reason} "
              f"steps={res.steps}")


def scripted(values):
    """A fake step and a monitor that returns values[k] after the k-th step."""
    state = {"k": 0}

    def step():
        state["k"] += 1

    def monitor():
        return values[state["k"] - 1]

    return step, monitor, state


def certify(values, **kw):
    step, monitor, state = scripted(values)
    counter = steady._Counter()
    args = dict(rtol=1e-4, check_every=5, num_passes=3, slow_rate=0.997, roundoff=1e-11,
                max_steps=10 ** 6, counter=counter, callback=None, phase="certify")
    args.update(kw)
    out = steady._certify(step, monitor, **args)
    return out, counter.steps


def per_block(blocks):
    """A monitor sequence (one value per step) whose block-end values are `blocks`."""
    seq = []
    for v in blocks:
        seq += [float("nan")] * 4 + [v]  # only every 5th step is read
    return seq


def gate_certify():
    print("driver certification on scripted monitor sequences", flush=True)
    budget = 2 * (3 + 3)
    # (i) block changes that GROW (R > 1) for four blocks, at |d| ~ 1e-9 of |m|, then a clean
    # geometric tail with R = 0.5: passes inside the 12-block budget.
    m0 = 1.0
    blocks = [m0]
    d = 1e-9
    for _ in range(4):
        d *= 3.0
        blocks.append(blocks[-1] + d)
    d = 2e-6
    for _ in range(6):
        d *= 0.5
        blocks.append(blocks[-1] + d)
    out, steps = certify(per_block(blocks), budget=budget, slow_exit=True,
                         growth_residual=lambda: 0.0, growth_ref=1.0)
    check(out == "pass" and steps <= 5 * budget, f"(i) R > 1 then a geometric tail: {out} after "
          f"{steps} steps ({steps // 5} blocks)")
    # (ii) a clean geometric tail at R = 0.99 (>= 0.997^5 = 0.985) whose remainder bound fails:
    # "slow" at the first block where R is defined.
    blocks = [1.0]
    d = 1e-4
    for _ in range(8):
        blocks.append(blocks[-1] + d)
        d *= 0.99
    out, steps = certify(per_block(blocks), budget=budget, slow_exit=True,
                         growth_residual=lambda: 0.0, growth_ref=1.0)
    check(out == "slow" and steps == 15, f"(ii) clean R = 0.99 tail: {out} after {steps} steps "
          "(expected 'slow' at step 15, the first block with a ratio)")
    # (iii) a sequence that never passes (alternating sign): "budget" after exactly 12 blocks.
    blocks = [1.0 + (1e-3 if k % 2 else 0.0) for k in range(40)]
    out, steps = certify(per_block(blocks), budget=budget, slow_exit=True,
                         growth_residual=lambda: 0.0, growth_ref=1.0)
    check(out == "budget" and steps == 5 * budget,
          f"(iii) never passing: {out} after {steps} steps (expected 'budget' at {5 * budget})")


def main():
    gate_certify()
    gate_g0b()
    gate_g1_small()
    gate_g7a()
    print("PASS" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


def test_march_to_steady():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
