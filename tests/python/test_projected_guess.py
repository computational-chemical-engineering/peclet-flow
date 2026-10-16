#!/usr/bin/env python
"""Gate: the projected initial guess of the projection solve (E, doc/vof_projection_cost_design.md
§5.3; `diagnostics.set_pressure_initial_guess`), opt-in, on the single-rank PCG and FCG drivers.

Configuration: the IBM cylinder (16^3, cylinder along y of radius N/4, cut-cell pressure, Stokes,
body force 1e-3, rtol 1e-8), 'projected' with history 2 against 'zero' on the same build.
  G-E3  step 1 of 'projected' (empty history: the cold solve) is bitwise step 1 of 'zero'; after a
        state write (set_velocity: the history restarts) the next step is bitwise the same step
        of a twin switched to 'zero' -- and the twin continued under 'projected' is NOT, so the
        guess was live;
  G-E2  every guessed solve ends at max|P(b - A x)| <= 1.5 rtol max|P b| (recomputed; the trace of
        a child run under PECLET_FLOW_MG_DEBUG=2 and _SOLVES, debug output only), x0 = 0 on
        every non-fluid cell, and 'projected' saves iterations over the run;
  raise 'projected' under the Chebyshev driver raises at the step, naming the condition; 'auto'
        there runs the cold solve, bitwise 'zero'.

Run: OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> python tests/python/test_projected_guess.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import os
import re
import subprocess
import sys

import numpy as np

try:
    import peclet.flow as pf
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

N, RTOL, STEPS = 16, 1e-8, 12
fails = 0


def check(ok, what):
    global fails
    print(f"  {what:96s} {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        fails += 1


def solver(guess, driver="pcg"):
    X, _, Z = np.meshgrid(*(np.arange(N) + 0.5,) * 3, indexing="ij")
    sdf = np.sqrt((X - N / 2) ** 2 + (Z - N / 2) ** 2) - 0.25 * N
    s = pf.Solver(N, N, N)
    s.set_rho(1.0)
    s.set_mu(0.01)
    s.set_dt(1.0)
    s.set_advection(False)
    s.set_body_force((1e-3, 0.0, 0.0))
    s.set_pressure_multigrid(True, levels=3)
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True)
    if driver == "fcg":
        s.set_pressure_fcg(True, 400, RTOL)
    elif driver == "cheb":
        s.set_pressure_chebyshev(True, 400, RTOL)
    else:
        s.set_pressure_pcg(True, 400, RTOL)
    s.diagnostics.set_pressure_initial_guess(guess, 2)
    return s


def state(s):
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel()
                           for a in (s.get_u(), s.get_v(), s.get_w(), s.get_p())])


def same(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def child(driver):
    """G-E2 run: 'projected' for STEPS steps (traced by the parent's env)."""
    s = solver("projected", driver)
    for _ in range(STEPS):
        s.step()


def e3(driver):
    a, z = solver("projected", driver), solver("zero", driver)
    a.step()
    z.step()
    check(same(state(a), state(z)), f"{driver}: G-E3 step 1 of 'projected' == 'zero', bitwise")
    b = solver("projected", driver)  # twin of a
    b.step()
    for _ in range(4):
        a.step()
        b.step()
    c = solver("projected", driver)  # a continued without any reset
    for _ in range(5):
        c.step()
    for k in range(3):  # the state write: restarts a's history, same values
        a.set_velocity(k, np.asfortranarray(np.asarray((a.get_u(), a.get_v(), a.get_w())[k])))
    b.diagnostics.set_pressure_initial_guess("zero", 2)
    for t in (a, b, c):
        t.step()
    check(same(state(a), state(b)),
          f"{driver}: G-E3 the step after set_velocity == the twin's 'zero' step, bitwise")
    check(not same(state(c), state(b)), f"{driver}: the guess is live (no reset: not bitwise 'zero')")


def e2(driver):
    env = dict(os.environ, PECLET_FLOW_MG_DEBUG="2", PECLET_FLOW_MG_DEBUG_SOLVES=str(STEPS + 2))
    out = subprocess.run([sys.executable, __file__, "--child", driver], env=env,
                         capture_output=True, text=True, check=False)
    rows = re.findall(r"\[mg\] guess solve \d+: hist (\d) .*? iters (\d+) .*?resid/rref (\S+) "
                      r"x0_nonfluid_nonzero (\d+)", out.stdout)
    guessed = [r for r in rows if int(r[0]) > 0]
    worst = max((float(r[2]) for r in guessed), default=float("nan"))
    nz = sum(int(r[3]) for r in guessed)
    check(out.returncode == 0 and len(guessed) >= STEPS - 2 and worst <= 1.5 * RTOL and nz == 0,
          f"{driver}: G-E2 {len(guessed)} guessed solves, max resid/rref {worst:.3e} <= 1.5 rtol, "
          f"x0 on solids {nz}")
    zero = solver("zero", driver)
    proj = solver("projected", driver)
    iz = ip = 0
    for _ in range(STEPS):
        zero.step()
        proj.step()
        iz += zero.diagnostics.last_pressure_iterations()
        ip += proj.diagnostics.last_pressure_iterations()
    check(ip < iz, f"{driver}: iterations over {STEPS} steps 'zero' {iz} -> 'projected' {ip}")


def raises():
    s = solver("projected", "cheb")
    what = ""
    try:
        s.step()
    except RuntimeError as e:
        what = str(e)
    check("not eligible here" in what and "Chebyshev" in what,
          "Chebyshev: 'projected' raises at the step: " + (what[:60] or "(no raise!)"))
    a, z = solver("auto", "cheb"), solver("zero", "cheb")
    for _ in range(3):
        a.step()
        z.step()
    check(same(state(a), state(z)), "Chebyshev: 'auto' == 'zero', bitwise")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
        sys.exit(0)
    print("projected_guess (doc/vof_projection_cost_design.md §5.3, G-E2, G-E3)")
    for d in ("pcg", "fcg"):
        e3(d)
        e2(d)
    raises()
    print(f"{'FAIL' if fails else 'PASS'} ({fails} failure{'s' if fails != 1 else ''})")
    sys.exit(1 if fails else 0)
