#!/usr/bin/env python
"""Gate: the balanced-force projection's pressure split P = P_dyn + P_b never counts the balanced
pressure twice (doc/collocated_varrho_forces.md §4.6.2, the re-split rule).

The option keeps P the total pressure and P_b ("p_balanced") its balanced part; a continuing step
does P += X - P_b, P_b = X. When P_b is NOT the split of the current P -- steps with the option off
moved P, or a restart restored "p" but not "p_balanced" -- that update adds the balanced pressure a
second time (measured before the fix on this column: a restart gave max|u| 1.25e-2 against 7.8e-14
continued; OFF 400 steps then ON gave 1.27e-2). The rule: such a step RE-SPLITS (P_b = X, P
unchanged), and restoring "p_balanced" together with "p" keeps the split.

Configuration: a walled ratio-1000 column (8 x 8 x 24, heavy lower half, walls at -z/+z, mu 0.1,
dt 1, g 0.1, Stokes, Chebyshev rtol 1e-12), on SolverColocated (V8: the option is its default) and
on Solver (option explicitly on). Scenarios, each against its twin:
  continued        30 steps, then one more               (the reference level)
  restart          u, v, w, p restored into a fresh solver, one step (re-split)
  restart + P_b    u, v, w, p and p_balanced restored, one step (keeps the split)
  OFF -> ON        400 steps OFF, then ON for 1 and 6 steps, against the OFF run continued
Gates: every scenario's velocity differs from its twin by < 1e-12 (max over u, v, w), and the two
restarts stay below 1e-12 absolute. On a static force a re-split step IS the OFF step, so the
OFF -> ON run tracks the OFF run to round-off; its absolute level is the OFF transient's (1.8e-4
cell velocity on V8 at step 400, exact on the staggered 1-D column), not the continued ON run's.

Run: OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> python tests/python/test_balanced_force_restart.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

try:
    import peclet.flow as pf
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

N, NZ, G = 8, 24, 0.1
TOL = 1e-12


def F(a):
    return np.asfortranarray(np.asarray(a, dtype=np.float64))


RHO = np.where(np.arange(NZ) < NZ // 2, 1000.0, 1.0)[None, None, :] * np.ones((N, N, NZ))


def build(grid, bfp):
    s = (pf.SolverColocated if grid == "colo" else pf.Solver)(N, N, NZ)
    s.set_rho(1.0)
    s.set_mu(0.1)
    s.set_dt(1.0)
    s.set_advection(False)
    s.set_domain_bc("-z", "wall")
    s.set_domain_bc("+z", "wall")
    if grid == "colo":
        s.set_collocated_scheme("gauge-exact")
    s.set_pressure_geometry(F(np.full((N, N, NZ), 10.0)))
    s.enable_cell_force()
    for nm, a in (("force_x", 0 * RHO), ("force_y", 0 * RHO), ("force_z", -RHO * G)):
        s.set_field(nm, F(a))
        s.diagnostics.exchange_field(nm)
    s.add_field("rho")
    s.set_field("rho", F(RHO))
    s.diagnostics.exchange_field("rho")
    s.diagnostics.set_density_mode("variable")
    if bfp is not None:
        s.set_balanced_force_projection(bfp)
    s.set_pressure_chebyshev(True, 2000, 1e-12)
    return s


def vel(s):
    return np.stack([np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w())])


def check(label, u, ref, absolute):
    d = float(np.max(np.abs(u - ref)))
    a = float(np.max(np.abs(u)))
    ok = np.isfinite(d) and d < TOL and (not absolute or a < TOL)
    print(f"  {label:<28s} max|u| {a:.3e}   |u - twin| {d:.3e}  {'ok' if ok else 'FAIL'}")
    return ok


def run(grid):
    on = None if grid == "colo" else True  # V8: the option is the default
    print(f"--- {'SolverColocated (V8, default ON)' if grid == 'colo' else 'Solver (ON)'}")
    ok = True
    a = build(grid, on)
    if not a.balanced_force_projection:
        print("  option not active  FAIL")
        return False
    if "p_balanced" not in a.field_names():  # registered as soon as the option is active
        print("  p_balanced not registered before the first step  FAIL")
        return False
    for _ in range(30):
        a.step()
    st = {n: np.asarray(a.get_field(n)).copy() for n in ("u", "v", "w", "p", "p_balanced")}
    a.step()
    ua = vel(a)
    ok &= check("continued", ua, ua, True)
    for keep_pb in (False, True):
        b = build(grid, on)
        for n, v in st.items():
            if n == "p_balanced" and not keep_pb:
                continue
            b.set_field(n, F(v))
            b.diagnostics.exchange_field(n)
        b.step()
        ok &= check("restart + p_balanced" if keep_pb else "restart (u, v, w, p)", vel(b), ua,
                    True)
    c, d = build(grid, False), build(grid, False)
    for _ in range(400):
        c.step()
        d.step()
    c.set_balanced_force_projection(True)
    for k in range(1, 7):
        c.step()
        d.step()
        if k in (1, 6):
            ok &= check(f"OFF 400 -> ON, step {k}", vel(c), vel(d), False)
    return ok


def main():
    ok = run("colo") & run("stag")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_balanced_force_restart():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
