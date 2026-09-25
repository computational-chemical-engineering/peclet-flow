#!/usr/bin/env python
"""The momentum solver never silently drops a variable viscosity.

The velocity multigrid takes a SCALAR mu. Combined with a variable viscosity (a 'mu' closure or
set_property_mode('variable')) it used to solve with the constant set_mu value instead: on a
mirror-pair case the explicit multigrid with mu(T) differed from its own constant-mu run by only
2e-4 and from the red-black smoother by 50-70 %. Now:

  1. an EXPLICIT set_velocity_solver('multigrid') / set_velocity_multigrid(True) raises ValueError
     with a variable viscosity, in either order (solver first, then the closure / the mode; or
     the other way round), and names the cause;
  2. the AUTO rule (which picks the V-cycle once kappa = 1 + 4 dt mu sum(w) / rho >= 13) keeps a
     variable viscosity on the red-black smoother -- velocity_multigrid_active() is False after
     the first step, while the same case at constant mu takes the V-cycle -- and the variable-mu
     result equals the explicit 'gauss_seidel' run bitwise.

Run: OMP_NUM_THREADS=4 PYTHONPATH=<build> python tests/python/test_velocity_solver_variable_mu.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

try:
    import peclet.flow as pf
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

N = 16


def make(variable_mu, solver=None, dt=2.0):
    """Walled box, a transported scalar T driving the body force (and mu(T) when variable_mu).
    dt = 2, mu = 0.5: kappa = 1 + 12 * 1.0 = 13 -> AUTO picks the V-cycle at constant mu."""
    s = pf.Solver(N, N, N)
    s.set_rho(1.0)
    s.set_mu(0.5)
    s.set_dt(dt)
    s.set_domain_bc("-y", "wall", (0, 0, 0))
    s.set_domain_bc("+y", "wall", (0, 0, 0))
    s.set_pressure_geometry(np.full((N, N, N), 10.0, order="F"))
    s.add_scalar("T", 0.0, "koren")
    X, Y, Z = np.meshgrid(*(np.arange(N) + 0.5,) * 3, indexing="ij")
    s.set_field("T", np.asfortranarray(np.exp(-((X - 8) ** 2 + (Y - 5) ** 2 + (Z - 8) ** 2) / 8)))
    if solver:
        s.diagnostics.set_velocity_solver(solver)
    if variable_mu:
        s.set_property_model("mu", "linear", "T", [0.5, -0.4])
    s.set_property_model("force_x", "linear", "T", [0.0, -0.02])
    return s


def raises(fn):
    try:
        fn()
    except ValueError as e:
        return "scalar viscosity" in str(e).lower() or "SCALAR viscosity" in str(e)
    return False


def main():
    ok = True

    def check(name, good):
        nonlocal ok
        ok &= bool(good)
        print(f"{name:66s} {'ok' if good else 'FAIL'}")

    # 1. explicit multigrid + variable viscosity: refused in every order
    def closure_then_solver():
        s = make(True)
        s.diagnostics.set_velocity_solver("multigrid")

    def solver_then_closure():
        make(True, solver="multigrid")

    def solver_then_mode():
        s = make(False, solver="multigrid")
        s.diagnostics.set_property_mode("variable")

    def closure_then_raw_setter():
        s = make(True)
        s.set_velocity_multigrid(True)

    check("explicit: mu closure, then set_velocity_solver('multigrid')", raises(closure_then_solver))
    check("explicit: set_velocity_solver('multigrid'), then mu closure", raises(solver_then_closure))
    check("explicit: set_velocity_solver('multigrid'), then property mode", raises(solver_then_mode))
    check("explicit: mu closure, then set_velocity_multigrid(True)", raises(closure_then_raw_setter))
    s = make(False, solver="multigrid")  # constant mu: the explicit V-cycle stays legal
    s.step()
    check("explicit multigrid at constant mu still runs", s.diagnostics.velocity_multigrid_active())

    # 2. AUTO: the V-cycle at constant mu, the smoother under a variable mu, bitwise == explicit GS
    c = make(False)
    c.step()
    check("AUTO, constant mu, kappa 13: V-cycle", c.diagnostics.velocity_multigrid_active())
    a, g = make(True), make(True, solver="gauss_seidel")
    for _ in range(3):
        a.step()
        g.step()
    check("AUTO, variable mu: red-black smoother", not a.diagnostics.velocity_multigrid_active())
    same = all(np.array_equal(x, y) for x, y in
               ((a.get_u(), g.get_u()), (a.get_v(), g.get_v()), (a.get_w(), g.get_w()),
                (a.get_p(), g.get_p())))
    check("AUTO, variable mu == explicit gauss_seidel, bitwise (3 steps)", same)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
