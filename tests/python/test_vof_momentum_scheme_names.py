#!/usr/bin/env python
"""Gate: diagnostics.set_vof_momentum_scheme(name) / vof_momentum_scheme() (flow
doc/vof_momentum_second_order.md §7, WO-2).

Checks, on a solver with enable_vof_momentum (no step is taken):
  1. the default is 'muscl_phase' (since 2026-10-31; note §12);
  2. every name ('upwind', 'muscl_mixed', 'muscl_mixed_capped', 'muscl_phase') round-trips through
     the getter;
  3. an unknown name raises ValueError and leaves the selection unchanged;
  4. the former spelling set_vof_momentum_muscl(True / False) selects 'muscl_mixed' / 'upwind';
  5. vof_momentum_diagnostics() carries 'capped'.

Run:  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \
          python tests/python/test_vof_momentum_scheme_names.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

NAMES = ("upwind", "muscl_mixed", "muscl_mixed_capped", "muscl_phase")


def main():
    try:
        import peclet.flow as pf
    except ImportError as e:
        print(f"SKIP: peclet.flow not importable ({e})")
        return 77
    n = 8
    s = pf.Solver(n, n, n)
    s.set_rho(10.0)
    s.set_dt(0.2)
    s.set_pressure_geometry(np.full((n, n, n), 1.0, order="F"))
    s.enable_vof()
    s.set_vof(np.zeros((n, n, n), order="F"))
    s.set_property_model("rho", "linear", "C", [1.0, 9.0])
    s.enable_vof_momentum(1.0, 10.0)
    d = s.diagnostics
    fails = []
    if d.vof_momentum_scheme() != "muscl_phase":
        fails.append(f"default is {d.vof_momentum_scheme()!r}, not 'muscl_phase'")
    for name in NAMES:
        d.set_vof_momentum_scheme(name)
        got = d.vof_momentum_scheme()
        print(f"set {name!r} -> {got!r}")
        if got != name:
            fails.append(f"{name!r} round-trips to {got!r}")
    try:
        d.set_vof_momentum_scheme("bogus")
        fails.append("'bogus' did not raise")
    except ValueError as e:
        print(f"'bogus' -> ValueError: {e}")
    if d.vof_momentum_scheme() != NAMES[-1]:
        fails.append("a rejected name changed the selection")
    for on, want in ((True, "muscl_mixed"), (False, "upwind")):
        d.set_vof_momentum_muscl(on)
        got = d.vof_momentum_scheme()
        print(f"set_vof_momentum_muscl({on}) -> {got!r}")
        if got != want:
            fails.append(f"set_vof_momentum_muscl({on}) gives {got!r}, not {want!r}")
    if "capped" not in d.vof_momentum_diagnostics():
        fails.append("vof_momentum_diagnostics() has no 'capped'")
    for f in fails:
        print("FAIL:", f)
    print("PASS" if not fails else "FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
