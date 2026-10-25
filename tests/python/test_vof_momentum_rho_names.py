#!/usr/bin/env python
"""Gate: enable_vof_momentum takes (rho_c0, rho_c1); the former keywords rho_gas / rho_liquid work.

The pair is (density at C = 0, density at C = 1), whichever phase that is. The old names assumed
C = 1 marks the liquid and misled callers whose C = 1 is the gas (the Q6 runs). `rho_c0=`/`rho_c1=`
is the canonical spelling (suite/docs/NAMING.md section 0.1); `rho_gas=`/`rho_liquid=` stay as an
overload for this release.

Checks, on a 12^3 periodic drop moving through a density-ratio-10 fluid for a few steps:
  1. positional, rho_c0=/rho_c1= and rho_gas=/rho_liquid= give bitwise identical u, v, w, p, C;
  2. mixed keywords (rho_c0=, rho_liquid=) raise TypeError;
  3. a pair swapped against the registered 'rho' closure raises, and the message names rho_c0 and
     rho_c1 and says SWAPPED.

Run:  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \
          python tests/python/test_vof_momentum_rho_names.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

N, STEPS = 12, 5
RHO_C0, RHO_C1 = 1.0, 10.0
CALLS = {
    "positional": lambda s: s.enable_vof_momentum(RHO_C0, RHO_C1),
    "canonical keywords": lambda s: s.enable_vof_momentum(rho_c0=RHO_C0, rho_c1=RHO_C1),
    "former keywords": lambda s: s.enable_vof_momentum(rho_gas=RHO_C0, rho_liquid=RHO_C1),
}


def build(pf, enable):
    s = pf.Solver(N, N, N)
    s.set_rho(RHO_C1)
    s.set_mu(0.05)
    s.set_dt(0.2)
    s.set_advection(False)
    s.set_pressure_geometry(np.full((N, N, N), 1.0, order="F"))
    s.set_pressure_pcg(True, 800, 1e-10)
    s.enable_vof()
    x = (np.arange(N) + 0.5)[:, None, None]
    y = (np.arange(N) + 0.5)[None, :, None]
    z = (np.arange(N) + 0.5)[None, None, :]
    r2 = (x - 0.5 * N) ** 2 + (y - 0.5 * N) ** 2 + (z - 0.5 * N) ** 2
    s.set_vof(np.asfortranarray(np.clip(0.5 + 3.0 - np.sqrt(r2), 0.0, 1.0)))
    s.set_property_model("rho", "linear", "C", [RHO_C0, RHO_C1 - RHO_C0])
    if enable is not None:
        enable(s)
    one = np.ones((N, N, N), order="F")
    s.set_state(0.5 * one, 0.3 * one, 0.0 * one)
    return s


def run(pf, enable):
    s = build(pf, enable)
    for _ in range(STEPS):
        s.step()
    assert s.vof_momentum_enabled()
    return {"u": s.get_u(), "v": s.get_v(), "w": s.get_w(), "p": s.get_p(), "C": s.get_vof()}


def main():
    try:
        import peclet.flow as pf
    except ImportError:
        print("SKIP: peclet.flow not importable (set PYTHONPATH to the build tree)")
        return 77
    try:
        ref = run(pf, CALLS["positional"])
        assert float(np.max(np.abs(ref["u"] - 0.5))) > 0.0 or float(np.max(np.abs(ref["p"]))) > 0.0, \
            "the fields did not evolve: the test is vacuous"
        for name in ("canonical keywords", "former keywords"):
            got = run(pf, CALLS[name])
            for k in ref:
                assert np.array_equal(ref[k], got[k]), f"{name}: field {k} differs from positional"
            print(f"  {name}: bitwise identical to positional")
        for bad in (lambda s: s.enable_vof_momentum(rho_c0=RHO_C0, rho_liquid=RHO_C1),
                    lambda s: s.enable_vof_momentum(rho_gas=RHO_C0, rho_c1=RHO_C1)):
            try:
                bad(build(pf, None))
            except TypeError:
                pass
            else:
                raise AssertionError("mixed keywords did not raise TypeError")
        print("  mixed keywords: TypeError")
        try:
            build(pf, lambda s: s.enable_vof_momentum(RHO_C1, RHO_C0))
        except RuntimeError as e:
            msg = str(e)
            assert "rho_c0" in msg and "rho_c1" in msg and "SWAPPED" in msg, f"message: {msg}"
            assert "rho_gas" not in msg and "rho_liquid" not in msg, f"message: {msg}"
        else:
            raise AssertionError("a swapped pair against the 'rho' closure did not raise")
        print("  swapped pair: error names rho_c0 / rho_c1")
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    print("OK: enable_vof_momentum(rho_c0, rho_c1) == positional == former keywords")
    return 0


if __name__ == "__main__":
    sys.exit(main())
