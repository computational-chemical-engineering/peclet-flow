#!/usr/bin/env python
"""Gate: set_vof_viscous_interface('auto' | 'sharp' | 'averaged') resolves per flow
doc/vof_sharp_viscous_stress.md §9 (USER 2026-11-04: 'auto' is the default).

One step of a small periodic VoF drop (mu = LinearMix(C), ratio 10) per configuration; the sharp
interface is in force iff diagnostics.vof_viscous_interface_census()['band_cells'] > 0.
  1. the default is 'auto', and every mode round-trips; an unknown one raises ValueError and leaves
     the selection unchanged;
  2. 'auto' -> sharp on the staggered Solver with VoF, mu = LinearMix(C) and the full stress;
     'sharp' -> sharp; 'averaged' -> not;
  3. 'auto' -> averaged, with ONE stderr notice over two steps, for: the Laplacian viscous stress,
     an immersed solid, porous momentum, phase change, the harmonic face viscosity, and
     SolverColocated; silently for a uniform mu (LinearMix with mu_1 = mu_0);
  4. an explicit 'sharp' raises at step() in each of those staggered configurations and without VoF;
     SolverColocated raises at the setter.

Run:  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \
          python tests/python/test_vof_viscous_interface_auto.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import os
import sys
import tempfile

import numpy as np

N = 12
NOTICE = "AUTO viscous interface fell back to 'averaged'"


def drop(n):
    x = np.arange(n) + 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt((X - 0.5 * n - 0.13) ** 2 + (Y - 0.5 * n - 0.27) ** 2 + (Z - 0.5 * n - 0.11) ** 2)
    return np.asfortranarray(np.clip(0.5 * n * 0.3 - r + 0.5, 0.0, 1.0))


def build(pf, cls=None, vof=True, mu1=10.0, setup=None):
    s = (cls or pf.Solver)(N, N, N)
    s.set_advection(False)
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(0.1)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, N), 10.0)))
    if vof:
        s.enable_vof()
        s.set_vof(drop(N))
        s.set_property_model("mu", "linear", "C", [1.0, mu1 - 1.0])
    if setup is not None:
        setup(s)
    return s


def run(s, steps=1):
    """Step `steps` times capturing fd 2; returns (band_cells after the last step, stderr text)."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as f:
        os.dup2(f.fileno(), 2)
        try:
            for _ in range(steps):
                s.step()
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
        f.seek(0)
        err = f.read().decode(errors="replace")
    return s.diagnostics.vof_viscous_interface_census()["band_cells"], err


def main():
    try:
        import peclet.flow as pf
    except ImportError as e:
        print(f"SKIP: peclet.flow not importable ({e})")
        return 77
    fails = []

    def check(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # 1. names
    s = build(pf)
    check(s.vof_viscous_interface() == "auto", f"default is {s.vof_viscous_interface()!r}")
    for m in ("sharp", "averaged", "auto"):
        s.set_vof_viscous_interface(m)
        check(s.vof_viscous_interface() == m, f"{m!r} round-trips to {s.vof_viscous_interface()!r}")
    try:
        s.set_vof_viscous_interface("harmonic")
        check(False, "'harmonic' did not raise")
    except ValueError as e:
        check(s.vof_viscous_interface() == "auto", f"'harmonic' -> ValueError ({e}), selection kept")
    check(not hasattr(s.diagnostics, "set_vof_viscous_interface"),
          "the developer-tier diagnostics.set_vof_viscous_interface is gone")

    # 2. in scope
    for mode, want in (("auto", True), ("sharp", True), ("averaged", False)):
        s = build(pf, setup=lambda t, m=mode: t.set_vof_viscous_interface(m))
        band, err = run(s, 2)
        check((band > 0) == want and NOTICE not in err,
              f"in scope, {mode!r}: band_cells {band}, sharp {'on' if want else 'off'}, no notice")

    # 3./4. out of scope: (label, setup) on the staggered Solver
    def solid(t):
        x = np.arange(N) + 0.5
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        sdf = np.sqrt((X - 1.5) ** 2 + (Y - 1.5) ** 2 + (Z - 1.5) ** 2) - 1.0
        t.set_solid(np.asfortranarray(sdf), cutcell_pressure=True)

    out = (
        ("laplacian stress", lambda t: t.set_viscous_stress("laplacian")),
        ("immersed solid", solid),
        ("porous momentum", lambda t: t.set_porous_continuity(True)),
        ("phase change", lambda t: t.enable_phase_change(1.0, 10.0, 1.0)),
        ("harmonic face mu", lambda t: t.diagnostics.set_property_mode("variable", True)),
    )
    for label, setup in out:
        try:
            band, err = run(build(pf, setup=setup), 2)
            check(band == 0 and err.count(NOTICE) == 1,
                  f"{label}, 'auto': band_cells {band}, notices {err.count(NOTICE)} (want 0, 1)")
        except Exception as e:  # noqa: BLE001 -- any raise under AUTO is a failure
            check(False, f"{label}, 'auto' raised {type(e).__name__}: {e}")

        def both(t, setup=setup):
            setup(t)
            t.set_vof_viscous_interface("sharp")

        try:
            run(build(pf, setup=both))
            check(False, f"{label}, 'sharp': did not raise")
        except RuntimeError as e:
            check("set_vof_viscous_interface('sharp')" in str(e), f"{label}, 'sharp' raises: {e}")

    # uniform mu: silent and inert
    band, err = run(build(pf, mu1=1.0), 2)
    check(band == 0 and NOTICE not in err, f"uniform mu, 'auto': band_cells {band}, no notice")
    # no VoF: silent under AUTO, refused under 'sharp'
    band, err = run(build(pf, vof=False), 1)
    check(band == 0 and NOTICE not in err, f"no VoF, 'auto': band_cells {band}, no notice")
    try:
        run(build(pf, vof=False, setup=lambda t: t.set_vof_viscous_interface("sharp")))
        check(False, "no VoF, 'sharp': did not raise")
    except RuntimeError as e:
        check("enable_vof" in str(e), f"no VoF, 'sharp' raises: {e}")

    # SolverColocated: 'auto' -> averaged with the notice; 'sharp' raises at the setter
    s = build(pf, cls=pf.SolverColocated)
    check(s.vof_viscous_interface() == "auto", "SolverColocated default is 'auto'")
    band, err = run(s, 2)
    check(band == 0 and err.count(NOTICE) == 1,
          f"SolverColocated, 'auto': band_cells {band}, notices {err.count(NOTICE)} (want 0, 1)")
    try:
        s.set_vof_viscous_interface("sharp")
        check(False, "SolverColocated 'sharp' did not raise")
    except RuntimeError as e:
        check(s.vof_viscous_interface() == "auto", f"SolverColocated 'sharp' raises: {e}")

    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
