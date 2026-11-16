#!/usr/bin/env python
"""Code-health reference tier `sharp_vof`: SHA-256 of a short VoF run with the SHARP viscous
interface in force (flow doc/vof_sharp_viscous_stress.md), at one thread.

tests/regression/state_hash.py has no case that resolves the sharp interface ('auto' needs VoF,
mu = LinearMix(C) varying and the full stress; its VoF cases have a constant mu), so a refactor
adjacent to the vof keep-out (MAC stress elements, VoF band worklists; RULES.md) would not be seen
by it. This script is the second half of that rule: "bitwise on state_hash AND a sharp VoF run".

Two staggered cases, a small periodic drop at viscosity ratio 10 under surface tension, five steps:
  sharp_drop          Stokes (set_advection(False)), 'auto' (the default) -> sharp
  sharp_drop_advect   with VoF momentum (rho 1 : 10, conservative VoF fluxes own the advection), 'auto' -> sharp
Each prints one line in state_hash's format plus band_cells; a case whose census reports no band
cells prints 'SHARP INACTIVE' (the tier's fail_regex), so the gate cannot pass on the averaged path.

Run:  OMP_NUM_THREADS=1 OMP_PROC_BIND=false PYTHONPATH=<build> python .codehealth/sharp_vof_hash.py
"""
import hashlib

import numpy as np

import peclet.flow as pf

N = 12


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a), dtype=np.float64).tobytes()).hexdigest()


def drop(n):
    x = np.arange(n) + 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt((X - 0.5 * n - 0.13) ** 2 + (Y - 0.5 * n - 0.27) ** 2 + (Z - 0.5 * n - 0.11) ** 2)
    return np.asfortranarray(np.clip(0.5 * n * 0.3 - r + 0.5, 0.0, 1.0))


def build(advect):
    s = pf.Solver(N, N, N)
    s.set_advection(advect)
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, N), 10.0)))
    s.enable_vof()
    s.set_vof(drop(N))
    s.set_property_model("mu", "linear", "C", [1.0, 9.0])
    if advect:
        s.set_property_model("rho", "linear", "C", [1.0, 9.0])
        s.enable_vof_momentum(1.0, 10.0)
    s.set_surface_tension(1.0)
    s.set_pressure_pcg(True, 500, 1e-12)  # the driver last: the rho closure re-selects Chebyshev
    s.set_dt(0.25 * s.capillary_dt())
    return s


def report(name, s):
    d = {"u": s.get_u(), "v": s.get_v(), "w": s.get_w(), "p": s.get_p(), "C": s.get_vof()}
    band = int(s.diagnostics.vof_viscous_interface_census()["band_cells"])
    h = hashlib.sha256()
    parts = []
    for k in sorted(d):
        hk = sha(d[k]); h.update(hk.encode()); parts.append(f"{k}={hk[:12]}")
    flag = "" if band > 0 else "  SHARP INACTIVE"
    print(f"{name:24s} {h.hexdigest()}  " + " ".join(parts) + f"  band_cells={band}{flag}", flush=True)


if __name__ == "__main__":
    for name, advect in (("sharp_drop", False), ("sharp_drop_advect", True)):
        s = build(advect)
        for _ in range(5):
            s.step()
        report(name, s)
