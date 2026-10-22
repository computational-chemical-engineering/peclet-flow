#!/usr/bin/env python
"""Gate: block VoF with CSF gives the same flow in a physical domain as in cell units.

Why this test exists. Two unit bugs were reachable only under `Solver(cells, extent=...)` with the
block container (`enable_vof_blocks` + `enable_vof_block_csf`), and nothing ran that combination:
  1. step() handed its INTERNAL dt to the physical-unit block advector, which converted it again;
     with tRef pinned by the first set_dt the first step threw the Weymouth-Yue cap.
  2. enable_vof_block_csf() copied the internal sigma into the container before the first set_dt
     pinned tRef; the block CSF then ran 1/tRef^2 (~1e9 here) too strong.
In cell units tRef = 1, so both are invisible there (flow ac09b68).

The case: one block-marker bubble (d = 10 h) in a periodic 16^3 box, the case-A groups of the
bubble-swarm validation (Ar 29.9, Bo 2, density and viscosity ratio 0.1), buoyancy with the mixture
weight removed, the volume-mean u_x held at 0. It runs twice: in cell units (dt = 1, rho_l = 1), and
in a physical box (h = 1e-4 m, rho_l = 1000 kg/m^3, dt = 2.5e-5 s) whose every input is the cell-unit
value times its unit scale. After STEPS steps u, v, w, p (converted with `unit_scales`) and C must
match to <= 1e-12 relative to each field's maximum. set_dt comes after enable_vof_block_csf on
purpose: that is the drivers' order (tests/study/case_a) and the one that exposed bug 2.

Run:  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \
          python tests/python/test_vof_blocks_physical_units.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import math
import sys

import numpy as np

N, D_CELLS, STEPS, RTOL = 16, 10.0, 30, 1e-12
AR, BO, RHO_RATIO, MU_RATIO = 29.9, 2.0, 0.1, 0.1
# Cell units: rho_l = 1, dt = 1; g makes dt = 1 about a quarter of the capillary limit.
G_CELL = 6.0e-5
SIGMA_CELL = (1.0 - RHO_RATIO) * G_CELL * D_CELLS**2 / BO
MU_CELL = math.sqrt((1.0 - RHO_RATIO) * G_CELL * D_CELLS**3) / AR
# The physical realisation: deliberately not powers of two.
H, RHO_REF, T_REF = 1.0e-4, 1000.0, 2.5e-5


def build(pf, phys):
    """The same case; `phys` scales every input from cell units to SI."""
    sl = H if phys else 1.0                       # length
    sr = RHO_REF if phys else 1.0                 # density
    st = T_REF if phys else 1.0                   # time
    rho_l, rho_g = sr, RHO_RATIO * sr
    mu_l = MU_CELL * sr * sl * sl / st
    sigma = SIGMA_CELL * sr * sl**3 / (st * st)
    g = G_CELL * sl / (st * st)
    s = pf.Solver((N, N, N), extent=(N * H,) * 3) if phys else pf.Solver(N, N, N)
    s.set_rho(rho_l)
    s.set_mu(mu_l)
    s.set_pressure_geometry(np.full((N, N, N), 1.0, order="F"))
    s.enable_vof()
    s.set_vof(np.zeros((N, N, N), order="F"))
    s.set_property_model("rho", "linear", "C", [rho_l, rho_g - rho_l])
    s.set_property_model("mu", "linear", "C", [mu_l, MU_RATIO * mu_l - mu_l])
    s.set_surface_tension(sigma)
    s.diagnostics.set_pressure_vcycle_precision("fp64")   # FP32 weights are not scale-covariant
    s.set_pressure_pcg(True, 800, 1e-10)
    c = 0.5 * N                                   # grid coordinates, as enable_vof_blocks takes
    s.enable_vof_blocks([(c, c, c, 0.5 * D_CELLS)])
    s.enable_vof_block_csf()
    alpha = float(np.mean(s.get_vof()))
    rho_av = rho_l + (rho_g - rho_l) * alpha
    s.set_property_model("force_x", "linear", "C", [-(rho_l - rho_av) * g, -(rho_g - rho_l) * g])
    s.set_superficial_velocity(True, "x", 0.0)
    s.update_properties()
    s.set_dt(st)                                  # the first set_dt pins tRef (after the CSF)
    return s


def fields(s):
    us = s.unit_scales
    vel = us["velocity_to_internal"]
    return {"u": s.get_u() * vel[0], "v": s.get_v() * vel[1], "w": s.get_w() * vel[2],
            "p": s.get_p() * us["pressure_to_internal"], "C": s.get_vof()}


def main():
    try:
        import peclet.flow as pf
    except ImportError:
        print("SKIP: peclet.flow not importable (set PYTHONPATH to the build tree)")
        return 77
    ref, phy = build(pf, False), build(pf, True)
    us = phy.unit_scales
    try:
        assert not us["identity"], "the extent did not give a physical domain"
        for k, v in (("h_ref", H), ("rho_ref", RHO_REF), ("t_ref", T_REF)):
            assert abs(us[k] / v - 1.0) <= 1e-14, f"unit_scales[{k!r}] = {us[k]!r}, expected {v}"
        cap = ref.vof_step_limits()["capillary_dt"]
        assert cap >= 2.0, f"dt = 1 is not well inside the capillary limit ({cap:.3f})"
        for n in range(STEPS):
            ref.step()
            try:
                phy.step()
            except RuntimeError as e:
                raise AssertionError(f"physical run threw at step {n + 1}: {e}") from e
        a, b = fields(ref), fields(phy)
        worst = 0.0
        for k in a:
            scale = float(np.max(np.abs(a[k])))
            err = float(np.max(np.abs(b[k] - a[k]))) / scale
            print(f"  {k}: max |cell| {scale:.4e}  physical vs cell {err:.2e}")
            assert err <= RTOL, f"{k}: physical vs cell units differ by {err:.3e} > {RTOL:g}"
            worst = max(worst, err)
        assert float(np.max(np.abs(a["u"]))) > 1e-4, "the bubble did not move: the test is vacuous"
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    print(f"OK: block VoF + CSF in a physical domain == cell units over {STEPS} steps "
          f"(worst {worst:.2e}, capillary dt {cap:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
