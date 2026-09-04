#!/usr/bin/env python3
"""WO-V9 item 3 — the g = 3 colour-halo cost under MPI, and what it is a fraction of.

The colour advector owns its own g = 3 block with its own `GridHaloTopology`, and exchanges it
once per sweep (three times a step, plus the momentum advector's two sibling fields per sweep).
That is a deeper halo than the solver's own G = 2 and it is exchanged more often than anything
else in the step, which is why PARIS's partial-column-sum trick (exchange sums over the swept
column instead of the colour band, halving the depth) exists.  This script measures what the
trick would be BUYING before anyone writes it.

Run:
    mpirun -np 1 python tests/study/vof_profile_mpi.py --case hysing1 --steps 20
    mpirun -np 2 python tests/study/vof_profile_mpi.py --case hysing1 --steps 20
    mpirun -np 4 python tests/study/vof_profile_mpi.py --case packed  --steps 20

Needs a `-DPECLET_FLOW_MPI=ON` module on PYTHONPATH.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

import peclet.flow as flow

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vof_profile as vp  # noqa: E402


def slice_block(glob, origin, size):
    o, n = origin, size
    return np.asfortranarray(glob[o[0]:o[0] + n[0], o[1]:o[1] + n[1], o[2]:o[2] + n[2]])


def build_hysing1(nx=64):
    p = vp.HYSING1
    gnz, gny = 2 * nx, 4
    sc = vp.Scale(nx / 1.0)
    R = sc.s * 0.25
    o, n = flow.mpi_block(nx, gny, gnz)
    s = flow.Solver(*n)
    s.set_rho(p["rho1"])
    s.set_mu(sc.mu(p["mu1"]))
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)
    s.init_mpi(nx, gny, gnz)
    s.set_pressure_geometry(np.full(tuple(n), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 600, 1e-12)
    s.enable_vof()
    C = 1.0 - vp.cylinder_fractions((nx, gny, gnz), R, nx / 2.0, sc.s * 0.5)
    s.set_vof(slice_block(C, o, n))
    s.set_property_model("rho", "linear", "C", [p["rho2"], p["rho1"] - p["rho2"]])
    s.set_property_model("mu", "linear", "C", [sc.mu(p["mu2"]), sc.mu(p["mu1"] - p["mu2"])])
    s.set_surface_tension(sc.sigma(p["sigma"]))
    s.enable_vof_momentum(p["rho2"], p["rho1"])
    s.set_property_model("force_z", "linear", "C",
                         [-sc.bodyforce(p["rho2"] * p["g"]),
                          -sc.bodyforce((p["rho1"] - p["rho2"]) * p["g"])])
    s.set_dt(0.5 * s.capillary_dt())
    return s, 600, f"hysing1 {nx}x{gny}x{gnz}"


def build_packed(nx=64, gnz=160):
    NG, RJ, FSDF = 26, 0.171875 * nx, 0.85
    RG = FSDF * RJ
    ZBED, RBUB, ZBUB = 0.5625 * nx, 0.15625 * nx, 0.28125 * nx
    ctr = vp.cached(f"bed_e7_{nx}_{gnz}", lambda: vp.rsa_bed(NG, float(nx), RJ, ZBED, seed=11))
    SDF = vp.cached(f"sdf_e7_{nx}_{gnz}", lambda: vp.bed_sdf(ctr, nx, gnz, RG))
    C0 = vp.cached(f"c0_e7_{nx}_{gnz}", lambda: vp.bubble_colour_sub(nx, gnz, RBUB, ZBUB))
    ratio, Eo, Mo, g, theta = 100.0, 10.0, 1e-3, 2e-4, 60.0
    D = 2 * RBUB
    rho_l, rho_g = 1.0, 1.0 / ratio
    drho = rho_l - rho_g
    sigma = drho * g * D * D / Eo
    mu_l = (Mo * rho_l ** 2 * sigma ** 3 / (g * drho)) ** 0.25
    mu_g = mu_l / ratio
    o, n = flow.mpi_block(nx, nx, gnz)
    s = flow.Solver(*n)
    s.set_rho(rho_l)
    s.set_mu(mu_l)
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)
    s.init_mpi(nx, nx, gnz)
    s.set_solid(slice_block(SDF, o, n), cutcell_pressure=True)
    s.enable_vof()
    s.set_vof(slice_block(C0, o, n))
    s.set_property_model("rho", "linear", "C", [rho_g, rho_l - rho_g])
    s.set_property_model("mu", "linear", "C", [mu_g, mu_l - mu_g])
    s.set_surface_tension(sigma)
    s.set_contact_angle(theta)
    s.enable_vof_momentum(rho_g, rho_l)
    s.set_property_model("force_z", "linear", "C", [-rho_g * g, -(rho_l - rho_g) * g])
    s.set_pressure_chebyshev(True, 600, 1e-12)
    s.set_dt(0.4 * s.capillary_dt())
    return s, 600, f"packed {nx}x{nx}x{gnz}"


BUILD = {"hysing1": build_hysing1, "packed": build_packed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="hysing1")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warm", type=int, default=5)
    a = ap.parse_args()
    s, cap, label = BUILD[a.case]()
    rank, size = s.rank(), s.size()
    pick = vp.dt_from_limits(0.4)
    for i in range(a.warm):
        pick(s, i)
        s.step()
    tm = vp.profile(s, a.steps, pick, cap, f"{label} np={size}")
    # C sum is a global reduction the solver already does; use it as the run's sanity read-out
    d = s.vof_diagnostics()
    if rank == 0:
        n = max(tm["steps"], 1)
        st = tm["step"] / n
        ex = tm["k_exchange"] / n
        print(f"\n=== {tm['label']}  (rank 0 of {size}) ===")
        print(f"  {tm['ms_per_step']:8.2f} ms/step wall   {1e3*st:8.2f} ms/step timed   "
              f"pressure {tm['press_max']}/{tm['press_cap']}"
              + ("  CAPPED" if tm["capped"] else ""))
        print(f"  g=3 colour exchange      {1e3*ex:9.3f} ms/step  "
              f"{100.0*ex/st if st else 0:6.2f} % of the step")
        for k in ("vof_advect", "vof_momentum_advect", "vof_bridge", "vof_momentum_bridge",
                  "curvature", "csf", "k_reconstruct", "k_fluxes", "k_sweep", "k_clip",
                  "predictor", "momentum_solve", "projection", "step"):
            print(f"  {k:<24s} {1e3*tm[k]/n:9.3f} ms/step  "
                  f"{100.0*(tm[k]/n)/st if st else 0:6.2f} %")
        print(f"  sum(C) = {d['sum']!r}   dt census cfl {tm['binds_cfl']} / capillary "
              f"{tm['binds_capillary']}")


if __name__ == "__main__":
    main()
