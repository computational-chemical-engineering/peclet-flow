#!/usr/bin/env python3
"""WO-V9 — the VoF performance profile: where a two-phase step actually goes.

Five cases, transcribed from the gallery drivers so the profile is of the code people run:

  hysing1   Hysing et al. (2009) case 1, quasi-2D 64 x 4 x 128   (examples/rising-bubble)
  packed    the E7 packed column, 64 x 64 x 160, bubble + bed    (examples/bubble-through-packing)
  trickle   the E6 trickle bed, 48 x 48 x 96, liquid distributor (examples/trickle-flow-packing)
  droplet   a 128^3 static droplet                              (examples/parasitic-currents)
  scriven   a 96^3 Scriven bubble, phase change                 (tests/study/vof_scriven.py)

Each case runs `--warm` steps to fill caches and settle the adaptive dt, then `--steps` TIMED
steps with `set_vof_timing(True)`.  Reported: the step's three coarse phases (predictor / momentum
solve / projection), the VoF stages (colour advection with its bridges split out, curvature, CSF,
phase change), the advector's own kernels (reconstruct / fluxes / sweeps / clip / g=3 exchange),
ms/step and the dt census (which of the two explicit limits binds).

Rule 3b: every case records max pressure iterations against its cap; a capped solve is flagged.

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_profile.py --case all
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_profile.py --case hysing1 --worklist off
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import peclet.flow as flow

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("VOF_PROFILE_CACHE", os.path.join(HERE, ".vof_profile_cache"))


# --------------------------------------------------------------------------- geometry helpers
def cylinder_fractions(shape, R, cx, cz, sub=32):
    """Volume fraction of a cylinder with its axis along y (a 2-D disc), exact in z."""
    nx, ny, nz = shape
    ax = (np.arange(nx)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub).ravel()
    half = np.sqrt(np.maximum(R * R - (ax - cx) ** 2, 0.0))
    z0, z1 = cz - half, cz + half
    C = np.zeros((nx, ny, nz))
    for k in range(nz):
        seg = np.maximum(np.minimum(z1, k + 1) - np.maximum(z0, k), 0.0)
        C[:, :, k] = seg.reshape(nx, sub).mean(axis=1)[:, None]
    return np.asfortranarray(C)


def sphere_fractions(shape, R, c, sub=24):
    """Liquid volume fraction of a sphere: exact in z, sub x sub subsampled in (x,y)."""
    nx, ny, nz = shape
    ax = (np.arange(nx)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub).ravel()
    ay = (np.arange(ny)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub).ravel()
    X, Y = ax[:, None], ay[None, :]
    half = np.sqrt(np.maximum(R * R - (X - c[0]) ** 2 - (Y - c[1]) ** 2, 0.0))
    z0, z1 = c[2] - half, c[2] + half
    C = np.zeros((nx, ny, nz))
    for k in range(nz):
        seg = np.maximum(np.minimum(z1, k + 1) - np.maximum(z0, k), 0.0)
        C[:, :, k] = seg.reshape(nx, sub, ny, sub).mean(axis=(1, 3))
    return np.asfortranarray(C)


def bubble_colour_sub(nx, nz, R, cz, sub=4):
    """C = liquid fraction; the gas bubble is the C = 0 sphere at (nx/2, nx/2, cz)."""
    q = (np.arange(sub) + 0.5) / sub
    C = np.zeros((nx, nx, nz))
    c = nx / 2.0
    for k in range(nz):
        pz = k + q
        acc = np.zeros((nx, nx))
        for a in q:
            px = (np.arange(nx) + a)[:, None]
            for b in q:
                py = (np.arange(nx) + b)[None, :]
                r2 = (px - c) ** 2 + (py - c) ** 2
                acc += ((r2[:, :, None] + (pz - cz) ** 2) < R * R).sum(axis=2)
        C[:, :, k] = 1.0 - acc / sub ** 3
    return np.asfortranarray(C)


def bed_sdf(centres, nx, nz, Rg, periodic_xy=True):
    """Signed distance to a union of spheres, positive in the fluid; periodic in x and y."""
    W = float(nx)
    g, gz = np.arange(nx) + 0.5, np.arange(nz) + 0.5
    X, Y, Z = np.meshgrid(g, g, gz, indexing="ij")
    phi = np.full((nx, nx, nz), 1e30)
    for k in range(len(centres)):
        dx = X - centres[k, 0]
        dy = Y - centres[k, 1]
        if periodic_xy:
            dx -= W * np.round(dx / W)
            dy -= W * np.round(dy / W)
        dz = Z - centres[k, 2]
        phi = np.minimum(phi, np.sqrt(dx * dx + dy * dy + dz * dz) - Rg)
    return np.asfortranarray(phi)


def rsa_bed(n, W, R, z0, seed, gap=0.02):
    """A deterministic random-sequential-addition sphere bed in a laterally periodic column.

    A stand-in for the gallery pages' DEM-settled packings: same grain radius, same column, same
    number of grains, and no `dem` import (which would put two Kokkos backends in one process).
    The PROFILE does not depend on which loose packing it is -- only on the cut-cell count, which
    is set by the grain count and radius.
    """
    rng = np.random.default_rng(seed)
    pts = []
    z = z0 + R
    tries = 0
    while len(pts) < n and tries < 400000:
        tries += 1
        p = np.array([rng.uniform(0, W), rng.uniform(0, W), rng.uniform(z0 + R, z0 + R + 3.6 * R)])
        ok = True
        for q in pts:
            d = p - q
            d[0] -= W * round(d[0] / W)
            d[1] -= W * round(d[1] / W)
            if d @ d < (2 * R * (1 + gap)) ** 2:
                ok = False
                break
        if ok:
            pts.append(p)
    return np.array(pts)


def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, name + ".npy")
    if os.path.exists(f):
        return np.load(f)
    a = fn()
    np.save(f, a)
    return a


# --------------------------------------------------------------------------- the timing harness
_LAST_DT = {"dt": 0.0}


def set_dt(s, dt):
    """Every picker goes through this so the harness can halve a rejected step's dt."""
    _LAST_DT["dt"] = dt
    s.set_dt(dt)
def profile(s, nsteps, pick_dt, cap, label, extra=None):
    """`pick_dt(s, i)` sets dt for step i and returns which limit bound it ('cfl'|'cap'|'fixed')."""
    s.set_vof_timing(True)
    binds = {"cfl": 0, "cap": 0, "fixed": 0}
    itmax, capped = 0, 0
    t0 = time.time()
    retries = 0
    for i in range(nsteps):
        binds[pick_dt(s, i)] += 1
        while True:  # the WY cap is a hard throw, not a warning (vof_scriven's own pattern)
            try:
                s.step()
                break
            except RuntimeError as ex:
                if "Weymouth-Yue boundedness cap" not in str(ex) or retries > 20:
                    raise
                retries += 1
                set_dt(s, 0.5 * _LAST_DT["dt"])
        it = s.last_pressure_iterations()
        itmax = max(itmax, it)
        capped += int(it >= cap)
    wall = time.time() - t0
    tm = dict(s.vof_timing())
    s.set_vof_timing(False)
    tm["label"] = label
    tm["wall"] = wall
    tm["ms_per_step"] = 1e3 * wall / nsteps
    tm["press_max"] = itmax
    tm["press_cap"] = cap
    tm["capped"] = capped
    tm["binds_cfl"] = binds["cfl"]
    tm["binds_capillary"] = binds["cap"]
    tm["binds_fixed"] = binds["fixed"]
    tm["dt_retries"] = retries
    if extra:
        tm.update(extra)
    return tm


def dt_from_limits(frac=0.4):
    def pick(s, i):
        L = s.vof_step_limits()
        dt = frac * min(L["cfl_dt"], L["capillary_dt"])
        set_dt(s, dt)
        return "cap" if L["capillary_binds"] else "cfl"
    return pick


# --------------------------------------------------------------------------- the five cases
class Scale:
    """physical -> solver units, the gallery's Scale (cell size 1, time in seconds)."""

    def __init__(self, s):
        self.s = s

    def mu(self, m):
        return self.s ** 2 * m

    def sigma(self, sg):
        return self.s ** 3 * sg

    def bodyforce(self, f):
        return self.s * f


HYSING1 = dict(rho1=1000.0, rho2=100.0, mu1=10.0, mu2=1.0, g=0.98, sigma=24.5)


def case_hysing1(nx=64, momentum=True, worklist=True):
    p = HYSING1
    nz, ny = 2 * nx, 4
    sc = Scale(nx / 1.0)
    R = sc.s * 0.25
    s = flow.Solver(nx, ny, nz)
    s.set_rho(p["rho1"])
    s.set_mu(sc.mu(p["mu1"]))
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)
    s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 600, 1e-12)
    s.enable_vof()
    s.set_vof_worklist(worklist)
    s.set_vof(np.asfortranarray(
        1.0 - cylinder_fractions((nx, ny, nz), R, nx / 2.0, sc.s * 0.5)))
    s.set_property_model("rho", "linear", "C", [p["rho2"], p["rho1"] - p["rho2"]])
    s.set_property_model("mu", "linear", "C",
                         [sc.mu(p["mu2"]), sc.mu(p["mu1"] - p["mu2"])])
    s.set_surface_tension(sc.sigma(p["sigma"]))
    if momentum:
        s.enable_vof_momentum(p["rho2"], p["rho1"])
    s.set_property_model("force_z", "linear", "C",
                         [-sc.bodyforce(p["rho2"] * p["g"]),
                          -sc.bodyforce((p["rho1"] - p["rho2"]) * p["g"])])
    s.set_dt(0.5 * s.capillary_dt())
    return s, dt_from_limits(0.4), 600, f"hysing1 {nx}x{ny}x{nz}"


def case_packed(nx=64, nz=160, worklist=True):
    """E7: a gas bubble released under a sphere bed in a closed, laterally periodic column."""
    NG, RJ, FSDF = 26, 0.171875 * nx, 0.85
    RG = FSDF * RJ
    ZBED, RBUB, ZBUB = 0.5625 * nx, 0.15625 * nx, 0.28125 * nx
    ctr = cached(f"bed_e7_{nx}_{nz}", lambda: rsa_bed(NG, float(nx), RJ, ZBED, seed=11))
    SDF = cached(f"sdf_e7_{nx}_{nz}", lambda: bed_sdf(ctr, nx, nz, RG))
    ratio, Eo, Mo, g, theta = 100.0, 10.0, 1e-3, 2e-4, 60.0
    D = 2 * RBUB
    rho_l, rho_g = 1.0, 1.0 / ratio
    drho = rho_l - rho_g
    sigma = drho * g * D * D / Eo
    mu_l = (Mo * rho_l ** 2 * sigma ** 3 / (g * drho)) ** 0.25
    mu_g = mu_l / ratio
    s = flow.Solver(nx, nx, nz)
    s.set_rho(rho_l)
    s.set_mu(mu_l)
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)
    s.set_solid(np.asfortranarray(SDF), cutcell_pressure=True)
    s.enable_vof()
    s.set_vof_worklist(worklist)
    s.set_vof(cached(f"c0_e7_{nx}_{nz}", lambda: bubble_colour_sub(nx, nz, RBUB, ZBUB)))
    s.set_property_model("rho", "linear", "C", [rho_g, rho_l - rho_g])
    s.set_property_model("mu", "linear", "C", [mu_g, mu_l - mu_g])
    s.set_surface_tension(sigma)
    s.set_contact_angle(theta)
    s.enable_vof_momentum(rho_g, rho_l)
    s.set_property_model("force_z", "linear", "C", [-rho_g * g, -(rho_l - rho_g) * g])
    s.set_pressure_chebyshev(True, 600, 1e-12)
    s.set_dt(0.4 * s.capillary_dt())
    return s, dt_from_limits(0.4), 600, f"packed {nx}x{nx}x{nz}"


def case_trickle(nx=48, nz=96, worklist=True):
    """E6: a liquid distributor over a dry sphere bed, outflow at the bottom."""
    RHO_L, MU_L, MU_RATIO, BO, THETA = 100.0, 50.0, 50.0, 1.2, 60.0
    R48, ZBED0, NG = 7.7, 22.0, 34
    sc = nx / 48.0
    R = R48 * sc
    ctr = cached(f"bed_e6_{nx}_{nz}", lambda: rsa_bed(NG, float(nx), R, ZBED0 * (nz / 96.0), 7))
    SDF = cached(f"sdf_e6_{nx}_{nz}", lambda: bed_sdf(ctr, nx, nz, R))
    ratio = 100.0
    rho_g, mu_g = RHO_L / ratio, MU_L / MU_RATIO
    drho = RHO_L - rho_g
    delta, umax = 3.0 * sc, 0.3 * sc
    g = 2 * MU_L * umax / (drho * delta ** 2)
    dp = 2 * R
    sigma = drho * g * dp ** 2 / BO
    rd = 8.0 * sc
    s = flow.Solver(nx, nx, nz)
    s.set_rho(RHO_L)
    s.set_mu(MU_L)
    for f in range(4):
        s.set_domain_bc(f, 0, 0, 0, 0)
    s.set_domain_bc(4, 3, 0, 0, 0)
    xc = (np.arange(nx) + 0.5)[:, None]
    yc = (np.arange(nx) + 0.5)[None, :]
    disc = ((xc - nx / 2) ** 2 + (yc - nx / 2) ** 2) < rd ** 2
    prof = np.zeros((nx, nx, 3))
    prof[:, :, 2] = np.where(disc, -umax, 0.0)
    s.set_domain_bc_profile(5, np.ascontiguousarray(prof))
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=6)
    s.set_pressure_solver_params(80)
    s.set_solid(np.asfortranarray(SDF), cutcell_pressure=True)
    s.enable_vof()
    s.set_vof_worklist(worklist)
    s.set_vof(np.zeros((nx, nx, nz), order="F"))
    s.set_surface_tension(sigma)
    s.set_contact_angle(THETA)
    s.set_property_model("rho", "linear", "C", [rho_g, RHO_L - rho_g])
    s.set_property_model("mu", "linear", "C", [mu_g, MU_L - mu_g])
    s.set_property_model("force_z", "linear", "C", [0.0, -drho * g])
    s.enable_vof_momentum(rho_g, RHO_L)
    s.set_pressure_fcg(True, 400, 1e-11)
    s.set_vof_inflow_profile(5, np.ascontiguousarray(disc.astype(float)))
    s.set_vof_backflow(4, 0.0)
    dt_cap = 0.5 * s.capillary_dt()
    s.set_dt(dt_cap)

    def pick(sv, i):
        L = sv.vof_step_limits()
        dt = min(dt_cap, 0.4 * L["cfl_dt"]) if L["cfl_dt"] > 0 else dt_cap
        set_dt(sv, dt)
        return "cap" if dt >= dt_cap - 1e-30 else "cfl"

    return s, pick, 400, f"trickle {nx}x{nx}x{nz}"


def case_droplet(n=128, worklist=True):
    """A stationary droplet, D/dx = n/2, in an n^3 periodic box (the parasitic-currents case)."""
    MU, SIGMA = 0.1, 1.0
    R = n / 4.0
    off = (0.13, 0.27, 0.11)
    s = flow.Solver(n, n, n)
    s.set_rho(1.0)
    s.set_mu(MU)
    s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 500, 1e-14)
    s.enable_vof()
    s.set_vof_worklist(worklist)
    s.set_vof(cached(f"drop_{n}", lambda: sphere_fractions(
        (n, n, n), R, (n / 2 + off[0], n / 2 + off[1], n / 2 + off[2]))))
    s.set_property_model("rho", "linear", "C", [1.0, 0.0])
    s.set_surface_tension(SIGMA)
    dt = 0.5 * s.capillary_dt()
    s.set_dt(dt)

    def pick(sv, i):
        set_dt(sv, dt)
        return "fixed"

    return s, pick, 500, f"droplet {n}^3"


def case_scriven(n=96, ja=0.5, ratio=10.0, r0=6.0, worklist=True):
    """WO-P23 rung P3: a Scriven bubble growing in superheated liquid (the phase-change pipeline)."""
    sys.path.insert(0, HERE)
    import vof_scriven as vs

    rr = 1.0 / ratio
    beta = vs.scriven_beta(ja, rr)
    t0 = (r0 / (2 * beta)) ** 2
    rho_l, rho_v, cpl = 1.0, rr, 1.0
    alpha_l = 1.0
    k_l = alpha_l * rho_l * cpl
    k_v = k_l / ratio
    dT = 1.0
    h_lv = rho_l * cpl * dT / (rho_v * ja)
    ctr = 0.5 * n
    s = flow.Solver(n, n, n)
    s.set_rho(rho_l)
    s.set_mu(1e-3)
    for f in range(6):
        s.set_domain_bc(f, 3)
    s.set_pressure_geometry(np.full((n, n, n), 1.0, order="F"))
    s.enable_vof()
    s.set_vof_worklist(worklist)
    s.set_vof(cached(f"scriven_c_{n}_{r0}", lambda: vs.sphere_colour_chunked(n, ctr, r0, 4)))
    s.set_property_model("rho", "linear", "C", [rho_v, rho_l - rho_v])
    s.set_pressure_fcg(True, 600, 1e-10)
    s.add_scalar("T", k_l / (rho_l * cpl), 1, 200)
    for f in range(6):
        s.set_scalar_bc("T", f, 2, dT)
    s.set_field("T", np.asfortranarray(cached(
        f"scriven_T_{n}_{r0}_{ja}",
        lambda: vs.initial_temperature(n, ctr, r0, beta, rr, dT, mode="similarity"))))
    s.enable_phase_change(rho_v, rho_l, h_lv)
    s.set_phase_change_thermal("T", 0.0, k_v, k_l, 0.0)
    s.set_phase_change_plane_dirichlet(True)
    s.set_phase_change_energy(rho_v * cpl, rho_l * cpl)
    s.set_phase_change_quadratic_fit(True)
    Rd = beta * math.sqrt(alpha_l / t0)
    dt = 0.4 * 0.2 / max(Rd * (1.0 - rr), 1e-30)
    s.set_dt(dt)

    # vof_scriven's own adaptive rule: predict, then correct from the LAST realised Courant
    # number (the a-priori continuum interface speed under-predicts the discrete field by ~2x on
    # the first steps, and vof_step_limits' cfl_dt is not defined before the first advection).
    st = {"dt": 0.25 * dt}

    def pick(sv, i):
        if i > 0:
            c = sv.vof_last_courant()
            st["dt"] *= min(1.2, max(0.5, 0.2 / max(c, 1e-30)))
        set_dt(sv, st["dt"])
        return "cfl"

    return s, pick, 600, f"scriven {n}^3 Ja={ja}"


CASES = {
    "hysing1": case_hysing1,
    "packed": case_packed,
    "trickle": case_trickle,
    "droplet": case_droplet,
    "scriven": case_scriven,
}


def show(tm):
    n = max(tm["steps"], 1)
    st = tm["step"] / n
    def pc(x):
        return 100.0 * (x / n) / st if st > 0 else 0.0
    vof = (tm["vof_advect"] + tm["vof_momentum_advect"] + tm["curvature"] + tm["csf"]
           + tm["phase_change"])
    print(f"\n=== {tm['label']} ===")
    print(f"  steps {n}   {tm['ms_per_step']:8.2f} ms/step (wall)   "
          f"{1e3*st:8.2f} ms/step (timed)   pressure {tm['press_max']}/{tm['press_cap']}"
          + ("  *** CAPPED ***" if tm["capped"] else ""))
    print(f"  dt census: CFL binds {tm['binds_cfl']}, capillary binds {tm['binds_capillary']}, "
          f"fixed {tm['binds_fixed']}")
    rows = [
        ("VoF total", vof),
        ("  colour advect", tm["vof_advect"]),
        ("    of which bridges", tm["vof_bridge"]),
        ("  momentum advect", tm["vof_momentum_advect"]),
        ("    of which bridges", tm["vof_momentum_bridge"]),
        ("  curvature", tm["curvature"]),
        ("  CSF", tm["csf"]),
        ("  phase change", tm["phase_change"]),
        ("kernels: reconstruct", tm["k_reconstruct"]),
        ("kernels: fluxes", tm["k_fluxes"]),
        ("kernels: sweeps", tm["k_sweep"]),
        ("kernels: clip", tm["k_clip"]),
        ("kernels: g=3 exchange", tm["k_exchange"]),
        ("kernels: freeze", tm["k_freeze"]),
        ("curv: compact", tm["kc_compact"]),
        ("curv: planes", tm["kc_planes"]),
        ("curv: height", tm["kc_height"]),
        ("curv: fallback", tm["kc_fallback"]),
        ("curv: census", tm["kc_census"]),
        ("predictor", tm["predictor"]),
        ("momentum solve", tm["momentum_solve"]),
        ("projection", tm["projection"]),
        ("step (total)", tm["step"]),
    ]
    for name, v in rows:
        print(f"  {name:<28s} {1e3*v/n:9.3f} ms/step  {pc(v):6.2f} %")
    other = tm["step"] - (vof + tm["predictor"] + tm["momentum_solve"] + tm["projection"])
    print(f"  {'remainder':<28s} {1e3*other/n:9.3f} ms/step  {pc(other):6.2f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--worklist", choices=("on", "off", "both"), default="on")
    ap.add_argument("--curv-worklist", choices=("on", "off", "both"), default="on")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    names = list(CASES) if a.case == "all" else a.case.split(",")
    wls = [True, False] if a.worklist == "both" else [a.worklist == "on"]
    cwls = [True, False] if a.curv_worklist == "both" else [a.curv_worklist == "on"]
    out = []
    for nm in names:
      for cwl in cwls:
        for wl in wls:
            s, pick, cap, label = CASES[nm](worklist=wl)
            s.set_vof_curvature_worklist(cwl)
            label += "" if cwl else "  [curv worklist OFF]"
            for i in range(a.warm):
                pick(s, i)
                s.step()
            tm = profile(s, a.steps, pick, cap,
                         label + ("" if wl else "  [worklist OFF]"),
                         extra={"case": nm, "worklist": wl, "curv_worklist": cwl})
            show(tm)
            out.append(tm)
            del s
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
