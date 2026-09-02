#!/usr/bin/env python3
"""WO-P23 rung P3 — SCRIVEN bubble growth (Scriven, Chem. Eng. Sci. 10:1, 1959).

A spherical vapour bubble growing in uniformly superheated liquid, thermally controlled:

    R(t) = 2 beta sqrt(alpha_l t) ,
    Ja   = rho_l c_pl (T_inf - T_sat) / (rho_v h_lv)
         = 2 beta^2 int_0^1 exp( -beta^2 [ (1-x)^-2 - 2(1 - rho_v/rho_l) x - 1 ] ) dx      (Scriven)
    T(r,t) = T_inf - (rho_v h_lv)/(rho_l c_pl) * 2 beta^2
             int_{1-R/r}^{1} exp( -beta^2 [ (1-x)^-2 - 2(1 - rho_v/rho_l) x - 1 ] ) dx .

The integral is tabulated numerically here (Gauss-Legendre on a substitution that resolves the
essential singularity at x -> 1); `beta` follows by bisection.

WHAT THE RUN MEASURES.  R(t) from the LIQUID VOLUME DEFICIT, R = (3 sum(1-C) / 4pi)^(1/3) — a
conserved-quantity read-out, not a contour, so it does not depend on any interface post-processing.
The gate is |R_num - R_exact| / R_exact over the last half of the run.

BOUNDARIES.  Outflow on all six faces (the net vapour production is a real volume flux and has to
leave; with pure liquid at every face and the reference density set to rho_l the boundary
coefficient `applyBoundaryOpenness` re-imposes IS the variable-density one, so WO-R2 item 1 is
side-stepped by construction) and T = T_inf Dirichlet on all six.

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_scriven.py [--n 128] [--ja 0.5,2,10]
"""
import argparse
import math
import sys

import numpy as np

import peclet.flow as pf

_GL = np.polynomial.legendre.leggauss(400)


def _scriven_integral(beta, rr, lo=0.0):
    """int_lo^1 exp(-beta^2[(1-x)^-2 - 2(1-rr) x - 1]) dx, by the substitution x = 1 - 1/u
    (u in [1/(1-lo), inf)) folded onto a finite interval with u = 1/v."""
    # substitute x = 1 - v  (v in (0, 1-lo]) -> integrand exp(-b^2[v^-2 - 2(1-rr)(1-v) - 1]) dv
    a, w = _GL
    hi = 1.0 - lo
    v = 0.5 * hi * (a + 1.0)
    ww = 0.5 * hi * w
    v = np.clip(v, 1e-300, None)
    ex = -beta * beta * (1.0 / (v * v) - 2.0 * (1.0 - rr) * (1.0 - v) - 1.0)
    return float(np.sum(ww * np.exp(np.clip(ex, -700.0, 700.0))))


def scriven_beta(ja, rr):
    lo, hi = 1e-6, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 2.0 * mid * mid * _scriven_integral(mid, rr) > ja:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def scriven_T(r, R, beta, rr, dT):
    """T_inf - dT * (2 beta^2 / Ja) * int_{1-R/r}^1 (...)  == the profile above, normalised so that
    T(R) = T_inf - dT."""
    if r <= R:
        return 0.0  # T_sat, measured as T - T_sat
    full = 2.0 * beta * beta * _scriven_integral(beta, rr)
    part = 2.0 * beta * beta * _scriven_integral(beta, rr, lo=1.0 - R / r)
    return dT - dT * part / full


def sphere_colour(n, cx, cy, cz, R, sub=4):
    """Liquid fraction (C = 1 liquid, 0 vapour) of a sphere of radius R, by sub^3 subsampling."""
    off = (np.arange(sub) + 0.5) / sub
    ax = np.arange(n)
    c = np.ones((n, n, n), order="F")
    lo = [max(0, int(math.floor(q - R - 2))) for q in (cx, cy, cz)]
    hi = [min(n, int(math.ceil(q + R + 2))) for q in (cx, cy, cz)]
    xs = (ax[lo[0]:hi[0], None] + off[None, :]).ravel()
    ys = (ax[lo[1]:hi[1], None] + off[None, :]).ravel()
    zs = (ax[lo[2]:hi[2], None] + off[None, :]).ravel()
    d2 = ((xs - cx) ** 2)[:, None, None] + ((ys - cy) ** 2)[None, :, None] + \
         ((zs - cz) ** 2)[None, None, :]
    inside = (d2 < R * R).reshape(hi[0] - lo[0], sub, hi[1] - lo[1], sub, hi[2] - lo[2], sub)
    frac = inside.mean(axis=(1, 3, 5))
    c[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1.0 - frac
    return np.asfortranarray(c)


def run(n, ja, ratio, r0, r1, cfl=0.2, alpha_l=1.0, sweeps=200, plane=True, consistent=True,
        quad=True, muscl=False, verbose=True):
    rr = 1.0 / ratio
    beta = scriven_beta(ja, rr)
    t0 = (r0 / (2 * beta)) ** 2 / alpha_l          # cells^2 / (cells^2/s) = s
    te = (r1 / (2 * beta)) ** 2 / alpha_l
    rho_l, rho_v = 1.0, rr
    cpl = 1.0
    k_l = alpha_l * rho_l * cpl
    rcp_l, rcp_v = rho_l * cpl, rho_v * cpl
    k_v = k_l / ratio
    dT = 1.0
    h_lv = rho_l * cpl * dT / (rho_v * ja)
    ctr = 0.5 * n

    s = pf.Solver(n, n, n)
    s.set_rho(rho_l)
    s.set_mu(1e-3)
    for f in range(6):
        s.set_domain_bc(f, 3)              # outflow everywhere: the vapour production must leave
    s.set_pressure_geometry(np.full((n, n, n), 1.0, order="F"))
    s.enable_vof()
    s.set_vof(sphere_colour(n, ctr, ctr, ctr, r0))
    s.set_property_model("rho", "linear", "C", [rho_v, rho_l - rho_v])
    s.set_pressure_fcg(True, 600, 1e-10)

    ax = (np.arange(n) + 0.5) - ctr
    rad = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    tprof = np.vectorize(lambda r: scriven_T(r, r0, beta, rr, dT))(rad)
    s.add_scalar("T", k_l / rcp_l, 1, sweeps)
    for f in range(6):
        s.set_scalar_bc("T", f, 2, dT)
    s.set_field("T", np.asfortranarray(tprof))
    s.enable_phase_change(rho_v, rho_l, h_lv)
    s.set_phase_change_thermal("T", 0.0, k_v, k_l, 0.0)
    s.set_phase_change_plane_dirichlet(plane)
    if consistent:
        s.set_phase_change_energy(rcp_v, rcp_l)
        s.set_phase_change_energy_muscl(muscl)
    s.set_phase_change_quadratic_fit(quad)

    # ADAPTIVE dt on the solver's OWN interface-local Courant number. The a-priori estimate
    # u = (1 - rho_v/rho_l) Rdot is the CONTINUUM interface speed and the discrete field overshoots
    # it by up to ~2x on the first steps (the deposit shell is one cell thick and the per-cell PLIC
    # areas of a freshly sub-sampled sphere are uneven), so a predicted dt trips the Weymouth-Yue
    # cap. Predict, then correct from `vof_last_courant()`, and halve on a rejected step.
    rows = []
    tcur, nst, itmax, capped = t0, 0, 0, 0
    Rd = beta * math.sqrt(alpha_l / tcur)
    dt = 0.4 * cfl / max(Rd * (1.0 - rr), 1e-30)
    while tcur < te:
        dt = min(dt, te - tcur)
        while True:
            s.set_dt(dt)
            try:
                s.step()
                break
            except RuntimeError as ex:
                if "Weymouth-Yue boundedness cap" not in str(ex):
                    raise
                dt *= 0.5
        tcur += dt
        nst += 1
        c_ = s.vof_last_courant()
        dt *= min(1.3, max(0.5, cfl / max(c_, 1e-30)))
        it = s.last_pressure_iterations()
        itmax = max(itmax, it)
        capped += 1 if it >= 600 else 0
        vol = float(n) ** 3 - s.get_vof().sum()
        rnum = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
        rex = 2 * beta * math.sqrt(alpha_l * tcur)
        dg = s.phase_change_diagnostics()
        # exact mdot = rho_v Rdot = rho_v beta sqrt(alpha_l/t)
        mex = rho_v * beta * math.sqrt(alpha_l / tcur)
        rows.append((tcur, rnum, rex, it, dg['mdot_mean'], mex))
    d = s.phase_change_diagnostics()
    half = rows[len(rows) // 2:]
    errs = [abs(r[1] - r[2]) / r[2] for r in half]
    if verbose:
        print(f"  N = {n}^3  Ja = {ja:g}  ratio {ratio:g}  beta = {beta:.6f}  "
              f"R {r0:g} -> {r1:g} cells  steps {nst}")
        for k in list(range(0, len(rows), max(1, len(rows) // 8))) + [len(rows) - 1]:
            t_, a, b_, it, mn_, mx_ = rows[k]
            print(f"      t {t_:10.4f}  R_num {a:8.4f}  R_exact {b_:8.4f}  "
                  f"rel {100*(a-b_)/b_:+7.3f} %  mdot {mn_:.5e} vs {mx_:.5e} "
                  f"({100*(mn_-mx_)/mx_:+7.3f} %)  iters {it}")
        print(f"      max |dR|/R over the LAST HALF = {100*max(errs):.3f} %   [gate 1 %]")
        print(f"      pressure iters max {itmax}/600 (capped {capped}), band_div "
              f"{d['band_div']:.3e}, C in [{d['min_C']:.2e}, {d['max_C']:.6f}], "
              f"T in [{d['T_min']:.3e}, {d['T_max']:.6f}], fallback {d['fallback_cells']}, "
              f"unresolved {d['unresolved']:.1e}")
        if capped:
            print("      *** CAPPED pressure solve: rule 3b, this run is INVALID")
    return max(errs), capped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ja", type=str, default="0.5,2,10")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--r0", type=float, default=6.0)
    ap.add_argument("--r1", type=float, default=20.0)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--no-plane", action="store_true")
    ap.add_argument("--no-consistent", action="store_true")
    ap.add_argument("--no-quad", action="store_true")
    ap.add_argument("--muscl", action="store_true")
    a = ap.parse_args()
    print(f"P3 Scriven bubble growth, {a.n}^3, plane Dirichlet {not a.no_plane}, "
          f"consistent energy {not a.no_consistent}, quadratic fit {not a.no_quad}, "
          f"energy MUSCL {a.muscl}")
    for ja in [float(x) for x in a.ja.split(",")]:
        run(a.n, ja, a.ratio, a.r0, a.r1, sweeps=a.sweeps, plane=not a.no_plane,
            consistent=not a.no_consistent, quad=not a.no_quad, muscl=a.muscl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
