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

INITIALISATION (`--init`, WO-P3b).  Scriven's solution carries a thermal boundary layer of
thickness ~R0/(2 beta^2) at t0, and the classic trap in this benchmark is to start from a UNIFORM
superheat with a sharp bubble, so that the first steps evaporate the superheat sitting AT the
interface and the radius keeps a permanent offset.  `--init similarity` (the DEFAULT, and what
this driver has always done) samples T(r, t0) from the profile above at the cell centres, with
R0 = 2 beta sqrt(alpha_l t0) by construction (t0 is derived from R0);  `--init cellavg` is the
same profile CELL-AVERAGED by 4^3 subsampling;  `--init uniform` is the trap itself, kept as the
CONTROL that shows what it costs.

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


def initial_temperature(n, ctr, R, beta, rr, dT, mode="similarity", sub=4):
    """T(r, t0) - T_sat on the cell centres, for the three initialisations of `--init`.

    'similarity' : Scriven's similarity profile sampled AT the cell centre.  This is what the
                   driver has always done and what Sato & Niceno (JCP 2013) and Malan et al.
                   (2021) do; it is the shipped default and this branch is byte-identical to the
                   pre-WO-P3b code.
    'cellavg'    : the same profile, CELL-AVERAGED by sub^3 subsampling (a finite-volume initial
                   state rather than a point sample -- the O(h^2 T'') half of the same question).
    'uniform'    : uniform superheat T_inf in the liquid, T_sat inside the bubble, i.e. NO thermal
                   boundary layer at t0.  The classic trap, kept as the control.
    """
    ax = (np.arange(n) + 0.5) - ctr
    rad = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    if mode == "similarity":
        return np.asfortranarray(np.vectorize(lambda r: scriven_T(r, R, beta, rr, dT))(rad))
    if mode == "uniform":
        return np.asfortranarray(np.where(rad > R, dT, 0.0))
    if mode != "cellavg":
        raise ValueError(f"unknown --init mode {mode!r}")
    # cell average: a fine RADIAL table (the profile is a function of r alone) + sub^3 subsampling,
    # done one cell-plane at a time so the 512^3 sample cloud is never materialised.
    rmax = math.sqrt(3.0) * n
    rt = np.linspace(0.0, rmax, 200001)
    tt = np.array([scriven_T(r, R, beta, rr, dT) for r in rt])
    off = (np.arange(sub) + 0.5) / sub
    qs = (np.arange(n)[:, None] + off[None, :]).ravel() - ctr
    out = np.empty((n, n, n))
    for i in range(n):
        xi = qs[i * sub:(i + 1) * sub]
        d2 = xi[:, None, None] ** 2 + qs[None, :, None] ** 2 + qs[None, None, :] ** 2
        t = np.interp(np.sqrt(d2), rt, tt)
        out[i] = t.reshape(sub, n, sub, n, sub).mean(axis=(0, 2, 4))
    return np.asfortranarray(out)


def sphere_colour_chunked(n, ctr, R, sub):
    """Liquid fraction of a sphere by sub^3 subsampling, one cell-plane at a time.

    WO-P3c: the AREA probe needs a colour field whose SLIVER cells survive.  `sub^3` subsampling
    quantizes C to multiples of 1/sub^3, so every cell whose true liquid fraction is below
    1/(2 sub^3) is rounded to exactly 0 or 1 and DROPS OUT of the interface entirely — at sub = 4
    that is a quarter of the interfacial cells of a sphere and 5-9 % of its area, while the VOLUME
    (and hence R0) moves by 1e-4 %.  That is the whole of WO-P3b's "the PLIC area of a sphere is
    5-9 % low": see the WO-P3c findings.  The chunked loop is what makes sub = 32 affordable
    (sub = 32 on R = 28 would otherwise materialise a 2000^3 sample cloud).
    """
    off = (np.arange(sub) + 0.5) / sub
    lo = max(0, int(math.floor(ctr - R - 2)))
    hi = min(n, int(math.ceil(ctr + R + 2)))
    q = (np.arange(lo, hi)[:, None] + off[None, :]).ravel() - ctr
    c = np.ones((n, n, n), order="F")
    m = hi - lo
    for i in range(m):
        xi = q[i * sub:(i + 1) * sub]
        d2 = xi[:, None, None] ** 2 + q[None, :, None] ** 2 + q[None, None, :] ** 2
        frac = (d2 < R * R).reshape(sub, m, sub, m, sub).mean(axis=(0, 2, 4))
        c[lo + i, lo:hi, lo:hi] = 1.0 - frac
    return np.asfortranarray(c)


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
        quad=True, muscl=False, init="similarity", sub=4, area_mode=None, verbose=True,
        budget=0):
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
    c0 = sphere_colour_chunked(n, ctr, r0, sub)
    s.set_vof(c0)
    # R0 CONSISTENCY (WO-P3b): the gate's read-out is the liquid volume deficit, so the number
    # that has to match 2 beta sqrt(alpha_l t0) is the one the SUB-SAMPLED colour field carries,
    # not the nominal r0.  Reported, not corrected.
    r0meas = (3.0 * (float(n) ** 3 - c0.sum()) / (4.0 * math.pi)) ** (1.0 / 3.0)
    s.set_property_model("rho", "linear", "C", [rho_v, rho_l - rho_v])
    s.set_pressure_fcg(True, 600, 1e-10)

    tprof = initial_temperature(n, ctr, r0, beta, rr, dT, mode=init)
    s.add_scalar("T", k_l / rcp_l, 1, sweeps)
    for f in range(6):
        s.set_scalar_bc("T", f, 2, dT)
    s.set_field("T", np.asfortranarray(tprof))
    s.enable_phase_change(rho_v, rho_l, h_lv)
    # Ablations kept because they are the findings' evidence. enable_phase_change turns WO-R2's
    # wisp guard OFF (the two are incompatible on a curved interface); PECLET_P23_WISP=1 puts it
    # back, which is how the 48 % / dt-collapse row of the findings is reproduced.
    import os
    if os.environ.get("PECLET_P23_WISP"):
        s.set_vof_wisp_eps(float(os.environ["PECLET_P23_WISP"]))
    if os.environ.get("PECLET_P23_NO_EXACTRES"):
        s.set_pressure_exact_residual(False)
    if os.environ.get("PECLET_P23_NO_OUTFLOWRHO"):
        s.set_outflow_rho_correction(False)
    s.set_phase_change_thermal("T", 0.0, k_v, k_l, 0.0)
    s.set_phase_change_plane_dirichlet(plane)
    if consistent:
        s.set_phase_change_energy(rcp_v, rcp_l)
        s.set_phase_change_energy_muscl(muscl)
    s.set_phase_change_quadratic_fit(quad)
    if area_mode is not None:
        s.set_phase_change_area(area_mode)
    if budget:
        # WO-P3f instrument (a): the energy budget of the energy solve, printed every `budget`
        # steps.  Off by default and inert in the solver when off.
        s.set_phase_change_budget(True)

    # ADAPTIVE dt on the solver's OWN interface-local Courant number. The a-priori estimate
    # u = (1 - rho_v/rho_l) Rdot is the CONTINUUM interface speed and the discrete field overshoots
    # it by up to ~2x on the first steps (the deposit shell is one cell thick and the per-cell PLIC
    # areas of a freshly sub-sampled sphere are uneven), so a predicted dt trips the Weymouth-Yue
    # cap. Predict, then correct from `vof_last_courant()`, and halve on a rejected step.
    rows = []
    tcur, nst, itmax, capped = t0, 0, 0, 0
    Rd = beta * math.sqrt(alpha_l / tcur)
    dt0 = 0.4 * cfl / max(Rd * (1.0 - rr), 1e-30)
    dt = dt0
    # dt-COLLAPSE GUARD. Without it a run whose interface velocity runs away halves dt forever and
    # the outer loop never terminates (it cost this campaign an 8-hour silent hang). A step that
    # needs less than 1e-4 of the initial dt is not a slow run, it is a diverged one; say so.
    dtmin = 1e-4 * dt0
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
                if dt < dtmin:
                    raise RuntimeError(
                        f"vof_scriven: dt COLLAPSED below {dtmin:.3e} (initial {dt0:.3e}) at step "
                        f"{nst}, t = {tcur:.4f} of {te:.4f}, last CFL {s.vof_last_courant():.4g} — "
                        f"the interface velocity is running away, not the time step being small")
        dt_used = dt
        tcur += dt
        nst += 1
        if verbose and nst % 200 == 0:
            print(f"      [step {nst}: t {tcur:.4f}/{te:.4f}, dt {dt:.4e}, "
                  f"cfl {s.vof_last_courant():.4f}]", flush=True)
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
        # WHERE the growth deficit sits (WO-P3b).  `mdot_mean` is an unweighted mean over the
        # interfacial CELLS, so it does not measure the integrated flux.  The integral is
        # `removed_volume` = sum mdot A dt / rho_l, and dividing it by the total PLIC area gives
        # the AREA-AVERAGED mdot; comparing the area itself against 4 pi R^2 separates a flux
        # error from a geometry error.
        area = dg['interface_area']
        # WO-P3e.  `dg['interface_area']` is the area `pcBuildInterface` measured at the HEAD of
        # this step, i.e. on the colour field BEFORE this step's regression and advection, while
        # `rnum` below is read AFTER them -- so `area/(4 pi rnum^2)` compares two different times
        # and is low by 2 dR/R, which at this scene's ~0.2 cells/step is 2.2 %.  That is the whole
        # of the "-2.15 % run area deficit" of WO-P3c/WO-P3d.  `vof_interface_area()` recomputes on
        # the CURRENT field, so `area_end` is the one that may be compared with `rnum`.
        area_end = s.vof_interface_area()
        meff = rho_l * dg['removed_volume'] / (dt_used * area) if area > 0 else float("nan")
        arel = area / (4.0 * math.pi * rnum * rnum) - 1.0
        arel_end = area_end / (4.0 * math.pi * rnum * rnum) - 1.0
        # WO-P3e: the REGRESSION's own step size.  `delta = mdot dt / rho_l` is the normal
        # displacement the plane shift applies, and it is the number that decides whether the
        # shift's O(delta/R) linearization can matter at all; `R_A = sqrt(A/4pi)` is the radius
        # the interface SHEET carries, next to the radius the liquid-volume deficit carries.
        delta = dg['removed_volume'] / area if area > 0 else float('nan')
        r_a = math.sqrt(area / (4.0 * math.pi))
        if budget and (nst % budget == 0 or nst == 1) and verbose:
            b = s.phase_change_budget()
            # E_lat: the latent heat the REGRESSION booked this step (W).  removed_volume is
            # sum mdot A dt / rho_l, so rho_l removed_volume h_lv / dt = sum mdot A h_lv.
            e_lat = rho_l * dg['removed_volume'] * h_lv / dt_used
            q = b['q_gfm']                       # W, < 0 = heat drawn OUT of the liquid
            dcls = (b['e_enter'] - b['e_leave']) / dt_used      # W lost at the class changes
            dovw = b['d_overwrite'] / dt_used                   # W injected by the Dirichlet rows
            dovn = b['d_overwrite_new'] / dt_used
            dH = (b['h_open_new'] - b['h_open']) / dt_used
            print(f"      [BUDGET step {nst:4d} t {tcur:9.4f} dt {dt_used:.3e}]  "
                  f"E_lat {e_lat:12.5e} W   q_gfm {q:12.5e} W  (-q_gfm/E_lat "
                  f"{-q/e_lat if e_lat else float('nan'):8.5f})")
            print(f"                     H_liq {b['h_liquid']:12.5e}  H_open {b['h_open']:12.5e}  "
                  f"dH_open/dt {dH:12.5e} W  |  class change: enter {b['e_enter']:11.4e} J "
                  f"leave {b['e_leave']:11.4e} J  net {dcls:12.5e} W ({100*dcls/e_lat if e_lat else float('nan'):+7.3f} % of E_lat)")
            print(f"                     overwrite {dovw:12.5e} W (new cells {dovn:12.5e} W, "
                  f"{100*dovn/e_lat if e_lat else float('nan'):+7.3f} % of E_lat)  |  "
                  f"L->I {b['n_enter_liquid']:5d} G->I {b['n_enter_gas']:5d} "
                  f"I->L {b['n_leave_liquid']:5d} I->G {b['n_leave_gas']:5d}  masked {b['n_masked']:6d}")
        rows.append((tcur, rnum, rex, it, dg['mdot_mean'], mex, meff, arel,
                     dg['interface_cells'], delta, r_a, dg['deficit_cells'],
                     dg['redistributed'], arel_end, area_end))
    d = s.phase_change_diagnostics()
    # thickness of the thermal boundary layer at t0: where the exact profile reaches 99 % of dT
    lo_, hi_ = r0, 200.0 * r0
    for _ in range(200):
        mid_ = 0.5 * (lo_ + hi_)
        if scriven_T(mid_, r0, beta, rr, dT) > 0.99 * dT:
            hi_ = mid_
        else:
            lo_ = mid_
    dt99 = hi_ - r0
    half = rows[len(rows) // 2:]
    errs = [abs(r[1] - r[2]) / r[2] for r in half]
    # GROWTH RATE vs OFFSET (WO-P3b).  R = 2 beta sqrt(alpha t)  <=>  R^2 = 4 beta^2 alpha t, a
    # STRAIGHT LINE through the origin.  Fitting R_num^2 against t over the last half separates the
    # two ways this gate can fail: a wrong growth RATE shows up in the slope (beta_eff != beta,
    # i.e. the method), a deficit acquired in the first steps and then carried shows up as a
    # negative INTERCEPT at the same slope (an effective time/radius offset, i.e. the start).
    _t = np.array([r[0] for r in half])
    _R2 = np.array([r[1] for r in half]) ** 2
    _sl, _ic = np.polyfit(_t, _R2, 1)
    beta_eff = math.sqrt(max(_sl, 0.0) / (4.0 * alpha_l))
    t_off = -_ic / _sl if _sl > 0 else float("nan")          # R^2 = 4 b^2 a (t - t_off)
    R_off = rows[-1][1] - 2 * beta_eff * math.sqrt(alpha_l * rows[-1][0])
    if verbose:
        print(f"  N = {n}^3  Ja = {ja:g}  ratio {ratio:g}  beta = {beta:.6f}  "
              f"R {r0:g} -> {r1:g} cells  steps {nst}  init {init} sub {sub}^3"
              + (f"  area mode {area_mode}" if area_mode is not None else ""))
        print(f"      init: thermal BL (99 % of dT) {dt99:.3f} cells, "
              f"R0/(2 beta^2) = {r0/(2*beta*beta):.3f} cells;  R0 from the colour field "
              f"{r0meas:.5f} vs exact {r0:.5f} ({100*(r0meas-r0)/r0:+.4f} %)")
        for k in list(range(0, len(rows), max(1, len(rows) // 8))) + [len(rows) - 1]:
            t_, a, b_, it, mn_, mx_, me_, ar_, nc_, de_, ra_, dc_, rd_, ae_, aa_ = rows[k]
            print(f"      t {t_:10.4f}  R_num {a:8.4f}  R_exact {b_:8.4f}  "
                  f"rel {100*(a-b_)/b_:+7.3f} %  mdot {mn_:.5e} vs {mx_:.5e} "
                  f"({100*(mn_-mx_)/mx_:+7.3f} %)  mdot_area {100*(me_-mx_)/mx_:+7.3f} %  "
                  f"A/4piR^2 {100*ar_:+6.2f} %  cells {nc_:6d} ({nc_/(4*math.pi*a*a):.3f}/h^2)"
                  f"  iters {it}")
            print(f"                 REGRESSION delta {de_:.5e} cells/step (delta/R "
                  f"{de_/a:.3e}), R_area {ra_:8.4f} ({100*(ra_-b_)/b_:+7.3f} % vs exact), "
                  f"clipped {dc_}, residue moved {rd_:.3e};  A_end/4piR^2 {100*ae_:+6.2f} % "
                  f"(R from A_end {math.sqrt(aa_/(4*math.pi)):8.4f}, {100*(math.sqrt(aa_/(4*math.pi))-b_)/b_:+7.3f} % vs exact)")
        early = ", ".join(f"{100*(r[4]-r[5])/r[5]:+.2f}" for r in rows[:6])
        print(f"      EARLY mdot rel error, steps 1..6: {early} %   "
              f"(last {100*(rows[-1][4]-rows[-1][5])/rows[-1][5]:+.2f} %)")
        _h = half
        print(f"      AREA-AVERAGED mdot rel error, last half: mean "
              f"{100*np.mean([(r[6]-r[5])/r[5] for r in _h]):+.3f} %, first {100*(_h[0][6]-_h[0][5])/_h[0][5]:+.3f} %, "
              f"last {100*(_h[-1][6]-_h[-1][5])/_h[-1][5]:+.3f} %;  A/(4 pi R^2) - 1 mean "
              f"{100*np.mean([r[7] for r in _h]):+.3f} %  [STALE, WO-P3e];  "
              f"A_end/(4 pi R^2) - 1 mean {100*np.mean([r[13] for r in _h]):+.3f} %")
        print(f"      R rel error at step 1 {100*(rows[0][1]-rows[0][2])/rows[0][2]:+.4f} %, "
              f"at the half point {100*(half[0][1]-half[0][2])/half[0][2]:+.4f} %, "
              f"at the end {100*(rows[-1][1]-rows[-1][2])/rows[-1][2]:+.4f} %")
        print(f"      GROWTH over the last half: beta_eff {beta_eff:.6f} vs beta {beta:.6f} "
              f"({100*(beta_eff-beta)/beta:+.3f} %), implied time offset {t_off:+.5f} of t0 "
              f"{t0:.5f} ({100*t_off/t0:+.2f} %), radius offset at the end {R_off:+.4f} cells")
        print(f"      max |dR|/R over the LAST HALF = {100*max(errs):.3f} %   [gate 1 %]")
        print(f"      pressure iters max {itmax}/600 (capped {capped}), band_div "
              f"{d['band_div']:.3e}, C in [{d['min_C']:.2e}, {d['max_C']:.6f}], "
              f"T in [{d['T_min']:.3e}, {d['T_max']:.6f}], fallback {d['fallback_cells']}, "
              f"unresolved {d['unresolved']:.1e}")
        if capped:
            print("      *** CAPPED pressure solve: rule 3b, this run is INVALID")
    return max(errs), capped


def plane_colour(n, nrm, sub=8, shift=0.37):
    """Liquid fraction (C = 1 liquid) of the half space n.(x - centre) < shift, by sub^3
    subsampling.  The sub-cell shift matters: a grid-aligned plane placed exactly on a cell face
    has NO mixed cell at all (CLAUDE.md, the VoF dt-limiter note), so the probe would read zero
    interfacial cells and no area."""
    nrm = np.asarray(nrm, dtype=float)
    nrm = nrm / np.linalg.norm(nrm)
    off = (np.arange(sub) + 0.5) / sub
    q = (np.arange(n)[:, None] + off[None, :]).ravel() - 0.5 * n
    d = nrm[0] * q[:, None, None] + nrm[1] * q[None, :, None] + nrm[2] * q[None, None, :] - shift
    frac = (d < 0.0).reshape(n, sub, n, sub, n, sub).mean(axis=(1, 3, 5))
    return np.asfortranarray(frac)


def plane_colour_periodic(n, nrm, sub=8, shift=0.37):
    """Liquid fraction of the PERIODIC tilted-plane family `0 <= (nrm . x) mod n < n/2`.

    WO-P3d.  A single half-space in a PERIODIC box is not a single plane: the wrap turns the
    domain faces into a second interface, and a JOINED surface reconstruction reports it (measured:
    the (1,1,0) row of `--area-probe=-2` comes out +5 % against the one-plane reference purely
    because of the seam, while a per-cell area misses it since the seam has no mixed cell).  The
    repair is a scene that is periodic BY CONSTRUCTION: with an INTEGER normal `nrm`, the level
    sets of `f = nrm . x  (mod n)` are closed flat surfaces of the torus, and the co-area formula
    gives their total area in closed form,

        integral |grad f| dV = |nrm|_2 n^3 = n * A(one level)   =>   A(one level) = |nrm|_2 n^2 ,

    so the two level sets bounding `f < n/2` carry EXACTLY `2 |nrm|_2 n^2`.  No edge convention, no
    seam, no reference uncertainty — for (0,0,1) it is 2 n^2, for (1,1,0) it is 2 sqrt(2) n^2.
    """
    nrm = np.asarray(nrm, dtype=float)
    off = (np.arange(sub) + 0.5) / sub
    q = (np.arange(n)[:, None] + off[None, :]).ravel()
    # the sub-cell shift is not cosmetic: with shift = 0 an integer normal puts the level set
    # exactly THROUGH the cell centres of a 45 deg plane (x + y is an integer at a half-integer
    # centre pair), which is a degenerate placement, and for (0,0,1) it puts it exactly on a cell
    # FACE, where there is no mixed cell at all and every per-cell area is 0 by construction.
    f = (nrm[0] * q[:, None, None] + nrm[1] * q[None, :, None] + nrm[2] * q[None, None, :]
         - shift) % n
    frac = (f < 0.5 * n).reshape(n, sub, n, sub, n, sub).mean(axis=(1, 3, 5))
    return np.asfortranarray(frac)


def cylinder_colour_chunked(n, ctr, R, sub):
    """Liquid fraction of an INFINITE cylinder of radius R along z (a curved interface with ONE
    non-zero principal curvature — the control that separates a metric error from a curvature
    error), by sub^2 subsampling of the transverse plane. Exact in z by construction, so its
    reference area 2 pi R n_z carries no z discretisation at all."""
    off = (np.arange(sub) + 0.5) / sub
    lo = max(0, int(math.floor(ctr - R - 2)))
    hi = min(n, int(math.ceil(ctr + R + 2)))
    q = (np.arange(lo, hi)[:, None] + off[None, :]).ravel() - ctr
    d2 = q[:, None] ** 2 + q[None, :] ** 2
    frac = (d2 < R * R).reshape(hi - lo, sub, hi - lo, sub).mean(axis=(1, 3))
    c = np.ones((n, n, n), order="F")
    c[lo:hi, lo:hi, :] = (1.0 - frac)[:, :, None]
    return np.asfortranarray(c)


def solenoidal_faces(n, amp, drift):
    """A face velocity field that is EXACTLY discretely divergence-free on the staggered grid.

    Two cellular (Taylor-Green) cells plus a uniform drift:

        u = A sin(k x_f) cos(k y_c) ,   v = -A cos(k x_c) sin(k y_f) + A sin(k y_f) cos(k z_c) ,
        w = -A cos(k y_c) sin(k z_f) ,   k = 2 pi / n ,   plus the constant `drift`.

    The discrete divergence vanishes to the LAST BIT, not to O(h^2), and that matters: `advect_vof`
    refuses a field whose `max_open_divergence_projected()` exceeds 1e-10 because Weymouth-Yue's
    exact conservation is conditional on it.  The identity is
    `sin(k(x+h)) - sin(k x) = 2 sin(kh/2) cos(k(x + h/2))`, and `x + h/2` of a FACE is exactly the
    CELL CENTRE the transverse factor is evaluated at — so the two axes' differences cancel term by
    term for each cellular pair.
    """
    k = 2.0 * math.pi / n
    f = np.arange(n, dtype=float)          # face coordinate of the LOW face of cell i
    c = np.arange(n, dtype=float) + 0.5    # cell centre
    u = amp * (np.sin(k * f)[:, None, None] * np.cos(k * c)[None, :, None]
               * np.ones(n)[None, None, :]) + drift[0]
    v = (-amp * np.cos(k * c)[:, None, None] * np.sin(k * f)[None, :, None] * np.ones(n)[None, None, :]
         + amp * np.ones(n)[:, None, None] * np.sin(k * f)[None, :, None] * np.cos(k * c)[None, None, :]
         + drift[1])
    w = (-amp * np.ones(n)[:, None, None] * np.cos(k * c)[None, :, None] * np.sin(k * f)[None, None, :]
         + drift[2])
    return (np.asfortranarray(u), np.asfortranarray(v), np.asfortranarray(w))


def area_advect(n=128, R=16.0, sub=16, steps=100, cfl=0.2, amp=0.0, drift=(0.5, 0.25, 0.125),
                modes=(0, 3, 4, 6, 7)):
    """WO-P3d gate (b): is the JOINED area BOUNDED under Weymouth-Yue advection?

    An a-priori probe measures a construction on an EXACT colour field.  A running one never sees
    one: WY leaves round-off residue in every cell its sweeps touch (down to 1e-300 -- the residue
    the V4 curvature needed `interfaceEps` for and the W0 block container needed `bubbleEps` for),
    and the interfacial-cell density climbs from the initialisation's value to ~1.5/h^2 within ten
    steps.  A per-cell area sums over those cells, so the question "does the area drift with the
    WISP population" is a real one; a joined sheet is built from the C = 1/2 crossing and should be
    blind to them.  Both are measured here, on the SAME field, step by step.

    `amp = 0` is the pure TRANSLATION control, where the exact answer stays `4 pi R^2` for ever.
    """
    ctr = 0.5 * n
    s = pf.Solver(n, n, n)
    s.set_rho(1.0)
    s.set_mu(1e-3)
    s.set_pressure_geometry(np.full((n, n, n), 1.0, order="F"))
    s.enable_vof()
    c0 = sphere_colour_chunked(n, ctr, R, sub)
    s.set_vof(c0)
    vol0 = float(n) ** 3 - c0.sum()
    ref = 4.0 * math.pi * (3.0 * vol0 / (4.0 * math.pi)) ** (2.0 / 3.0)
    u, v, w = solenoidal_faces(n, amp, drift)
    s.set_velocity(0, u)
    s.set_velocity(1, v)
    s.set_velocity(2, w)
    div = s.max_open_divergence_projected()
    dt = cfl / max(s.vof_max_courant(), 1e-30)
    print(f"  ADVECTION area gate: {n}^3, sphere R = {R:g} (sub {sub}^3), {steps} WY steps at "
          f"cfl {cfl:g} (dt {dt:.4g}), amp {amp:g}, drift {drift}")
    print(f"      max|div(open u)| of the prescribed field {div:.3e}  (advect_vof gate 1e-10);  "
          f"reference 4 pi R^2 = {ref:.4f}")

    def sample(tag):
        row = {}
        for m in modes:
            s.set_phase_change_area(m)
            row[m] = s.vof_interface_area()
        cc = s.get_vof()
        mixed = int(np.count_nonzero((cc > 0.0) & (cc < 1.0)))
        wisp = int(np.count_nonzero(((cc > 0.0) & (cc < 1e-8)) | ((cc < 1.0) & (cc > 1 - 1e-8))))
        vol = float(n) ** 3 - cc.sum()
        print(f"      {tag:>6}  mixed {mixed:7d}  wisps {wisp:7d}  dV/V {(vol-vol0)/vol0:+.3e}  "
              + "  ".join(f"m{m} {100*(row[m]/ref-1):+8.3f} %" for m in modes))
        return row

    hist = {m: [] for m in modes}
    r = sample("0")
    for m in modes:
        hist[m].append(r[m])
    for i in range(1, steps + 1):
        s.advect_vof(dt)
        if i % 10 == 0 or i == steps:
            r = sample(str(i))
            for m in modes:
                hist[m].append(r[m])
    print("      DRIFT over the run (max-min)/ref, and the last value:")
    for m in modes:
        h = np.array(hist[m])
        print(f"        mode {m}: span {100*(h.max()-h.min())/ref:7.3f} %   "
              f"last {100*(h[-1]/ref-1):+8.3f} %   first {100*(h[0]/ref-1):+8.3f} %")


def _mc_area(c):
    """Marching-cubes area of the C = 1/2 level set — an INDEPENDENT geometric reference for the
    summed PLIC area (it agrees with 4 pi R^2 to 0.5 % on the exact spheres below)."""
    try:
        from skimage import measure
    except ImportError:
        return float("nan")
    v, f, _, _ = measure.marching_cubes(np.ascontiguousarray(1.0 - c), level=0.5)
    return float(measure.mesh_surface_area(v, f))


def area_probe(n, radii, ratio=100.0, sub=4, mode=None, shape="sphere"):
    """A-PRIORI probe (WO-P3b): the summed PLIC interface area of an EXACT sphere, against 4 pi R^2.

    No time stepping, no energy solve, no velocity — the exact sphere fractions are set, one
    `apply_phase_change(0.0)` builds the interface, and `phase_change_diagnostics()['interface_area']`
    is read.  This isolates the geometry of `plicArea` + the MYC normals from every other part of
    the rung, and it is the quantity the R(t) gate integrates: the bubble grows as int mdot dA.
    """
    rho_l, rho_v = 1.0, 1.0 / ratio
    print(f"  a-priori INTERFACE AREA probe, {n}^3, exact sphere fractions (sub = {sub}^3), "
          f"area mode {0 if mode is None else mode}")
    prev = None
    for R in radii:
        ctr = 0.5 * n
        s = pf.Solver(n, n, n)
        s.set_rho(rho_l)
        s.set_mu(1e-3)
        s.set_pressure_geometry(np.full((n, n, n), 1.0, order="F"))
        s.enable_vof()
        if R > 0 and shape == "cylinder":
            c0 = cylinder_colour_chunked(n, ctr, R, sub)
            # R from the volume deficit, exactly as the sphere row does: V = pi R^2 n_z
            rm = math.sqrt((float(n) ** 3 - c0.sum()) / (math.pi * float(n)))
            ref, lbl = 2.0 * math.pi * rm * float(n), "2 pi R n_z"
        elif R > 0:
            c0 = sphere_colour_chunked(n, ctr, R, sub)
            ref, lbl = 4.0 * math.pi * ((3.0 * (float(n) ** 3 - c0.sum()) / (4.0 * math.pi))
                                        ** (1.0 / 3.0)) ** 2, "4 pi R^2  "
        elif R <= -5:  # WO-P3d: the PERIODIC tilted-plane family, with an EXACT analytic area.
            nrm = {-5: (0, 0, 1), -6: (1, 1, 0), -7: (1, 1, 1), -8: (1, 2, 3)}[int(R)]
            c0 = plane_colour_periodic(n, nrm, sub=sub)
            ref = 2.0 * math.sqrt(float(nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2)) * float(n) ** 2
            lbl = f"periodic {nrm}"
        else:  # R < 0 selects a PLANE: -1 = (0,0,1), -2 = (1,1,0), -3 = (1,1,1), -4 = (1,2,3).
            # INDICATIVE ONLY: a plane meets the domain faces, so both the PLIC sum and the
            # marching-cubes reference carry an edge effect.  The SPHERE rows are the clean ones.
            nrm = {-1: (0, 0, 1), -2: (1, 1, 0), -3: (1, 1, 1), -4: (1, 2, 3)}[int(R)]
            c0 = plane_colour(n, nrm)
            ref, lbl = float("nan"), f"plane {nrm}"
        s.set_vof(c0)
        s.set_property_model("rho", "linear", "C", [rho_v, rho_l - rho_v])
        s.enable_phase_change(rho_v, rho_l, 1.0)
        s.set_mass_flux_uniform(0.0)
        if mode is not None:
            s.set_phase_change_area(mode)
        s.apply_phase_change(0.0)
        d = s.phase_change_diagnostics()
        mc = _mc_area(c0) if R > -5 else float("nan")   # the periodic rows have an EXACT ref
        if not (ref == ref):
            ref = mc
        rel = d['interface_area'] / ref - 1.0
        ordr = ""
        if prev is not None and R > 0 and prev[0] > 0:
            ordr = f"   order vs R = {prev[0]:g}: {math.log(abs(prev[1]) / abs(rel)) / math.log(R / prev[0]):.3f}"
        # WO-P3d: `interface_area` is the sum over cells `pcIsInterfacial` accepts, i.e. what the
        # flux integral can USE; the joined sheet also books area to cells the wisp predicate calls
        # pure, and that part is dropped.  Report both, and never quote the usable sum alone: the
        # drop is a CANCELLATION against the sheet's own error, not accuracy.
        orph = ""
        if (mode or 0) >= 4:
            tot = d['interface_area'] + d['area_orphan']
            orph = (f" orphan {d['area_no_cascade_cells']} cells / {d['area_orphan']:.4f} h^2"
                    f" -> SHEET {tot:.4f} ({100*(tot/ref-1):+7.3f} %)")
        print(f"      R {R:6.2f} {lbl}  ({d['interface_cells']:6d} cells"
              + (f", HF {d['area_hf_cells']:6d} PV {d['area_pv_cells']:5d} "
                 f"none {d['area_no_cascade_cells']}" if mode else "") + orph + ")  "
              f"A_sum  {d['interface_area']:12.4f}  ref {ref:12.4f}  "
              f"marching cubes {mc:12.4f} ({100*(mc/ref-1):+6.3f} %)  "
              f"A_sum rel {100*rel:+7.3f} %{ordr}")
        prev = (R, rel)



def _sph_moments(w, ax, ctr, lmax=4):
    """Real spherical-harmonic moments of a signed cell weight `w` (a removal density), returned
    as the per-l RMS coefficient normalised by |sum w|.

    The regression shifts each cell's plane along that cell's OWN normal, so a systematic
    under-shift on (say) the octant diagonals would leave the bubble faceted.  A cubic lattice can
    only produce CUBIC-harmonic anisotropy, whose leading term is l = 4 (l = 2 vanishes by the
    three mirror symmetries), so the l = 2 row is the control and the l = 4 row is the signal.
    """
    import numpy as np
    X = (ax - ctr[0])[:, None, None] * np.ones((1, len(ax), len(ax)))
    Y = (ax - ctr[1])[None, :, None] * np.ones((len(ax), 1, len(ax)))
    Z = (ax - ctr[2])[None, None, :] * np.ones((len(ax), len(ax), 1))
    r = np.sqrt(X * X + Y * Y + Z * Z)
    r = np.where(r > 0, r, 1.0)
    u, v, t = X / r, Y / r, Z / r
    tot = float(w.sum())
    out = {}
    # l = 2: the five real components of the traceless quadrupole
    q = [u * v, v * t, t * u, u * u - v * v, (3.0 * t * t - 1.0) / math.sqrt(3.0)]
    out[2] = math.sqrt(sum(float((w * c).sum()) ** 2 for c in q) / 5.0)
    # l = 4: the cubic invariant (the only l = 4 combination a cubic lattice can excite) plus the
    # remaining independent components, measured as the RMS of the nine real l = 4 harmonics
    # written in cartesian form (unnormalised — the RATIO to the isotropic reference is what is
    # read, and both fields go through the identical expressions).
    u2, v2, t2 = u * u, v * v, t * t
    k = [u2 * u2 + v2 * v2 + t2 * t2 - 3.0 / 5.0,
         u * v * (u2 - v2), v * t * (v2 - t2), t * u * (t2 - u2),
         u * v * (7.0 * t2 - 1.0), v * t * (7.0 * u2 - 1.0), t * u * (7.0 * v2 - 1.0),
         (u2 - v2) * (7.0 * t2 - 1.0), (35.0 * t2 * t2 - 30.0 * t2 + 3.0)]
    out[4] = math.sqrt(sum(float((w * c).sum()) ** 2 for c in k) / 9.0)
    return tot, out


def _cubic_bins(n, ctr, nb=4):
    """(ctr is a 3-tuple of cell-index centres.)"""
    """Direction classes by the cubic invariant s = u_x^4 + u_y^4 + u_z^4, which runs from 1/3 on
    the BODY DIAGONAL (111) through 1/2 on a FACE diagonal (110) to 1 on an AXIS (100).  Binning
    the removed volume by s and dividing by the same bins of the EXACT removal is a direct,
    reference-free read-out of whether the regression is isotropic."""
    import numpy as np
    a = np.arange(n) + 0.5
    x, y, z = a - ctr[0], a - ctr[1], a - ctr[2]
    r2 = x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None, :] ** 2
    r2 = np.where(r2 > 0, r2, 1.0)
    s = (x[:, None, None] ** 4 + y[None, :, None] ** 4 + z[None, None, :] ** 4) / (r2 * r2)
    edges = np.linspace(1.0 / 3.0, 1.0, nb + 1)
    edges[-1] = 1.0 + 1e-9
    return np.digitize(s, edges) - 1, edges


def regress_probe(n=128, radii=(16.0,), deltas=(0.05, 0.1, 0.2), sub=16, modes=(0, 6),
                  ratio=100.0, advect_steps=0, drift=(0.5, 0.25, 0.125), cfl=0.2, nbin=4):
    """WO-P3e — the a-priori INTERFACE REGRESSION probe.  No time stepping, no energy solve, no
    velocity: an exact sphere goes in, ONE `apply_phase_change(dt)` with a uniform prescribed mdot
    runs the plane shift + clip-and-redistribute, and the colour field that comes out is compared
    against the analytically known answer.

    With `rho_l = 1` and `mdot = 1` the step's normal displacement is exactly `delta = dt`, so the
    exact liquid volume removed from a sphere of radius R is the shell

        dV = 4/3 pi ((R+delta)^3 - R^3) = 4 pi R^2 delta (1 + delta/R + delta^2/(3 R^2))

    (the bubble is the GAS, so evaporation GROWS it: R -> R + delta), the new radius from the
    volume deficit is R + delta exactly, and the new interfacial area is 4 pi (R+delta)^2.  Those
    three are the gate.  The fourth read-out is ISOTROPY: the shift acts along each cell's own
    normal with the cell's own area, so an under-shift on the octant diagonals would leave the
    bubble faceted -- measured both as the ratio of the removed volume to the exact removed volume
    in bins of the cubic invariant `u_x^4 + u_y^4 + u_z^4`, and as the l = 2 / l = 4 spherical
    harmonic moments of the removal density.

    `--regress-advect N` runs the sphere through N Weymouth-Yue steps of the PURE TRANSLATION field
    of gate (b) first, so the fractions are the ones a running solver actually carries (WY re-creates
    the sliver/wisp population within ten steps) while the exact reference stays a sphere.
    """
    ctr0 = 0.5 * n
    rho_l, rho_v = 1.0, 1.0 / ratio
    print(f"  a-priori REGRESSION probe, {n}^3, exact sphere fractions (sub = {sub}^3), "
          f"ratio {ratio:g}, {advect_steps} WY pre-steps")
    for R in radii:
        c0 = sphere_colour_chunked(n, ctr0, R, sub)
        cstart, ctr = c0, (ctr0, ctr0, ctr0)
        if advect_steps:
            s = pf.Solver(n, n, n)
            s.set_rho(1.0)
            s.set_mu(1e-3)
            s.set_pressure_geometry(np.full((n, n, n), 1.0, order="F"))
            s.enable_vof()
            s.set_vof(c0)
            u, v, w = solenoidal_faces(n, 0.0, drift)
            s.set_velocity(0, u)
            s.set_velocity(1, v)
            s.set_velocity(2, w)
            dta = cfl / max(s.vof_max_courant(), 1e-30)
            for _ in range(advect_steps):
                s.advect_vof(dta)
            cstart = s.get_vof()
            ctr = tuple(ctr0 + d * dta * advect_steps for d in drift)
            del s
        vg0 = float(n) ** 3 - cstart.sum()
        R0 = (3.0 * vg0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        mixed0 = int(np.count_nonzero((cstart > 0.0) & (cstart < 1.0)))
        # the exact colour field the pre-steps SHOULD have produced (a translated sphere), as the
        # control on how far the advected fractions are from exact
        cex0 = (sphere_colour(n, ctr[0], ctr[1], ctr[2], R, sub=sub) if advect_steps else c0)
        l1 = float(np.abs(cstart - cex0).sum())
        print(f"    R {R:g}: R0 from the volume {R0:.5f} ({100*(R0-R)/R:+.4f} %), mixed cells "
              f"{mixed0}, |C - C_exact|_1 {l1:.4e}")
        bins, edges = _cubic_bins(n, ctr, nbin)
        for mode in modes:
            for delta in deltas:
                s = pf.Solver(n, n, n)
                s.set_rho(rho_l)
                s.set_mu(1e-3)
                s.set_pressure_geometry(np.full((n, n, n), 1.0, order="F"))
                s.enable_vof()
                s.set_vof(cstart)
                s.set_property_model("rho", "linear", "C", [rho_v, rho_l - rho_v])
                s.enable_phase_change(rho_v, rho_l, 1.0)
                s.set_phase_change_area(mode)
                s.set_mass_flux_uniform(1.0)
                a_bef = s.vof_interface_area()
                s.apply_phase_change(delta)
                d = s.phase_change_diagnostics()
                c1 = s.get_vof()
                a_aft = s.vof_interface_area()
                del s
                vg1 = float(n) ** 3 - c1.sum()
                R1 = (3.0 * vg1 / (4.0 * math.pi)) ** (1.0 / 3.0)
                rem = cstart - c1
                removed = float(rem.sum())
                rex = 4.0 * math.pi * R0 * R0 * delta * (1.0 + delta / R0
                                                         + delta * delta / (3.0 * R0 * R0))
                R1x = R0 + delta
                print(f"      mode {mode}  delta {delta:6.3f}   "
                      f"dV {removed:12.6f} vs exact {rex:12.6f} ({100*(removed/rex-1):+8.4f} %)   "
                      f"R1 {R1:9.5f} vs {R1x:9.5f} ({100*(R1-R1x)/R1x:+8.5f} %)")
                print(f"                            A_before {a_bef:11.4f} "
                      f"({100*(a_bef/(4*math.pi*R0*R0)-1):+7.3f} %)  "
                      f"A_after {a_aft:11.4f} ({100*(a_aft/(4*math.pi*R1x*R1x)-1):+7.3f} %)  "
                      f"A_regr_used {d['interface_area']:11.4f}  removed_vol(diag) "
                      f"{d['removed_volume']:.6f}")
                pure_l = int(np.count_nonzero((cstart >= 1.0) & (c1 < 1.0)))
                pure_g = int(np.count_nonzero((cstart <= 0.0) & (c1 > 0.0)))
                emptied = int(np.count_nonzero((cstart > 0.0) & (c1 <= 0.0)))
                print(f"                            CENSUS clipped at 0 {d['deficit_cells']:6d}, "
                      f"at 1 {d['excess_cells']:5d}, |residue| moved {d['redistributed']:.6e}, "
                      f"unresolved {d['unresolved']:.2e}; cells emptied {emptied}, pure LIQUID "
                      f"cells touched {pure_l}, pure GAS cells touched {pure_g}; "
                      f"C in [{d['min_C']:.3e}, {d['max_C']:.6f}]")
                if bins is None:
                    continue
                cex1 = (sphere_colour_chunked(n, ctr0, R + delta, sub) if not advect_steps
                        else sphere_colour(n, ctr[0], ctr[1], ctr[2], R + delta, sub=sub))
                ex = cex0 - cex1
                rat = []
                for b in range(nbin):
                    m = bins == b
                    se, sa = float(ex[m].sum()), float(rem[m].sum())
                    rat.append(sa / se if se != 0 else float("nan"))
                lbl = "  ".join(f"s<{edges[b+1]:.3f} {100*(rat[b]-1):+7.3f} %" for b in range(nbin))
                print(f"                            ISOTROPY (removed/exact by cubic bin, "
                      f"axis=1 body-diag=1/3): {lbl}")
                ax = np.arange(n) + 0.5
                t_a, m_a = _sph_moments(rem, ax, ctr)
                t_e, m_e = _sph_moments(ex, ax, ctr)
                t_d, m_d = _sph_moments(rem - ex, ax, ctr)
                print(f"                            MOMENTS  l2/l0: actual {m_a[2]/abs(t_a):.3e} "
                      f"exact {m_e[2]/abs(t_e):.3e};  l4/l0: actual {m_a[4]/abs(t_a):.3e} exact "
                      f"{m_e[4]/abs(t_e):.3e};  residual l0 {t_d/t_e:+.3e} l2 "
                      f"{m_d[2]/abs(t_e):.3e} l4 {m_d[4]/abs(t_e):.3e}")


def _mdot_scene(n, R, ja, ratio, sub, alpha_l, area_mode, quad, plane, geom, dt, prof):
    """One a-priori mdot evaluation: `geom` x `prof`, the 2x2 that separates the two curvatures.

    `geom` = 'sphere' (the exact sphere, sub^3 fractions) or 'plane' (a flat interface normal to x
    at a sub-cell offset, the same sub^3 fractions).  `prof` = 'scriven' (Scriven's similarity
    profile of a bubble of radius R at the matching time, as a function of the signed distance to
    the interface -- so the sphere and plane rows carry IDENTICAL T', T'' at the interface) or
    'linear' (T = T'(R) d exactly, zero profile curvature).

    ONE `apply_phase_change(dt)` evaluates the shipped estimator: no energy solve, no velocity, no
    time stepping.  The only things measured are the one-sided gradient fit and the area on that
    field.  `plane x linear` is the exactness control (the fit's model is the exact state);
    `sphere x linear` isolates the INTERFACE's curvature; `plane x scriven` isolates the PROFILE's
    curvature and its resolution; `sphere x scriven` is the run's own configuration.
    """
    rr = 1.0 / ratio
    beta = scriven_beta(ja, rr)
    t = (R / (2 * beta)) ** 2 / alpha_l
    rho_l, rho_v = 1.0, rr
    cpl = 1.0
    k_l = alpha_l * rho_l * cpl
    rcp_l, rcp_v = rho_l * cpl, rho_v * cpl
    k_v = k_l / ratio
    dT = 1.0
    h_lv = rho_l * cpl * dT / (rho_v * ja)
    ctr = 0.5 * n
    Rdot = beta * math.sqrt(alpha_l / t)
    mdot_ex = rho_v * Rdot
    gradT = mdot_ex * h_lv / k_l                       # dT/dr at the interface, K/cell

    ax = (np.arange(n) + 0.5) - ctr
    shift = 0.37
    if geom == "sphere":
        c0 = sphere_colour_chunked(n, ctr, R, sub)
        rad = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
        Rmeas = (3.0 * (float(n) ** 3 - c0.sum()) / (4.0 * math.pi)) ** (1.0 / 3.0)
        ref_area = 4.0 * math.pi * Rmeas * Rmeas
        d = rad - R                                    # signed distance, + into the LIQUID
    else:
        # a flat interface normal to x at x = ctr + shift, LIQUID on the +x side (C = 1 liquid).
        off = (np.arange(sub) + 0.5) / sub
        q = (np.arange(n)[:, None] + off[None, :]).ravel() - ctr - shift
        frac = (q > 0.0).reshape(n, sub).mean(axis=1)
        c0 = np.asfortranarray(np.repeat(frac, n * n).reshape(n, n, n))
        d = (ax - shift)[:, None, None] * np.ones((1, n, n))
        Rmeas = R
        ref_area = float(n) ** 2
    if prof == "linear":
        tprof = np.asfortranarray(np.where(d > 0.0, gradT * d, 0.0))
    else:
        tprof = np.asfortranarray(np.vectorize(lambda x: scriven_T(R + x, R, beta, rr, dT))(
            np.maximum(d, 0.0)))

    s = pf.Solver(n, n, n)
    s.set_rho(rho_l)
    s.set_mu(1e-3)
    s.set_pressure_geometry(np.full((n, n, n), 1.0, order="F"))
    s.enable_vof()
    s.set_vof(c0)
    s.set_property_model("rho", "linear", "C", [rho_v, rho_l - rho_v])
    s.add_scalar("T", k_l / rcp_l, 1, 1)
    for f in range(6):
        s.set_scalar_bc("T", f, 1, 0.0)                # Neumann: the fit reads +-2 cells only
    s.set_field("T", tprof)
    s.enable_phase_change(rho_v, rho_l, h_lv)
    s.set_phase_change_thermal("T", 0.0, k_v, k_l, 0.0)
    s.set_phase_change_plane_dirichlet(plane)
    s.set_phase_change_energy(rcp_v, rcp_l)
    s.set_phase_change_quadratic_fit(quad)
    if area_mode is not None:
        s.set_phase_change_area(area_mode)
    s.set_phase_change_budget(True)     # WO-P3f: also read the GFM heat the SAME fields draw
    s.apply_phase_change(dt)
    dg = s.phase_change_diagnostics()
    bud = s.phase_change_budget()
    md = np.asarray(s.get_field("mdot"))
    cc = np.asarray(c0)
    iface = (cc > 1e-12) & (cc < 1.0 - 1e-12)
    A = dg['interface_area']
    m_area = rho_l * dg['removed_volume'] / (dt * A) if A > 0 else float('nan')
    # the two interfacial FLUXES on the same fields: the one the regression books
    # (`mdot h_lv A_Gamma`, from the least-squares one-sided fit) and the one the energy solve's
    # plane-anchored rows would actually draw (`q_gfm`), against the EXACT `mdot_ex h_lv A_exact`.
    e_lat = rho_l * dg['removed_volume'] * h_lv / dt
    q_ex = mdot_ex * h_lv * ref_area
    return dict(beta=beta, t=t, Rdot=Rdot, mdot_ex=mdot_ex, area=A, ref_area=ref_area,
                m_area=m_area, cells=int(dg['interface_cells']), m_cell=md[iface],
                Rmeas=Rmeas, gradT=gradT, bl=dT / gradT, e_lat=e_lat, q_gfm=bud['q_gfm'],
                q_ex=q_ex,
                delta=dg['removed_volume'] / A if A > 0 else float('nan'),
                clipped=int(dg['deficit_cells']))


def mdot_probe(n, radii, ja=0.5, ratio=100.0, sub=16, alpha_l=1.0, area_mode=None, quad=True,
               plane=True, dt=None, geoms=("sphere", "plane"), profs=("scriven",)):
    """WO-P3f gate (b): the a-priori CURVATURE BIAS of the one-sided mdot fit.

    For each radius the exact sphere carries Scriven's similarity profile at the matching time and
    the shipped estimator is evaluated ONCE.  The area-weighted mdot is compared with the analytic
    `mdot = rho_v beta sqrt(alpha/t)`; the flat-interface row with the SAME profile is the control.
    """
    print(f"  a-priori MDOT probe, {n}^3, Ja {ja:g}, ratio {ratio:g}, exact fractions "
          f"(sub = {sub}^3), area mode {0 if area_mode is None else area_mode}, "
          f"quadratic fit {quad}, plane Dirichlet {plane}")
    for prof in profs:
      for geom in geoms:
        prev = None
        for R in radii:
            step = dt
            if step is None:
                # the RUN's own regression step: delta = mdot dt / rho_l ~ 1e-3 cells
                rr = 1.0 / ratio
                b_ = scriven_beta(ja, rr)
                t_ = (R / (2 * b_)) ** 2 / alpha_l
                step = 1e-3 / max(rr * b_ * math.sqrt(alpha_l / t_), 1e-300)
            r = _mdot_scene(n, R, ja, ratio, sub, alpha_l, area_mode, quad, plane, geom, step,
                            prof)
            rel = r['m_area'] / r['mdot_ex'] - 1.0
            relA = r['area'] / r['ref_area'] - 1.0
            mc = r['m_cell']
            cell_rel = mc / r['mdot_ex'] - 1.0
            ordr = ""
            if prev is not None:
                ordr = (f"   order in h/R vs R = {prev[0]:g}: "
                        f"{math.log(abs(prev[1]) / abs(rel)) / math.log(R / prev[0]):.3f}")
            print(f"      {prof:7s} {geom:6s} R {R:6.2f}  t {r['t']:10.4f}  "
                  f"cells {r['cells']:6d}  A/A_ref {100*relA:+7.3f} %  BL {r['bl']:6.3f} cells  "
                  f"delta {r['delta']:.3e}  clipped {r['clipped']}")
            print(f"                     mdot_area {r['m_area']:.6e} vs exact {r['mdot_ex']:.6e}  "
                  f"rel {100*rel:+7.3f} %{ordr}")
            print(f"                     GFM heat -q_gfm {-r['q_gfm']:.6e} vs exact "
                  f"{r['q_ex']:.6e} (rel {100*(-r['q_gfm']/r['q_ex']-1):+7.3f} %), "
                  f"-q_gfm/E_lat {-r['q_gfm']/r['e_lat']:8.5f}")
            print(f"                     per-CELL mdot/exact - 1: mean {100*cell_rel.mean():+7.3f} "
                  f"% sd {100*cell_rel.std():6.3f} %  p05 {100*np.percentile(cell_rel,5):+7.3f} "
                  f"p50 {100*np.percentile(cell_rel,50):+7.3f} p95 "
                  f"{100*np.percentile(cell_rel,95):+7.3f} %")
            prev = (R, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ja", type=str, default="0.5,2,10")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--r0", type=float, default=6.0)
    ap.add_argument("--r1", type=float, default=20.0)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--cfl", type=float, default=0.2,
                    help="target interface-local Courant number of the adaptive dt (WO-P3b uses "
                         "it as the temporal-refinement ablation of the growth-rate deficit)")
    ap.add_argument("--no-plane", action="store_true")
    ap.add_argument("--no-consistent", action="store_true")
    ap.add_argument("--no-quad", action="store_true")
    ap.add_argument("--muscl", action="store_true")
    ap.add_argument("--init", choices=("similarity", "cellavg", "uniform"), default="similarity",
                    help="initial T(r, t0): Scriven's similarity profile at the cell centre "
                         "(default, = the shipped behaviour), the same profile cell-averaged, or "
                         "a uniform superheat with a sharp bubble (the classic trap; control)")
    ap.add_argument("--sub", type=int, default=4,
                    help="sub^3 subsampling of the RUN's initial sphere colour field (WO-P3c: "
                         "sub = 4 drops a quarter of the interfacial cells and 6 %% of the area)")
    ap.add_argument("--area-mode", type=int, default=None,
                    help="0 = PLIC/MYC area (rung P0/P1), 1 = cascade metric, 2 = cascade normal, "
                         "3 = cascade footprint; WO-P3d 4..7 = the JOINED marching-tetrahedra "
                         "sheet (4 C=1/2 centroid, 5 C=1/2 split, 6 PLIC-distance centroid, "
                         "7 PLIC-distance split)")
    ap.add_argument("--area-shape", choices=("sphere", "cylinder"), default="sphere",
                    help="what the area probe puts in the box (the cylinder has one zero "
                         "principal curvature and an exact z direction)")
    ap.add_argument("--area-sub", type=int, default=4,
                    help="sub^3 subsampling of the area probe's EXACT sphere fractions (WO-P3c: "
                         "the sliver cells a coarse sub drops are what WO-P3b measured)")
    ap.add_argument("--area-probe", type=str, default="",
                    help="comma-separated radii in cells: run the a-priori PLIC-area probe on an "
                         "exact sphere instead of the growth run, and exit")
    ap.add_argument("--area-advect", action="store_true",
                    help="WO-P3d gate (b): is the area BOUNDED under Weymouth-Yue advection? "
                         "Runs a sphere through a solenoidal field (or a pure translation with "
                         "--advect-amp 0) and prints every area mode on the SAME field beside the "
                         "mixed-cell and wisp census")
    ap.add_argument("--advect-amp", type=float, default=0.0)
    ap.add_argument("--advect-steps", type=int, default=100)
    ap.add_argument("--advect-r", type=float, default=16.0)
    ap.add_argument("--regress-probe", type=str, default="",
                    help="WO-P3e: comma-separated radii in cells — run the a-priori INTERFACE "
                         "REGRESSION probe (one apply_phase_change at a uniform mdot on an exact "
                         "sphere, against the analytic shell volume / radius / area / isotropy) "
                         "instead of the growth run, and exit")
    ap.add_argument("--regress-delta", type=str, default="0.05,0.1,0.2",
                    help="WO-P3e: comma-separated normal displacements per step, in cells")
    ap.add_argument("--regress-modes", type=str, default="0,6",
                    help="WO-P3e: comma-separated set_phase_change_area modes to run the probe on")
    ap.add_argument("--regress-advect", type=int, default=0,
                    help="WO-P3e: Weymouth-Yue pre-steps (pure translation) before the regression "
                         "step, so the fractions are advection-realistic")
    ap.add_argument("--budget", type=int, default=0,
                    help="WO-P3f instrument (a): print the ENERGY BUDGET of the energy solve every "
                         "N steps (0 = off, the shipped behaviour)")
    ap.add_argument("--mdot-probe", type=str, default="",
                    help="WO-P3f gate (b): comma-separated radii in cells - run the a-priori MDOT "
                         "probe (Scriven's similarity profile on an EXACT sphere, one "
                         "apply_phase_change, no energy solve) against the analytic mdot, with a "
                         "flat-interface control carrying the same profile; then exit")
    ap.add_argument("--mdot-geom", type=str, default="sphere,plane",
                    help="WO-P3f: which scenes the mdot probe runs (sphere and/or plane)")
    ap.add_argument("--mdot-prof", type=str, default="scriven",
                    help="WO-P3f: which temperature profiles the mdot probe runs "
                         "('scriven' and/or 'linear'; linear x plane is the exactness control)")
    ap.add_argument("--mdot-dt", type=float, default=0.0,
                    help="WO-P3f: fixed regression dt for the mdot probe (0 = pick it so the "
                         "normal displacement is the run's own 1e-3 cells)")
    a = ap.parse_args()
    if a.area_advect:
        area_advect(a.n, R=a.advect_r, sub=a.area_sub, steps=a.advect_steps, cfl=a.cfl,
                    amp=a.advect_amp)
        return 0
    if a.regress_probe:
        regress_probe(a.n, [float(x) for x in a.regress_probe.split(",")],
                      deltas=[float(x) for x in a.regress_delta.split(",")],
                      sub=a.area_sub, modes=[int(x) for x in a.regress_modes.split(",")],
                      ratio=a.ratio, advect_steps=a.regress_advect, cfl=a.cfl)
        return 0
    if a.mdot_probe:
        mdot_probe(a.n, [float(x) for x in a.mdot_probe.split(",")],
                   ja=float(a.ja.split(",")[0]), ratio=a.ratio, sub=a.area_sub,
                   area_mode=a.area_mode, quad=not a.no_quad, plane=not a.no_plane,
                   dt=(a.mdot_dt if a.mdot_dt > 0 else None),
                   geoms=tuple(a.mdot_geom.split(",")),
                   profs=tuple(a.mdot_prof.split(",")))
        return 0
    if a.area_probe:
        area_probe(a.n, [float(x) for x in a.area_probe.split(",")], ratio=a.ratio,
                   sub=a.area_sub, mode=a.area_mode, shape=a.area_shape)
        return 0
    print(f"P3 Scriven bubble growth, {a.n}^3, cfl {a.cfl:g}, sweeps {a.sweeps}, "
          f"plane Dirichlet {not a.no_plane}, "
          f"consistent energy {not a.no_consistent}, quadratic fit {not a.no_quad}, "
          f"energy MUSCL {a.muscl}, init {a.init}, sub {a.sub}^3, area mode {a.area_mode}")
    for ja in [float(x) for x in a.ja.split(",")]:
        run(a.n, ja, a.ratio, a.r0, a.r1, cfl=a.cfl, sweeps=a.sweeps, plane=not a.no_plane,
            consistent=not a.no_consistent, quad=not a.no_quad, muscl=a.muscl, init=a.init,
            sub=a.sub, area_mode=a.area_mode, budget=a.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
