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
        quad=True, muscl=False, init="similarity", verbose=True):
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
    c0 = sphere_colour(n, ctr, ctr, ctr, r0)
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
        meff = rho_l * dg['removed_volume'] / (dt_used * area) if area > 0 else float("nan")
        arel = area / (4.0 * math.pi * rnum * rnum) - 1.0
        rows.append((tcur, rnum, rex, it, dg['mdot_mean'], mex, meff, arel))
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
              f"R {r0:g} -> {r1:g} cells  steps {nst}  init {init}")
        print(f"      init: thermal BL (99 % of dT) {dt99:.3f} cells, "
              f"R0/(2 beta^2) = {r0/(2*beta*beta):.3f} cells;  R0 from the colour field "
              f"{r0meas:.5f} vs exact {r0:.5f} ({100*(r0meas-r0)/r0:+.4f} %)")
        for k in list(range(0, len(rows), max(1, len(rows) // 8))) + [len(rows) - 1]:
            t_, a, b_, it, mn_, mx_, me_, ar_ = rows[k]
            print(f"      t {t_:10.4f}  R_num {a:8.4f}  R_exact {b_:8.4f}  "
                  f"rel {100*(a-b_)/b_:+7.3f} %  mdot {mn_:.5e} vs {mx_:.5e} "
                  f"({100*(mn_-mx_)/mx_:+7.3f} %)  mdot_area {100*(me_-mx_)/mx_:+7.3f} %  "
                  f"A/4piR^2 {100*ar_:+6.2f} %  iters {it}")
        early = ", ".join(f"{100*(r[4]-r[5])/r[5]:+.2f}" for r in rows[:6])
        print(f"      EARLY mdot rel error, steps 1..6: {early} %   "
              f"(last {100*(rows[-1][4]-rows[-1][5])/rows[-1][5]:+.2f} %)")
        _h = half
        print(f"      AREA-AVERAGED mdot rel error, last half: mean "
              f"{100*np.mean([(r[6]-r[5])/r[5] for r in _h]):+.3f} %, first {100*(_h[0][6]-_h[0][5])/_h[0][5]:+.3f} %, "
              f"last {100*(_h[-1][6]-_h[-1][5])/_h[-1][5]:+.3f} %;  A/(4 pi R^2) - 1 mean "
              f"{100*np.mean([r[7] for r in _h]):+.3f} %")
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


def _mc_area(c):
    """Marching-cubes area of the C = 1/2 level set — an INDEPENDENT geometric reference for the
    summed PLIC area (it agrees with 4 pi R^2 to 0.5 % on the exact spheres below)."""
    try:
        from skimage import measure
    except ImportError:
        return float("nan")
    v, f, _, _ = measure.marching_cubes(np.ascontiguousarray(1.0 - c), level=0.5)
    return float(measure.mesh_surface_area(v, f))


def area_probe(n, radii, ratio=100.0, sub=4):
    """A-PRIORI probe (WO-P3b): the summed PLIC interface area of an EXACT sphere, against 4 pi R^2.

    No time stepping, no energy solve, no velocity — the exact sphere fractions are set, one
    `apply_phase_change(0.0)` builds the interface, and `phase_change_diagnostics()['interface_area']`
    is read.  This isolates the geometry of `plicArea` + the MYC normals from every other part of
    the rung, and it is the quantity the R(t) gate integrates: the bubble grows as int mdot dA.
    """
    rho_l, rho_v = 1.0, 1.0 / ratio
    print(f"  a-priori PLIC AREA probe, {n}^3, exact sphere fractions (sub = {sub}^3)")
    prev = None
    for R in radii:
        ctr = 0.5 * n
        s = pf.Solver(n, n, n)
        s.set_rho(rho_l)
        s.set_mu(1e-3)
        s.set_pressure_geometry(np.full((n, n, n), 1.0, order="F"))
        s.enable_vof()
        if R > 0:
            c0 = sphere_colour(n, ctr, ctr, ctr, R, sub=sub)
            ref, lbl = 4.0 * math.pi * ((3.0 * (float(n) ** 3 - c0.sum()) / (4.0 * math.pi))
                                        ** (1.0 / 3.0)) ** 2, "4 pi R^2  "
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
        s.apply_phase_change(0.0)
        d = s.phase_change_diagnostics()
        mc = _mc_area(c0)
        if not (ref == ref):
            ref = mc
        rel = d['interface_area'] / ref - 1.0
        ordr = ""
        if prev is not None and R > 0 and prev[0] > 0:
            ordr = f"   order vs R = {prev[0]:g}: {math.log(abs(prev[1]) / abs(rel)) / math.log(R / prev[0]):.3f}"
        print(f"      R {R:6.2f} {lbl}  ({d['interface_cells']:6d} interfacial cells)  "
              f"A_PLIC {d['interface_area']:12.4f}  ref {ref:12.4f}  "
              f"marching cubes {mc:12.4f} ({100*(mc/ref-1):+6.3f} %)  "
              f"A_PLIC rel {100*rel:+7.3f} %{ordr}")
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
    ap.add_argument("--area-probe", type=str, default="",
                    help="comma-separated radii in cells: run the a-priori PLIC-area probe on an "
                         "exact sphere instead of the growth run, and exit")
    a = ap.parse_args()
    if a.area_probe:
        area_probe(a.n, [float(x) for x in a.area_probe.split(",")], ratio=a.ratio)
        return 0
    print(f"P3 Scriven bubble growth, {a.n}^3, cfl {a.cfl:g}, sweeps {a.sweeps}, "
          f"plane Dirichlet {not a.no_plane}, "
          f"consistent energy {not a.no_consistent}, quadratic fit {not a.no_quad}, "
          f"energy MUSCL {a.muscl}, init {a.init}")
    for ja in [float(x) for x in a.ja.split(",")]:
        run(a.n, ja, a.ratio, a.r0, a.r1, cfl=a.cfl, sweeps=a.sweeps, plane=not a.no_plane,
            consistent=not a.no_consistent, quad=not a.no_quad, muscl=a.muscl, init=a.init)
    return 0


if __name__ == "__main__":
    sys.exit(main())
