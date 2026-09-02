#!/usr/bin/env python3
"""WO-V7 case 1 — the PORE DOUBLET: which branch fills first, and when.

Two straight channels of width `w` and `2w` leave a common inlet plenum and rejoin in a common
outlet plenum.  Liquid is injected at the inlet face at a fixed uniform velocity `U`; the doublet
is gas-filled at t = 0.  The question is the classical one (Chatzis & Dullien, *J. Can. Pet.
Tech.* 22:97, 1983; Moore & Slobod 1956): **which branch does the liquid fill first?**

    capillary-dominated (Ca -> 0), WETTING invasion (theta < 90):  the NARROW branch, because the
        driving capillary pressure 2 sigma cos(theta)/w is larger there;
    capillary-dominated, NON-WETTING invasion (theta > 90):  the WIDE branch, because the
        *resisting* entry pressure |2 sigma cos(theta)|/w is smaller there;
    viscous-dominated (Ca -> large):  the WIDE branch in BOTH cases, because a slit's hydraulic
        conductance goes as w^3.

So the wetting sweep is the discriminating one: it must INVERT between low and high Ca, while the
non-wetting sweep must NOT (both limits pick the wide branch).

CONVENTIONS AND CAVEATS, stated on every run (WO-V7 preamble):
  * the contact angle is the STATIC V5b fill.  The V6 dynamic angle is qualitative until V6b adds
    the velocity-side Navier slip, so the capillary numbers below are **apparent**: they are
    mu_l * U_inlet / sigma with U_inlet the prescribed superficial inlet velocity, not a measured
    contact-line speed.
  * density ratio 100.  Higher is not rated: V2b's uniform-velocity identity is floored at 1.2e-7
    by the solver's FLOAT momentum-operator storage, and WO-R's G3 gas-over-pool caveat (a pool at
    ratio 1000 picks up 3e-2 of the inlet speed) is unresolved.
  * no periodic net body force anywhere (VOF_PLAN 13.2 item 6): the drive is an inflow/outflow pair.
  * the solid is kept clear of the inflow and outflow planes (flow's open-BC/cut-cell defect,
    `doc/cutcell_openbc_convergence.md`).
  * a run whose pressure solve touches its cap is INVALID and is reported as such (rule 3b).

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/pore_scale/pore_doublet.py [--quick]
                                     [--theta 45,135] [--ca 1e-4,1e-3,1e-2]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import peclet.flow as pf

# ------------------------------------------------------------------ scene (all lengths in cells)
NX, NY = 88, 4
PLEN_IN, BRANCH, PLEN_OUT = 20, 48, 20        # x layout: inlet plenum | branches | outlet plenum
W_NARROW, W_WIDE = 16, 32                     # the 1:2 width pair
Z_WALL_BOT, Z_SEPT, Z_WALL_TOP = 8, 8, 16     # bottom wall / septum / top wall thickness
# WHERE the SDF wall sits inside its cell is load-bearing, and it is not a nicety (WO-S finding 5,
# reproduced here at 30x the strength).  With the walls on INTEGER coordinates -- i.e. exactly on a
# cell face, so no wall cell is cut at all -- the channel-mouth corner reads max|u| = 1.12e+02 on
# the FIRST step against a physical 0.42, and the run then diverges geometrically (max|u| 1.5e+08
# by step 300 while the dt limiter chases it down to 1e-9).  Shifting every wall by a QUARTER cell
# gives 3.56 on the first step, DECAYING to 1.2 -- same scene, same everything else.  See the
# WO-V7 findings table.
WALL_SHIFT = 0.25
NZ = Z_WALL_BOT + W_WIDE + Z_SEPT + W_NARROW + Z_WALL_TOP     # 8+32+8+16+16 = 80
X1, X2 = PLEN_IN, PLEN_IN + BRANCH
ZW_LO, ZW_HI = Z_WALL_BOT, Z_WALL_BOT + W_WIDE                     # wide branch  [8, 40)
ZN_LO, ZN_HI = ZW_HI + Z_SEPT, ZW_HI + Z_SEPT + W_NARROW           # narrow       [48, 64)

# ------------------------------------------------------------------------------------ properties
SIGMA = 100.0
RHO_L, RHO_G = 100.0, 1.0          # density ratio 100 (see the docstring)
MU_L, MU_G = 4.0, 0.04             # viscosity ratio 100
CAP_CFL = 0.5                      # safety factor on the Brackbill capillary limit
PRESS_CAP = 400
NPY_DIR = ""
PRESS_RTOL = 1e-8                  # measured: 1e-11 costs 41 -> 60 FCG iterations for nothing
MG_LEVELS = 5                      # measured sweet spot on this grid (3: 81 iters, 8: 41 at +14 %)
VEL_SWEEPS = 20


def box2d(px, pz, xlo, xhi, zlo, zhi):
    """Exact signed distance (outside) to an axis-aligned box in the (x, z) plane."""
    cx, cz = 0.5 * (xlo + xhi), 0.5 * (zlo + zhi)
    ex, ez = 0.5 * (xhi - xlo), 0.5 * (zhi - zlo)
    qx, qz = np.abs(px - cx) - ex, np.abs(pz - cz) - ez
    out = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qz, 0.0) ** 2)
    return out + np.minimum(np.maximum(qx, qz), 0.0)


def doublet_sdf():
    """> 0 in the fluid (flow's convention).  Three solid boxes, all clear of x = 0 and x = NX."""
    x = (np.arange(NX) + 0.5)[:, None]
    z = (np.arange(NZ) + 0.5)[None, :]
    BIG = 100.0
    q = WALL_SHIFT
    d = np.minimum.reduce([
        box2d(x, z, X1 + q, X2 + q, -BIG, ZW_LO + q),          # floor of the wide branch
        box2d(x, z, X1 + q, X2 + q, ZW_HI + q, ZN_LO + q),     # the septum between the branches
        box2d(x, z, X1 + q, X2 + q, ZN_HI + q, NZ + BIG),      # roof of the narrow branch
    ])
    return np.asfortranarray(np.broadcast_to(d[:, None, :], (NX, NY, NZ)).astype(float).copy())


class Health:
    """Rule 3b: a capped pressure solve makes a run INVALID."""

    def __init__(self, cap=PRESS_CAP):
        self.cap, self.iters, self.div, self.capped = cap, 0, 0.0, 0

    def sample(self, s):
        it = s.last_pressure_iterations()
        self.iters = max(self.iters, it)
        self.capped += int(it >= self.cap)
        self.div = max(self.div, s.max_open_divergence_projected())

    @property
    def valid(self):
        return self.capped == 0

    def __str__(self):
        return (f"pressure {self.iters}/{self.cap} ({self.capped} capped) "
                f"{'OK' if self.valid else '*** CAPPED -> RUN INVALID ***'}, "
                f"max|div(open u)|_projected {self.div:.3e}")


def build(theta_deg, U, momentum=True):
    s = pf.Solver(NX, NY, NZ)
    s.set_rho(RHO_L)
    s.set_mu(MU_L)
    s.set_dt(0.1)
    s.set_domain_bc(0, 2, U, 0.0, 0.0)      # -x inflow, uniform U
    s.set_domain_bc(1, 3, 0.0, 0.0, 0.0)    # +x outflow
    s.set_domain_bc(4, 1, 0.0, 0.0, 0.0)    # -z wall  (buried in the solid)
    s.set_domain_bc(5, 1, 0.0, 0.0, 0.0)    # +z wall  (buried in the solid)
    # y stays periodic: the case is quasi-2D.
    s.set_velocity_solver_params(VEL_SWEEPS)
    s.set_pressure_multigrid(True, levels=MG_LEVELS)
    s.set_pressure_solver_params(80)
    s.set_solid(doublet_sdf(), cutcell_pressure=True)
    s.enable_vof()
    # t = 0 is the front AT the branch entrances: the inlet plenum is already liquid-filled.  This
    # is not a shortcut for its own sake — filling the plenum from the inlet at Ca = 1e-4 costs as
    # many steps again as the branches themselves, and the plenum carries no physics.
    c0 = np.zeros((NX, NY, NZ), order="F")
    c0[:X1, :, :] = 1.0
    s.set_vof(c0)
    s.set_property_model("rho", "linear", "C", [RHO_G, RHO_L - RHO_G])
    s.set_property_model("mu", "linear", "C", [MU_G, MU_L - MU_G])
    if momentum:
        s.enable_vof_momentum(RHO_G, RHO_L)
    s.set_surface_tension(SIGMA)
    s.set_capillary_cfl(CAP_CFL)
    s.set_contact_angle(theta_deg)
    # THE DRIVER IS SELECTED LAST — `set_property_model("rho", ...)` fires set_density_mode, which
    # reselects Chebyshev and silently discards an earlier choice (WO-H; vof_open_boundaries.py).
    s.set_pressure_fcg(True, PRESS_CAP, PRESS_RTOL)
    s.set_vof_inflow(0, 1.0)                # liquid enters
    s.set_vof_backflow(1, 0.0)              # gas backflow at the outlet (inletOutlet)
    return s


def _run_end(mask):
    """Length of the run of True that starts at index 0 (0 if the first entry is False)."""
    idx = np.nonzero(~mask)[0]
    return float(idx[0]) if idx.size else float(mask.size)


def fronts(C, eps):
    """Two front positions per branch, in cells from the branch entrance X1, plus the branch
    liquid saturation.

    `tip`  the leading edge: the last x, contiguous from the entrance, at which ANY cell of the
           cross-section is more than half liquid.
    `bar`  the cross-section-AVERAGED front: the last x, contiguous from the entrance, at which the
           openness-weighted mean colour of the cross-section exceeds a half.

    Both are reported because they differ by the length of the meniscus, and that length scales
    with the CHANNEL WIDTH: at theta = 135 the meniscus in the 32-cell branch is about twice as
    long as in the 16-cell one, so `bar` alone carries a systematic bias of order w/2 in favour of
    the NARROW branch.  Breakthrough is declared on `tip`, which has no such bias.
    """
    out = {}
    for name, (zlo, zhi) in (("narrow", (ZN_LO, ZN_HI)), ("wide", (ZW_LO, ZW_HI))):
        c = C[X1:X2, :, zlo:zhi]
        e = eps[X1:X2, :, zlo:zhi]
        wet = ((c > 0.5) & (e > 0.0)).any(axis=(1, 2))
        col = (c * e).sum(axis=(1, 2)) / np.maximum(e.sum(axis=(1, 2)), 1e-30)
        tip = _run_end(wet)
        bar = _run_end(col > 0.5)
        sat = float((c * e).sum() / max(e.sum(), 1e-30))
        out[name] = (tip, bar, sat)
    return out


def run(theta_deg, ca, steps, sample_every, label, momentum=True, budget=None):
    U = ca * SIGMA / MU_L
    Re = RHO_L * U * W_NARROW / MU_L
    print("\n" + "=" * 100)
    print(f"PORE DOUBLET  theta = {theta_deg:g} deg   Ca(apparent) = {ca:g}   [{label}]")
    print("=" * 100)
    print(f"  grid {NX}x{NY}x{NZ}; branches x in [{X1}, {X2}), narrow w = {W_NARROW} "
          f"z in [{ZN_LO}, {ZN_HI}), wide 2w = {W_WIDE} z in [{ZW_LO}, {ZW_HI})")
    print(f"  sigma {SIGMA:g}, rho {RHO_L:g}/{RHO_G:g} (ratio {RHO_L/RHO_G:g}), "
          f"mu {MU_L:g}/{MU_G:g} (ratio {MU_L/MU_G:g}), momentum consistency "
          f"{'ON' if momentum else 'off'}")
    print(f"  inlet U = Ca*sigma/mu_l = {U:.6g} cells/s -> channel Reynolds "
          f"rho_l U w / mu_l = {Re:.4g}")
    ct = math.cos(math.radians(theta_deg))
    pc_n, pc_w = 2 * SIGMA * ct / W_NARROW, 2 * SIGMA * ct / W_WIDE
    # viscous scale: the pressure drop of the wide branch at the mean branch velocity
    ubr = U * NZ / (W_NARROW + W_WIDE)
    dpv = 12.0 * MU_L * ubr * BRANCH / W_WIDE ** 2
    print(f"  capillary pressure 2 sigma cos(theta)/w:  narrow {pc_n:+.4g}, wide {pc_w:+.4g}, "
          f"difference {pc_n - pc_w:+.4g}")
    print(f"  viscous scale 12 mu_l u_branch L / (2w)^2 = {dpv:.4g}  ->  "
          f"|dPc| / dPvisc = {abs(pc_n - pc_w)/dpv:.4g}  "
          f"({'CAPILLARY' if abs(pc_n-pc_w) > dpv else 'VISCOUS'}-dominated by this estimate)")

    s = build(theta_deg, U, momentum)
    L = s.vof_step_limits()
    print(f"  dt census at t = 0: capillary_dt {L['capillary_dt']:.5g} (x{CAP_CFL} = "
          f"{CAP_CFL*L['capillary_dt']:.5g}), WY cfl_dt {L['cfl_dt']:.5g}, binding "
          f"{L['binding']:.5g}  ->  {'CAPILLARY' if L['capillary_binds'] else 'CFL'} binds")
    print("  (capillary_dt is a function of the CURRENT density field: with the domain still all")
    print("   gas it reads the gas value; it relaxes once liquid is present.  dt is therefore")
    print("   re-picked from the solver's own limiter EVERY step.)")
    dt = CAP_CFL * L["capillary_dt"]
    s.set_dt(dt)

    h = Health()
    eps = np.asarray(s.vof_geometry(0))
    hist = []
    bt = {"narrow": None, "wide": None}
    ncap_cfl = 0
    t0 = time.time()
    t = 0.0
    nsteps = 0
    umax_gas = 0.0
    for i in range(steps):
        # keep dt under BOTH limits, re-picked from the solver's own limiter every step
        L = s.vof_step_limits()
        dtc = CAP_CFL * L["capillary_dt"]
        dt = min(dtc, 0.8 * L["cfl_dt"]) if L["cfl_dt"] > 0 else dtc
        ncap_cfl += int(dt < dtc * 0.999)
        s.set_dt(dt)
        s.step()
        t += dt
        nsteps += 1
        h.sample(s)
        if (i + 1) % sample_every == 0 or i == steps - 1:
            C = np.asarray(s.get_vof())
            f = fronts(C, eps)
            hist.append((t, f["narrow"][0], f["narrow"][1], f["narrow"][2],
                         f["wide"][0], f["wide"][1], f["wide"][2]))
            if len(hist) % 10 == 1:
                print(f"    step {i+1:7d}  t {t:10.4g}  tip n/w {f['narrow'][0]:5.0f}/"
                      f"{f['wide'][0]:5.0f}  bar n/w {f['narrow'][1]:5.0f}/{f['wide'][1]:5.0f}  "
                      f"S n/w {f['narrow'][2]:.3f}/{f['wide'][2]:.3f}  "
                      f"({time.time()-t0:.0f} s, {1000*(time.time()-t0)/(i+1):.0f} ms/step)",
                      flush=True)
            for k in ("narrow", "wide"):
                if bt[k] is None and f[k][0] >= BRANCH - 1:
                    bt[k] = (t, i + 1)
            # spurious-current probe: max|u| in the pure gas AHEAD of both fronts
            xg = int(min(max(f["narrow"][0], 0.0), max(f["wide"][0], 0.0))) + X1 + 6
            if xg < X2:
                g = max(np.abs(np.asarray(fn())[xg:X2, :, :]).max()
                        for fn in (s.get_u, s.get_v, s.get_w))
                umax_gas = max(umax_gas, float(g))
            if bt["narrow"] and bt["wide"]:
                break
            if budget and time.time() - t0 > budget:
                print(f"    *** wall-clock budget {budget:.0f} s reached at step {i+1} "
                      f"(t = {t:.5g}) — the run is reported as a PARTIAL fill ***")
                break
    wall = time.time() - t0
    C = np.asarray(s.get_vof())
    f = fronts(C, eps)
    d = s.vof_diagnostics()
    print(f"\n  ran {nsteps} steps to t = {t:.4g} s in {wall:.0f} s "
          f"({1000*wall/max(nsteps,1):.2f} ms/step, SHARED GPU)")
    print(f"  dt: final {dt:.5g}; the WY CFL limit displaced the capillary one on "
          f"{ncap_cfl} of the {nsteps} steps ({100.0*ncap_cfl/max(nsteps,1):.1f} % -> the "
          f"capillary limit binds on the rest)")
    print(f"  final fronts (cells into the branch, of {BRANCH}):  TIP narrow {f['narrow'][0]:.0f}"
          f" / wide {f['wide'][0]:.0f};  MEAN narrow {f['narrow'][1]:.0f} / wide "
          f"{f['wide'][1]:.0f}")
    print(f"  final branch liquid saturation:  narrow {f['narrow'][2]:.4f}  "
          f"wide {f['wide'][2]:.4f}")
    for k in ("narrow", "wide"):
        print(f"  breakthrough {k:6s}: " + (f"t = {bt[k][0]:.5g} s at step {bt[k][1]}"
                                           if bt[k] else "NOT reached in this run"))
    winner = None
    if bt["narrow"] and bt["wide"]:
        winner = "narrow" if bt["narrow"][0] < bt["wide"][0] else "wide"
    elif bt["narrow"]:
        winner = "narrow"
    elif bt["wide"]:
        winner = "wide"
    elif f["narrow"][0] != f["wide"][0]:
        winner = "narrow" if f["narrow"][0] > f["wide"][0] else "wide"
    elif f["narrow"][2] != f["wide"][2]:
        winner = "narrow" if f["narrow"][2] > f["wide"][2] else "wide"
    print(f"  FILLS FIRST: {winner if winner else 'undecided (neither front advanced)'}")
    print(f"  max|u| in the gas ahead of both fronts {umax_gas:.4e} = {umax_gas/U:.3f} U  "
          f"-> Ca = mu_l max|u|/sigma = {MU_L*umax_gas/SIGMA:.3e} against the imposed {ca:g}.")
    print("    This is an UPPER BOUND on the spurious current, not a measurement of it: the gas")
    print("    ahead of the front is genuinely being displaced, so it carries the real flow too.")
    print(f"  colour: sum {d['sum']:.6g}, solid_sum {d['solid_sum']:.3e}, "
          f"min/max over uncut fluid {d['min_fluid']:.3e} / {d['max_fluid']:.6f}, "
          f"clipped {d['clipped_volume']:.3e}")
    print(f"  {h}")
    if NPY_DIR:
        os.makedirs(NPY_DIR, exist_ok=True)
        np.save(os.path.join(NPY_DIR,
                             f"doublet_C_theta{int(theta_deg)}_ca{ca:g}.npy"), C[:, NY // 2, :])
        np.save(os.path.join(NPY_DIR, "doublet_eps.npy"), eps[:, NY // 2, :])
    return dict(theta=theta_deg, ca=ca, U=U, Re=Re, steps=nsteps, t=t, wall=wall,
                ms_per_step=1000 * wall / max(nsteps, 1), dt=dt,
                tip_narrow=f["narrow"][0], tip_wide=f["wide"][0],
                bar_narrow=f["narrow"][1], bar_wide=f["wide"][1],
                sat_narrow=f["narrow"][2], sat_wide=f["wide"][2],
                bt_narrow=bt["narrow"][0] if bt["narrow"] else None,
                bt_wide=bt["wide"][0] if bt["wide"] else None,
                winner=winner, iters=h.iters, capped=h.capped, div=h.div,
                valid=h.valid, umax_gas=umax_gas, ca_spurious=MU_L * umax_gas / SIGMA,
                pc_narrow=pc_n, pc_wide=pc_w, dp_visc=dpv, hist=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--theta", default="45,135")
    ap.add_argument("--ca", default="1e-4,1e-3,1e-2")
    ap.add_argument("--no-momentum", action="store_true")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="wall-clock seconds per (theta, Ca); 0 = no limit")
    ap.add_argument("--npy", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    global NPY_DIR
    NPY_DIR = a.npy
    thetas = [float(x) for x in a.theta.split(",")]
    cas = [float(x) for x in a.ca.split(",")]
    print("WO-V7 case 1 — the pore doublet.  Static contact angle (V5b); the capillary numbers")
    print("are APPARENT (mu_l U_inlet / sigma), because contact-line mobility is set by the")
    print("solver's numerical slip until V6b lands (VOF_PLAN 13.2 item 5).")
    rows = []
    for th in thetas:
        for ca in cas:
            # the travel time is ~ (plenum + branch/contraction)/U; budget 1.6x of it
            U = ca * SIGMA / MU_L
            ttrav = BRANCH * (W_NARROW + W_WIDE) / (NZ * U)
            dt0 = CAP_CFL * math.sqrt((RHO_L + RHO_G) / (4 * math.pi * SIGMA))
            steps = int(1.7 * ttrav / dt0)
            if a.quick:
                steps = min(steps, 400)
            rows.append(run(th, ca, steps, max(steps // 120, 5),
                            "quick" if a.quick else "full",
                            momentum=not a.no_momentum,
                            budget=a.budget or None))
    print("\n" + "=" * 100)
    print("SUMMARY — pore doublet")
    print("=" * 100)
    print(f"{'theta':>6} {'Ca':>8} {'Re':>7} {'t_bt narrow':>12} {'t_bt wide':>12} "
          f"{'fills first':>12} {'tip_n':>6} {'tip_w':>6} {'S_n':>7} {'S_w':>7} {'iters':>6} "
          f"{'ms/step':>8} {'valid':>6}")
    for r in rows:
        bn = f"{r['bt_narrow']:.4g}" if r["bt_narrow"] else "-"
        bw = f"{r['bt_wide']:.4g}" if r["bt_wide"] else "-"
        print(f"{r['theta']:6g} {r['ca']:8g} {r['Re']:7.3g} {bn:>12} {bw:>12} "
              f"{str(r['winner']):>12} {r['tip_narrow']:6.0f} {r['tip_wide']:6.0f} "
              f"{r['sat_narrow']:7.4f} {r['sat_wide']:7.4f} "
              f"{r['iters']:6d} {r['ms_per_step']:8.2f} "
              f"{'yes' if r['valid'] else 'NO':>6}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\n  wrote {a.out}")
    return 0 if all(r["valid"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
