#!/usr/bin/env python3
"""WO-P23 rung P2 — the SUCKING INTERFACE (Welch & Wilson, JCP 160:662, 2000).

A planar interface with SATURATED VAPOUR on one side and SUPERHEATED LIQUID on the other. All the
latent heat comes from the liquid, so this is the first rung on which the LIQUID half of the
one-sided gradient fit, the per-phase k(C) / rho c_p(C) closures and the consistent rho c_p T
transport are all live at once (rung P1's liquid was saturated and pinned, so none of them entered).

GEOMETRY.  Vapour occupies x < X(t) against a wall held at T_sat; liquid occupies x > X(t) and is
superheated to T_inf far away.  The interface moves INTO the liquid and the liquid is pushed AWAY
from it at u_l = mdot (1/rho_v - 1/rho_l), so the domain needs an OUTLET at the liquid end: the net
vapour production is a real volume flux that has to leave.  (WO-P01's closed-column trick with a
balancing sink cannot be used here — the sink removes VOLUME from cells whose colour is 1, so the
colour drops there, and over a run this long the required removal is many cells' worth.)

WHY THE OUTLET IS SOUND AT HIGH DENSITY RATIO.  `applyBoundaryOpenness` re-imposes the literal 1.0
at a boundary face instead of the variable-density coefficient `open * rho0/rho_f` (WO-R2 item 1).
Here the outlet sits in PURE LIQUID, so choosing the reference density `set_rho(rho_l)` makes
`rho0/rho_f == 1` exactly at that face and the overwrite is the correct coefficient — the defect is
side-stepped by construction rather than tolerated.  Reported with every run.

THE EXACT SOLUTION (derived in the docstring of `exact_b`, checked against Welch & Wilson):

    X(t) = 2 b sqrt(alpha_l t) ,     b = Ja exp(-g^2) / (sqrt(pi) erfc(g)) ,  g = b rho_v/rho_l
    Ja   = rho_l c_pl (T_inf - T_sat) / (rho_v h_lv)
    T(x,t) = T_inf - (T_inf - T_sat) erfc(s)/erfc(g),  s = x/(2 sqrt(alpha_l t)) - b(1 - rho_v/rho_l)

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_sucking.py [--ratio 10] [--ns 64,128,256]
"""
import argparse
import math
import sys

import numpy as np

import peclet.flow as pf


def exact_b(ja, rr):
    """Solve b = Ja exp(-g^2)/(sqrt(pi) erfc(g)), g = b*rr,  rr = rho_v/rho_l.

    Derivation.  The vapour is isothermal at T_sat (it is bounded by a wall at T_sat), so it is at
    rest and mdot = rho_v Xdot.  The liquid moves at the uniform u_l = Xdot (1 - rho_v/rho_l), so in
    the Galilean frame x' = x - integral(u_l) the liquid temperature obeys the plain heat equation
    and is an erfc in x'/(2 sqrt(alpha_l t)).  Matching T(X) = T_sat fixes the amplitude and the
    Stefan condition mdot h_lv = k_l dT/dx|_X gives the fixed point above.
    """
    b = ja / math.sqrt(math.pi)
    for _ in range(200):
        g = b * rr
        b = ja * math.exp(-g * g) / (math.sqrt(math.pi) * math.erfc(g))
    return b


def run(n, ratio, ja, x0p=0.10, xep=0.25, alpha_l=1.0, cfl=0.2, fo=0.5, ny=4, nz=4,
        kratio=None, rcpratio=None, consistent=True, plane=True, quad=False, verbose=True):
    """One resolution.  The PHYSICAL problem is fixed (domain length 1, alpha_l = 1, the vapour
    layer growing from x0p to xep of the domain) and only h = 1/n changes."""
    rr = 1.0 / ratio                      # rho_v / rho_l
    b = exact_b(ja, rr)
    t0 = (x0p / (2 * b)) ** 2 / alpha_l
    te = (xep / (2 * b)) ** 2 / alpha_l

    # solver (cell) units: h = 1, so alpha in cells^2/s is alpha_l * n^2
    al = alpha_l * n * n
    rho_l, rho_v = 1.0, rr
    cpl = 1.0
    k_l = al * rho_l * cpl                                  # alpha_l = k/(rho c_p)
    rcp_l = rho_l * cpl
    rcp_v = rcp_l / (rcpratio if rcpratio else ratio)       # default: c_p equal, so rcp ~ rho
    k_v = k_l / (kratio if kratio else ratio)
    dT = 1.0
    h_lv = rho_l * cpl * dT / (rho_v * ja)

    X0 = x0p * n
    s = pf.Solver(n, ny, nz)
    s.set_rho(rho_l)          # reference density = the OUTLET phase; see the module docstring
    s.set_mu(1e-3)
    s.set_domain_bc(0, 1)     # -x wall (the vapour side)
    s.set_domain_bc(1, 3)     # +x outflow (the liquid leaves)
    s.set_pressure_geometry(np.full((n, ny, nz), 1.0, order="F"))
    s.enable_vof()
    c = np.zeros((n, ny, nz), order="F")
    for i in range(n):
        c[i, :, :] = min(1.0, max(0.0, (i + 1) - X0))       # liquid on the HIGH side
    s.set_vof(c)
    s.set_property_model("rho", "linear", "C", [rho_v, rho_l - rho_v])
    s.set_pressure_fcg(True, 4000, 1e-10)                   # AFTER the rho closure (CLAUDE.md)

    # temperature: T_sat = 0 at the wall, the similarity profile in the liquid
    t = np.zeros((n, ny, nz), order="F")
    for i in range(n):
        x = (i + 0.5) / n
        if x > x0p:
            sv = x / (2 * math.sqrt(alpha_l * t0)) - b * (1.0 - rr)
            t[i, :, :] = dT - dT * math.erfc(sv) / math.erfc(b * rr)
    # The far-field Dirichlet is the EXACT profile value at x = 1, refreshed every step. The domain
    # is finite and the similarity solution is not: at t_end the liquid thermal layer reaches
    # erfc(s) ~ 8e-3 at the outlet, so a fixed T = T_inf there is a 0.86 % TRUNCATION error that
    # does not converge under grid refinement (measured: |T - T_exact| flat at 0.838 % / 0.834 % at
    # N = 64 / 128 with the fixed value). Imposing the exact far-field value removes a
    # domain-truncation error, not a discretisation error.
    def farT(tt):
        sv = 1.0 / (2 * math.sqrt(alpha_l * tt)) - b * (1.0 - rr)
        return dT - dT * math.erfc(sv) / math.erfc(b * rr)

    s.add_scalar("T", al, 1, 60)
    s.set_scalar_bc("T", 0, 2, 0.0)       # saturated vapour against the wall
    s.set_scalar_bc("T", 1, 2, farT(t0))  # superheated far field (exact, refreshed per step)
    s.set_field("T", t)
    s.enable_phase_change(rho_v, rho_l, h_lv)
    s.set_phase_change_thermal("T", 0.0, k_v, k_l, 0.0)
    s.set_phase_change_plane_dirichlet(plane)
    s.set_phase_change_quadratic_fit(quad)
    if consistent:
        s.set_phase_change_energy(rcp_v, rcp_l)

    # initial liquid velocity (the projection would find it anyway; this removes a startup transient)
    Xdot = b * math.sqrt(alpha_l / t0) * n           # cells/s
    ul = Xdot * (1.0 - rr)
    u = np.zeros((n, ny, nz), order="F")
    u[int(X0) + 2:, :, :] = ul
    s.set_field("u", u)

    # dt from a FIXED diffusive number Fo = alpha dt/h^2 (so dt ~ h^2 and the backward-Euler /
    # explicit-mdot time error cannot contaminate a first- or second-order SPATIAL statement — the
    # same protocol as the P1 Stefan ladder), capped by the interface-local Courant number.
    tcur, nsteps, itmax, capped = t0, 0, 0, 0
    while tcur < te:
        Xd = b * math.sqrt(alpha_l / tcur) * n
        dt = min(cfl / max(Xd * (1.0 - rr), 1e-30), fo / al)
        dt = min(dt, (te - tcur))
        s.set_dt(dt)
        s.set_scalar_bc("T", 1, 2, farT(tcur + dt))
        s.step()
        tcur += dt
        nsteps += 1
        it = s.last_pressure_iterations()
        itmax = max(itmax, it)
        if it >= 4000:
            capped += 1
    layer = n - s.get_vof().sum() / (ny * nz)      # vapour thickness in cells
    exact = 2 * b * math.sqrt(alpha_l * te) * n
    d = s.phase_change_diagnostics()
    # temperature profile error against the similarity solution, over the LIQUID
    tnum = s.get_field("T")[:, 0, 0]
    terr, tn = 0.0, 0
    for i in range(n):
        x = (i + 0.5) / n
        xg = 2 * b * math.sqrt(alpha_l * te)
        if x > xg + 2.0 / n:
            sv = x / (2 * math.sqrt(alpha_l * te)) - b * (1.0 - rr)
            ex = dT - dT * math.erfc(sv) / math.erfc(b * rr)
            terr = max(terr, abs(tnum[i] - ex))
            tn += 1
    if verbose:
        print(f"  N = {n:4d}  steps {nsteps:5d}  b {b:.6f}  layer {layer:9.5f} cells "
              f"(exact {exact:9.5f})  rel {100*(layer-exact)/exact:+8.4f} %")
        print(f"           |T - T_exact|_inf over the liquid = {terr:.4e} ({100*terr/dT:.3f} % of "
              f"dT, {tn} cells)   C in [{d['min_C']:.2e}, {d['max_C']:.6f}]")
        print(f"           pressure iters max {itmax}/4000 (capped steps {capped}), "
              f"band_div {d['band_div']:.3e}, T in [{d['T_min']:.4e}, {d['T_max']:.6f}], "
              f"unresolved {d['unresolved']:.1e}, fallback {d['fallback_cells']}")
    return abs((layer - exact) / exact), terr / dT, capped, itmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--ja", type=float, default=1.0)
    ap.add_argument("--ns", type=str, default="64,128,256")
    ap.add_argument("--kratio", type=float, default=None)
    ap.add_argument("--rcpratio", type=float, default=None)
    ap.add_argument("--no-consistent", action="store_true")
    ap.add_argument("--no-plane", action="store_true")
    ap.add_argument("--no-quad", action="store_true")
    ap.add_argument("--fo", type=float, default=0.5)
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    print(f"P2 sucking interface: rho_l/rho_v = {a.ratio:g}, Ja = {a.ja:g}, "
          f"k ratio {a.kratio or a.ratio:g}, rho*c_p ratio {a.rcpratio or a.ratio:g}, "
          f"consistent energy {not a.no_consistent}, plane Dirichlet {not a.no_plane}, "
          f"quadratic fit {not a.no_quad}, Fo {a.fo:g}")
    errs, terrs = [], []
    for n in ns:
        e, te_, capped, itmax = run(n, a.ratio, a.ja, kratio=a.kratio, rcpratio=a.rcpratio,
                                    consistent=not a.no_consistent, plane=not a.no_plane,
                                    quad=not a.no_quad, fo=a.fo)
        errs.append(e)
        terrs.append(te_)
        if capped:
            print(f"  *** {capped} CAPPED pressure solves at N = {n}: rule 3b, run INVALID")
    for x, y, na, nb in zip(errs[:-1], errs[1:], ns[:-1], ns[1:]):
        print(f"  observed order {na} -> {nb}: {math.log2(x/y):.3f}")
    if len(errs) > 2:
        print(f"  observed order {ns[0]} -> {ns[-1]} (fit): "
              f"{math.log2(errs[0]/errs[-1])/math.log2(ns[-1]/ns[0]):.3f}")
    print(f"  GATE: order >= 1.4 and |T - T_exact| <= 1 % of dT at N = {ns[-1]} "
          f"(got {100*terrs[-1]:.3f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
