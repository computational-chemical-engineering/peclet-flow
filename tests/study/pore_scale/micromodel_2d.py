#!/usr/bin/env python3
"""WO-V7 case 3 — a ZHAO-2019-STYLE 2-D MICROMODEL: wettability and the invasion pattern.

A disordered array of ~56 SDF cylinders (a jittered lattice of posts, the geometry class Zhao,
MacMinn & Juanes, *PNAS* 116:13799 (2019) used) fills a quasi-2D cell that is gas-filled at t = 0.
Liquid is injected through the -x face at a fixed capillary number and leaves at +x.  The sweep is
over the contact angle only: theta = 45 (imbibition), 90 (neutral), 135 (drainage).

Zhao et al.'s qualitative result, which this page reproduces or fails to:
**wettability controls the pattern**.  Strong imbibition gives a COMPACT, high-saturation
displacement (cooperative pore filling, corner films run ahead); drainage gives a FINGERED,
low-saturation one (capillary fingering / invasion percolation).  The quantitative read-outs are
the saturation at breakthrough and the box-counting fractal dimension of the invaded region.

CONVENTIONS AND CAVEATS (WO-V7 preamble; stated on every run):
  * STATIC contact angle (V5b).  Ca is APPARENT: mu_l U_inlet / sigma.
  * density ratio 100; inflow/outflow (never a periodic net force); the posts are clear of both
    open faces; a capped pressure solve makes the run INVALID.
  * **corner films are NOT represented.**  VOF_PLAN 4/V7 names this as the lesson to design
    against: the strong-imbibition end of Zhao's benchmark is decided by films in the corners of
    the posts, which a 5.7-cell-radius post on a unit-spacing grid cannot resolve.  Expect the
    compact/fingered TREND, not the saturations.

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/pore_scale/micromodel_2d.py [--quick]
        [--theta 45,90,135] [--ca 1e-3] [--budget 1800] [--npy DIR]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import peclet.flow as pf

# ------------------------------------------------------------------------------------- the scene
NX, NY, NZ = 128, 128, 4
# The work order asks for ~60 posts at porosity ~0.6 on a 128^2 grid.  MEASURED: that array (56
# posts, R = 5.7, narrowest throat 3.08 cells -- a square array at porosity 0.6 has throats of
# 0.286 x the lattice spacing) ran to PV = 0.121 and then emitted
# `CutcellMG::solveFCG: preconditioner produced non-finite z` with max|u| at 93 x the inlet
# velocity.  The array below trades post COUNT for throat WIDTH -- 30 posts, narrowest throat 6.4
# cells, porosity 0.712 -- and survives further.  Two candidate mechanisms for the 3-cell failure,
# NOT separated by this campaign: (a) the theta-fill writes a three-cell band into the solid on
# each side of a throat, so at 3 cells the two posts' bands meet in the middle of it (the same
# overlap WO-S recorded as making its 4-cell-plate Jurin scene inconclusive); (b) a Haines jump
# through a throat whose meniscus radius is ~1.5 cells is simply unresolved -- the local velocity
# of a real pore-filling event runs up towards the capillary velocity sigma/mu_l = 25, which is
# 1000 x the inlet velocity here, and a wider throat resolves it better.  Separating them needs a
# theta-sweep at fixed throat width and is left for V9/V6b.
NCOL, NROW = 5, 6                    # 30 posts on a staggered, jittered lattice
X0, DX = 20.0, 21.5                  # first post column centre and the column spacing
DY = NY / NROW                       # 21.33
R_POST = 6.5                         # -> pi R^2 / (DX*DY) = 0.289 solid, porosity ~0.71
JITTER = 1.5
MIN_GAP = 6.0                        # surface-to-surface, enforced by rejection: TWO wetting
                                     # bands of three cells each must fit inside a throat
X_IN = 10.0                          # prefill: liquid up to here at t = 0 (clear of every post)
X_BT = 118.0                         # breakthrough plane

SIGMA = 100.0
RHO_L, RHO_G = 100.0, 1.0
MU_L, MU_G = 4.0, 0.04
CAP_CFL = 0.5
PRESS_CAP = 400
PRESS_RTOL = 1e-8
MG_LEVELS = 5
VEL_SWEEPS = 20


def posts(seed=3):
    """A staggered lattice of post centres with bounded jitter, rejected until every pair is at
    least MIN_GAP apart at the surfaces (y periodic)."""
    rng = np.random.default_rng(seed)
    base = []
    for i in range(NCOL):
        for j in range(NROW):
            base.append((X0 + i * DX, (j + 0.5 * (i % 2)) * DY))
    base = np.array(base)
    for _ in range(20000):
        P = base + rng.uniform(-JITTER, JITTER, base.shape)
        P[:, 1] %= NY
        d = P[None, :, :] - P[:, None, :]
        d[:, :, 1] -= NY * np.round(d[:, :, 1] / NY)
        r = np.sqrt((d ** 2).sum(-1))
        np.fill_diagonal(r, 1e9)
        if (r - 2 * R_POST).min() >= MIN_GAP:
            return P, float((r - 2 * R_POST).min()), float(np.sort((r - 2*R_POST).ravel())[:len(P)*3].mean())
    raise RuntimeError("no admissible jittered lattice found — relax MIN_GAP")


def post_sdf(P):
    x = (np.arange(NX) + 0.5)[:, None]
    y = (np.arange(NY) + 0.5)[None, :]
    phi = np.full((NX, NY), 1e30)
    for cx, cy in P:
        dy = y - cy
        dy -= NY * np.round(dy / NY)
        phi = np.minimum(phi, np.sqrt((x - cx) ** 2 + dy ** 2) - R_POST)
    return np.asfortranarray(np.broadcast_to(phi[:, :, None], (NX, NY, NZ)).astype(float).copy())


class Health:
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


def box_dimension(mask, sizes=(1, 2, 4, 8, 16, 32)):
    """Box-counting dimension of a 2-D boolean set; returns (D, [(eps, N)])."""
    pts = []
    nx, ny = mask.shape
    for e in sizes:
        mx, my = (nx // e) * e, (ny // e) * e
        blk = mask[:mx, :my].reshape(mx // e, e, my // e, e).any(axis=(1, 3))
        n = int(blk.sum())
        if n > 0:
            pts.append((e, n))
    if len(pts) < 3:
        return float("nan"), pts
    x = np.log(np.array([1.0 / p[0] for p in pts]))
    y = np.log(np.array([float(p[1]) for p in pts]))
    return float(np.polyfit(x, y, 1)[0]), pts


def front_stats(inv):
    """Roughness of the invasion front, the discriminator Zhao's experiment is really about.

    `inv` is the boolean invaded set of the array region in the mid-plane, indexed (x, y).  For
    every transverse row y that the liquid reached, the front is the furthest x it reached; the
    MEAN of that is how far the displacement has got and its STANDARD DEVIATION is how ragged it
    is.  A compact (cooperative-filling) displacement has a small std and reaches every row; a
    fingered one has a large std, a large max/mean ratio, and leaves rows untouched.
    """
    from scipy import ndimage
    reach = np.array([np.max(np.nonzero(inv[:, j])[0]) if inv[:, j].any() else -1
                      for j in range(inv.shape[1])])
    ok = reach >= 0
    lab, n = ndimage.label(inv)
    return dict(cells=int(inv.sum()), rows=int(ok.sum()), nrows=int(inv.shape[1]),
                mean=float(reach[ok].mean()) if ok.any() else 0.0,
                std=float(reach[ok].std()) if ok.any() else 0.0,
                max=int(reach.max()), clusters=int(n))


def run(theta_deg, ca, sdf, steps, budget=None, npy_dir=""):
    U = ca * SIGMA / MU_L
    Re = RHO_L * U * 2 * R_POST / MU_L
    print("\n" + "=" * 100)
    print(f"MICROMODEL   theta = {theta_deg:g} deg   Ca(apparent) = {ca:g}")
    print("=" * 100)
    print(f"  grid {NX}x{NY}x{NZ} (quasi-2D, periodic in y and z); inlet U = {U:.6g} cells/s "
          f"-> post Reynolds {Re:.4g}")

    s = pf.Solver(NX, NY, NZ)
    s.set_rho(RHO_L)
    s.set_mu(MU_L)
    s.set_dt(0.05)
    s.set_domain_bc(0, 2, U, 0.0, 0.0)
    s.set_domain_bc(1, 3, 0.0, 0.0, 0.0)
    s.set_velocity_solver_params(VEL_SWEEPS)
    s.set_pressure_multigrid(True, levels=MG_LEVELS)
    s.set_pressure_solver_params(80)
    s.set_solid(sdf, cutcell_pressure=True)
    s.enable_vof()
    c0 = np.zeros((NX, NY, NZ), order="F")
    c0[: int(X_IN), :, :] = 1.0
    s.set_vof(c0)
    s.set_property_model("rho", "linear", "C", [RHO_G, RHO_L - RHO_G])
    s.set_property_model("mu", "linear", "C", [MU_G, MU_L - MU_G])
    s.enable_vof_momentum(RHO_G, RHO_L)
    s.set_surface_tension(SIGMA)
    s.set_capillary_cfl(CAP_CFL)
    s.set_contact_angle(theta_deg)
    s.set_pressure_fcg(True, PRESS_CAP, PRESS_RTOL)     # driver LAST
    s.set_vof_inflow(0, 1.0)
    s.set_vof_backflow(1, 0.0)

    eps = np.asarray(s.vof_geometry(0))
    arr = slice(int(X_IN), int(X_BT))
    pore = float(eps[arr].sum())
    L0 = s.vof_step_limits()
    print(f"  dt census at t = 0: capillary_dt {L0['capillary_dt']:.5g} (x{CAP_CFL}), "
          f"WY cfl_dt {L0['cfl_dt']:.5g} -> "
          f"{'CAPILLARY' if L0['capillary_binds'] else 'CFL'} binds "
          f"(capillary_dt is state-dependent; dt is re-picked every step)")

    h = Health()
    hist = []
    bt = None
    ncfl = 0
    t = 0.0
    n = 0
    every = max(steps // 600, 5)
    t0 = time.time()
    partial = False
    Cbt = None
    # the injected liquid volume in PORE VOLUMES of the array: the comparable clock when a run
    # cannot afford to reach breakthrough
    pv_stop = 0.10
    for i in range(steps):
        L = s.vof_step_limits()
        dtc = CAP_CFL * L["capillary_dt"]
        dt = min(dtc, 0.8 * L["cfl_dt"]) if L["cfl_dt"] > 0 else dtc
        ncfl += int(dt < dtc * 0.999)
        s.set_dt(dt)
        s.step()
        t += dt
        n += 1
        h.sample(s)
        if (i + 1) % every == 0 or i == steps - 1:
            C = np.asarray(s.get_vof())
            S = float((C[arr] * eps[arr]).sum() / pore)
            front = float(C[int(X_BT) - 1, :, :].max())
            pv = U * NY * NZ * t / pore
            hist.append((t, S, front, pv))
            if len(hist) % 5 == 1:
                um = max(float(np.abs(np.asarray(f())).max())
                         for f in (s.get_u, s.get_v, s.get_w))
                print(f"    step {i+1:7d}  t {t:10.4g}  dt {dt:9.3g}  PV {pv:.3f}  S {S:.4f}  "
                      f"front {front:.3f}  max|u| {um:9.2e}  ({time.time()-t0:.0f} s, "
                      f"{1000*(time.time()-t0)/(i+1):.0f} ms/step)", flush=True)
            if bt is None and front > 0.5:
                bt = (t, i + 1, S)
                Cbt = C.copy()
                print(f"    breakthrough at t = {t:.5g} (step {i+1}), array saturation "
                      f"S_bt = {S:.4f}")
                break
            if pv >= pv_stop:
                partial = True
                print(f"    *** {pv_stop:.2f} pore volumes injected at step {i+1} "
                      f"(t = {t:.5g}, S = {S:.4f}) without breakthrough - stopping here so the "
                      f"three angles are compared at the SAME injected volume ***")
                break
            if budget and time.time() - t0 > budget:
                partial = True
                print(f"    *** wall-clock budget {budget:.0f} s reached at step {i+1} "
                      f"(t = {t:.5g}, S = {S:.4f}) — PARTIAL, no breakthrough ***")
                break
    wall = time.time() - t0
    C = np.asarray(s.get_vof()) if Cbt is None else Cbt
    S = float((C[arr] * eps[arr]).sum() / pore)
    mid = NZ // 2
    fluid2 = eps[:, :, mid] > 0.0
    inv = (C[:, :, mid] > 0.5) & fluid2
    D, pts = box_dimension(inv[arr])
    fs = front_stats(inv[arr])
    frac_pore_area = float((eps[arr][:, :, mid] > 0).mean())
    print(f"\n  ran {n} steps to t = {t:.5g} s in {wall:.0f} s "
          f"({1000*wall/max(n,1):.2f} ms/step, SHARED GPU)")
    print(f"  dt final {dt:.5g}; the WY CFL limit displaced the capillary one on {ncfl} of "
          f"{n} steps ({100.0*ncfl/max(n,1):.1f} %)")
    print(f"  injected liquid: {U*NY*NZ*t/pore:.4f} pore volumes of the array")
    print(f"  saturation of the post array: {S:.4f}" + ("  (AT BREAKTHROUGH)" if bt else
                                                        "  (partial run, no breakthrough)"))
    print(f"  box-counting dimension of the invaded region in the mid-plane: D = {D:.4f}")
    print("    " + "  ".join(f"eps={e}:N={m}" for e, m in pts))
    print(f"    (a compact invasion of this pore space would read D -> 2; the pore area itself "
          f"is {frac_pore_area:.3f} of the array)")
    print(f"  front roughness: reached {fs['rows']}/{fs['nrows']} transverse rows; front position "
          f"mean {fs['mean']:.2f} cells, std {fs['std']:.2f}, deepest finger {fs['max']}; "
          f"{fs['clusters']} connected invaded cluster(s)")
    print("    (a COMPACT displacement reaches every row with a small std; a FINGERED one has a "
          "large std, a deep max and leaves rows untouched)")
    d = s.vof_diagnostics()
    print(f"  colour: solid_sum {d['solid_sum']:.3e}, min/max over uncut fluid "
          f"{d['min_fluid']:.3e}/{d['max_fluid']:.6f}, clipped {d['clipped_volume']:.3e}")
    print(f"  {h}")
    if npy_dir:
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, f"micromodel_C_theta{int(theta_deg)}.npy"), C[:, :, mid])
        np.save(os.path.join(npy_dir, "micromodel_eps.npy"), eps[:, :, mid])
    return dict(theta=theta_deg, ca=ca, U=U, Re=Re, steps=n, t=t, wall=wall,
                ms_per_step=1000 * wall / max(n, 1),
                t_bt=bt[0] if bt else None, S_bt=bt[2] if bt else None, S=S, D=D,
                boxes=pts, front=fs, iters=h.iters, capped=h.capped, div=h.div, valid=h.valid,
                pv=U * NY * NZ * t / pore,
                partial=partial, cfl_fraction=ncfl / max(n, 1), hist=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--theta", default="45,90,135")
    ap.add_argument("--ca", default="1e-3")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--npy", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    print("WO-V7 case 3 — a Zhao-2019-style 2-D micromodel.  Static contact angle (V5b); the")
    print("capillary number is APPARENT (mu_l U_inlet / sigma).")
    P, gmin, gmean = posts()
    sdf = post_sdf(P)
    por = float((sdf[:, :, 0][int(X_IN):int(X_BT)] > 0).mean())
    print(f"\n  {len(P)} posts of radius {R_POST} on a staggered jittered lattice "
          f"(spacing {DX:g} x {DY:g}, jitter +-{JITTER})")
    print(f"  porosity of the array region x = {X_IN:g} .. {X_BT:g}: {por:.4f}")
    print(f"  narrowest throat (surface-to-surface) {gmin:.3f} cells; mean of the tightest "
          f"{3*len(P)} pair gaps {gmean:.3f} cells")
    print(f"  the meniscus radius in a throat of half-width {gmin/2:.2f} cells is at or below the "
          f"~2.5-cell floor where the height-function cascade always falls back to the")
    print("  PLIC-volumetric paraboloid fit (V3 finding) — read the pattern, not the curvature")
    rows = []
    for th in [float(x) for x in a.theta.split(",")]:
        for ca in [float(x) for x in a.ca.split(",")]:
            U = ca * SIGMA / MU_L
            ttrav = (X_BT - X_IN) / (U / max(por, 0.2))
            dt0 = CAP_CFL * math.sqrt((RHO_L + RHO_G) / (4 * math.pi * SIGMA))
            steps = int(3.0 * ttrav / dt0)
            if a.quick:
                steps = min(steps, 600)
            rows.append(run(th, ca, sdf, steps, budget=a.budget or None, npy_dir=a.npy))
    print("\n" + "=" * 100)
    print("SUMMARY — micromodel")
    print("=" * 100)
    print(f"{'theta':>6} {'Ca':>8} {'t_bt':>10} {'S_bt':>8} {'D_box':>8} {'front std':>10} "
          f"{'iters':>6} {'ms/step':>8} {'valid':>6}")
    for r in rows:
        tb = f"{r['t_bt']:.4g}" if r["t_bt"] else "-"
        sb = f"{r['S_bt']:.4f}" if r["S_bt"] else f"({r['S']:.4f})"
        print(f"{r['theta']:6g} {r['ca']:8g} {tb:>10} {sb:>8} {r['D']:8.4f} "
              f"{r['front']['std']:10.2f} {r['iters']:6d} "
              f"{r['ms_per_step']:8.2f} {'yes' if r['valid'] else 'NO':>6}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\n  wrote {a.out}")
    return 0 if all(r["valid"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
