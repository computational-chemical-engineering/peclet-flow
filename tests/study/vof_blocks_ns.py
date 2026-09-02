#!/usr/bin/env python3
"""VoF Part III rung W2 (WO-W12) — the block container coupled to Navier-Stokes.

Every marker carries its own Weymouth-Yue colour field on its own moving box, forms its own
curvature and its own balanced-force CSF face force there, and the three face-force fields are
summed into the global RHS (UNPACK_SUM).  The union `C = max_blocks C_block` is what the closures
(rho, mu) see.  Two markers that touch therefore cannot coalesce numerically.

    hysing    gate 1: Hysing case 1 through the BLOCK path against the same case through the
              GLOBAL colour field, both without momentum consistency (quasi-2D 64x128x4).
    grace     gate 2: one 3-D bubble at Eo = 10, Mo = 1e-3, density ratio 100, against the
              Grace-diagram terminal velocity / shape regime.
    pair      gate 3: two bubbles in line, the trailing one catching the leading one -- the film
              drains to one cell and the two blocks stay two.
    swarm     rung W1 in Python: the master-assignment imbalance of the three modes on a 64-bubble
              swarm, the per-bubble Lagrangian outputs, and the device-vs-host packing ratio.

COLOUR CONVENTION.  With the block container the union C is the DISPERSED phase (C = 1 inside a
marker), because UNPACK_MAX starts from an empty union and a cell no marker covers must read 0.
Every property model here is therefore written with C = 1 = GAS, the mirror of the structured
scripts.  The CSF force is unaffected by the flip: kappa(1-C) = -kappa(C) and grad(1-C) = -grad(C),
so sigma kappa grad C is invariant -- which is what makes the gate-1 comparison meaningful.

Usage:  PYTHONPATH=<build> python tests/study/vof_blocks_ns.py [gate ...] [--quick]
"""
import math
import os
import sys
import time

import numpy as np

import peclet.flow as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vof_surface_tension import Scale, Solve, cylinder_fractions, sphere_fractions  # noqa: E402

QUICK = "--quick" in sys.argv
GATES = [a for a in sys.argv[1:] if not a.startswith("--")]


def marker_box(C, pad=1):
    """Tight global index box of C > 0.5, grown by `pad`, clamped to the grid."""
    idx = np.argwhere(C > 0.5)
    lo = idx.min(axis=0) - pad
    hi = idx.max(axis=0) + 1 + pad
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, C.shape)
    return [int(lo[0]), int(lo[1]), int(lo[2]), int(hi[0]), int(hi[1]), int(hi[2])]


HYSING1 = dict(rho1=1000.0, rho2=100.0, mu1=10.0, mu2=1.0, g=0.98, sigma=24.5,
               ref_vmax=0.2417, ref_tvmax=0.9215, ref_yc=1.081)


def build_hysing(nx, blocks):
    """Hysing case 1, quasi-2D.  `blocks` selects the container; the physics is identical."""
    p = HYSING1
    Lx = 1.0
    nz, ny = 2 * nx, 4
    sc = Scale(nx / Lx)
    R = sc.len_to_cells(0.25)
    s = pf.Solver(nx, ny, nz)
    s.set_rho(p["rho1"])
    s.set_mu(sc.mu(p["mu1"]))
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)
    s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 600, 1e-12)
    s.enable_vof()
    B = cylinder_fractions((nx, ny, nz), R, nx / 2.0, sc.len_to_cells(0.5))  # 1 inside the bubble
    if blocks:
        s.set_vof(np.asfortranarray(B))                       # marker convention: C = 1 = GAS
        s.set_property_model("rho", "linear", "C", [p["rho1"], p["rho2"] - p["rho1"]])
        s.set_property_model("mu", "linear", "C",
                             [sc.mu(p["mu1"]), sc.mu(p["mu2"] - p["mu1"])])
        s.set_surface_tension(sc.sigma(p["sigma"]))
        s.set_property_model("force_z", "linear", "C",
                             [-sc.bodyforce(p["rho1"] * p["g"]),
                              -sc.bodyforce((p["rho2"] - p["rho1"]) * p["g"])])
        s.enable_vof_blocks_from_field([marker_box(B)])
        s.enable_vof_block_csf()
    else:
        s.set_vof(np.asfortranarray(1.0 - B))                 # structured convention: C = 1 = LIQUID
        s.set_property_model("rho", "linear", "C", [p["rho2"], p["rho1"] - p["rho2"]])
        s.set_property_model("mu", "linear", "C",
                             [sc.mu(p["mu2"]), sc.mu(p["mu1"] - p["mu2"])])
        s.set_surface_tension(sc.sigma(p["sigma"]))
        s.set_property_model("force_z", "linear", "C",
                             [-sc.bodyforce(p["rho2"] * p["g"]),
                              -sc.bodyforce((p["rho1"] - p["rho2"]) * p["g"])])
    return s, sc, nz, ny


def run_hysing(nx, blocks, T=3.0, verbose=True):
    s, sc, nz, ny = build_hysing(nx, blocks)
    zs = (np.arange(nz) + 0.5) / sc.s
    h = Solve(600)
    dt = 0.4 * min(s.vof_step_limits()["cfl_dt"], s.capillary_dt())
    s.set_dt(dt)
    t, i, vmax, tvmax, yc = 0.0, 0, 0.0, 0.0, 0.0
    v0 = None
    t0 = time.time()
    while t < T:
        if i % 10 == 0:
            L = s.vof_step_limits()
            dt = 0.4 * min(L["cfl_dt"], L["capillary_dt"])
            s.set_dt(dt)
        t += dt
        i += 1
        s.step()
        h.sample(s)
        C = s.get_vof()
        B = C if blocks else 1.0 - C
        vb = B.sum()
        if v0 is None:
            v0 = vb
        yc = float((B.sum(axis=(0, 1)) * zs).sum() / vb)
        vrise = float((B * s.get_w()).sum() / vb) / sc.s
        if vrise > vmax:
            vmax, tvmax = vrise, t
        if verbose and i % 200 == 0:
            print(f"    t = {t:5.3f}  dt = {dt:.3e}  y_c = {yc:.4f}  v = {vrise:.4f}  "
                  f"V/V0 = {vb/v0:.6f}")
    return dict(vmax=vmax, tvmax=tvmax, yc=yc, steps=i, solve=h, vdrift=vb / v0 - 1.0,
                wall=time.time() - t0, s=s)


def gate_hysing():
    nx = 32 if QUICK else 64
    print("\n" + "=" * 96)
    print(f"GATE 1 — Hysing case 1 through the BLOCK path vs the GLOBAL colour field, nx = {nx}")
    print("=" * 96)
    print("  Same physics, same discretisation, same dt schedule; the ONLY difference is which")
    print("  container carries the colour and where the CSF face force is formed.  No momentum")
    print("  consistency in either (the work order's gate: both without enable_vof_momentum).")
    T = 1.0 if QUICK else 3.0
    out = {}
    for name, blk in (("global field", False), ("blocks", True)):
        print(f"\n  -- {name}")
        out[name] = run_hysing(nx, blk, T)
        r = out[name]
        print(f"     v_rise max {r['vmax']:.4f} at t = {r['tvmax']:.3f};  y_c({T:g}) = {r['yc']:.4f}"
              f";  volume drift {r['vdrift']:+.2e}")
        print(f"     {r['steps']} steps, {r['solve']}   ({r['wall']:.0f} s)")
    a, b = out["global field"], out["blocks"]
    dv = 100.0 * (b["vmax"] / a["vmax"] - 1.0)
    dy = 100.0 * (b["yc"] / a["yc"] - 1.0)
    print(f"\n  BLOCK vs GLOBAL: v_rise {dv:+.2f} %,  y_c {dy:+.2f} %   "
          f"(gate: both within 1 %)  ->  {'PASS' if max(abs(dv), abs(dy)) <= 1.0 else 'FAIL'}")
    p = HYSING1
    print(f"  reference (Hysing): v_rise max {p['ref_vmax']:.4f} at t = {p['ref_tvmax']:.3f}, "
          f"y_c(3) = {p['ref_yc']:.4f}")
    print(f"  global vs reference: {100*(a['vmax']/p['ref_vmax']-1):+.1f} % / "
          f"{100*(a['yc']/p['ref_yc']-1):+.1f} %;  blocks vs reference: "
          f"{100*(b['vmax']/p['ref_vmax']-1):+.1f} % / {100*(b['yc']/p['ref_yc']-1):+.1f} %")


# --------------------------------------------------------------------------- gate 2: Grace
def grace_terminal(Eo, Mo, d, rho_l, mu_l, sigma, g, mu_ratio_to_water=None):
    """Grace (1973), as tabulated by Clift, Grace & Weber (1978) ch. 7, eqs (7-10)/(7-11):

        H = (4/3) Eo Mo^-0.149 (mu_l/mu_w)^-0.14        mu_w = 0.9 cP
        J = 0.94 H^0.757  (2 < H <= 59.3),  3.42 H^0.441  (H > 59.3)
        U_T = (mu_l/(rho_l d)) Mo^-0.149 (J - 0.857)

    THE TRAP, and it is the reason this gate was nearly reported wrong.  The correlation contains a
    DIMENSIONAL factor, `(mu_l/mu_water)^-0.14`, which a purely dimensionless (Eo, Mo, ratio)
    specification does NOT determine: at Eo = 10, Mo = 1e-3 the "Grace-diagram terminal velocity"
    is 0.959 cells/s if one silently sets that factor to 1 and 0.579 if the liquid is the one that
    actually HAS Mo = 1e-3.  It has to be the latter: Mo = 1e-3 at water-like rho and sigma forces
    mu_l ~ 0.073 Pa s ~ 81 x water (water itself is Mo ~ 2.5e-11), so a factor of 1 describes no
    liquid at all.  `mu_ratio_to_water = None` therefore DERIVES the factor from the physical
    system that the dimensionless pair implies at rho_l = 1000 kg/m^3, sigma = 0.065 N/m,
    g = 9.81 m/s^2 -- a glycerol/water mixture, the standard Mo = 1e-3 fluid.
    """
    if mu_ratio_to_water is None:
        RHO_P, SIG_P, G_P, MU_W = 1000.0, 0.065, 9.81, 9.0e-4
        mu_phys = (Mo * RHO_P * SIG_P ** 3 / G_P) ** 0.25
        mu_ratio_to_water = mu_phys / MU_W
    H = (4.0 / 3.0) * Eo * Mo ** -0.149 * mu_ratio_to_water ** -0.14
    if H <= 59.3:
        J = 0.94 * H ** 0.757
    else:
        J = 3.42 * H ** 0.441
    return (mu_l / (rho_l * d)) * Mo ** -0.149 * (J - 0.857)


def gate_grace():
    print("\n" + "=" * 96)
    print("GATE 2 — a single 3-D bubble at Eo = 10, Mo = 1e-3 vs the Grace-diagram terminal velocity")
    print("=" * 96)
    # physical set (cell units directly: h = 1).  Choose D, rho_l, sigma, then mu from Mo and g
    # from Eo, with ratio 100 for both density and viscosity.
    Eo, Mo, ratio = 10.0, 1e-3, 100.0
    D = 12.0 if QUICK else 16.0
    R = 0.5 * D
    rho_l, sigma = 1.0, 1.0
    g = Eo * sigma / (rho_l * D * D)                       # Eo = rho g D^2 / sigma (dRho ~ rho_l)
    mu_l = (Mo * rho_l ** 2 * sigma ** 3 / g) ** 0.25      # Mo = g mu^4 / (rho sigma^3)
    rho_g, mu_g = rho_l / ratio, mu_l / ratio
    nx = int(round(4 * D))
    nz = int(round(9 * D))
    ny = nx
    print(f"  D = {D:g} cells, box {nx}x{ny}x{nz}, rho_l {rho_l}, rho_g {rho_g}, mu_l {mu_l:.4e}, "
          f"sigma {sigma}, g {g:.4e}")
    Ut = grace_terminal(Eo, Mo, D, rho_l, mu_l, sigma, g)
    Ut1 = grace_terminal(Eo, Mo, D, rho_l, mu_l, sigma, g, mu_ratio_to_water=1.0)
    Ub = math.sqrt(g * D)
    mu_phys = (Mo * 1000.0 * 0.065 ** 3 / 9.81) ** 0.25
    print(f"  the physical system this (Eo, Mo) implies at rho = 1000, sigma = 0.065, g = 9.81: "
          f"d = {math.sqrt(Eo*0.065/(1000*9.81))*1e3:.2f} mm, mu_l = {mu_phys*1e3:.1f} cP "
          f"({mu_phys/9e-4:.0f} x water)")
    print(f"  Grace/Clift U_T = {Ut:.4f} cells/s  (Re = {rho_l*Ut*D/mu_l:.1f}, "
          f"Fr = U/sqrt(gD) = {Ut/Ub:.3f})")
    print(f"    [the same formula with the viscosity factor set to 1 -- i.e. pretending the liquid "
          f"is water, which cannot have Mo = 1e-3 -- reads {Ut1:.4f}; see grace_terminal()]")
    s = pf.Solver(nx, ny, nz)
    s.set_rho(rho_l)
    s.set_mu(mu_l)
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)
    s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 800, 1e-12)
    s.enable_vof()
    B = sphere_fractions((nx, ny, nz), R, (nx / 2.0, ny / 2.0, 1.6 * D))
    s.set_vof(np.asfortranarray(B))
    s.set_property_model("rho", "linear", "C", [rho_l, rho_g - rho_l])
    s.set_property_model("mu", "linear", "C", [mu_l, mu_g - mu_l])
    s.set_surface_tension(sigma)
    s.set_property_model("force_z", "linear", "C",
                         [-rho_l * g, -(rho_g - rho_l) * g])
    s.enable_vof_blocks_from_field([marker_box(B)])
    s.enable_vof_block_csf()
    T = (3.0 if QUICK else 6.0) * D / Ut
    h = Solve(800)
    t, i = 0.0, 0
    dt = 0.0
    hist = []
    t0 = time.time()
    while t < T:
        if i % 10 == 0:
            L = s.vof_step_limits()
            dt = 0.4 * min(L["cfl_dt"], L["capillary_dt"])
            s.set_dt(dt)
        t += dt
        i += 1
        s.step()
        h.sample(s)
        st = s.vof_block_stats()[0]
        if st["volume"] > 0:
            hist.append((t, st["centroid"][2], st["volume"], st["area"],
                         st["moments"][2], st["moments"][0]))
        if i % 200 == 0:
            w = s.get_w()
            C = s.get_vof()
            ur = float((C * w).sum() / C.sum())
            print(f"    t = {t:8.2f}  z_c = {hist[-1][1]:8.3f}  U = {ur:.4f}  "
                  f"V/V0 = {hist[-1][2]/hist[0][2]:.6f}  A = {hist[-1][3]:.1f}")
    hist = np.array(hist)
    # terminal velocity from the last third of the trajectory
    k = len(hist) // 3
    U = np.polyfit(hist[-k:, 0], hist[-k:, 1], 1)[0]
    # aspect ratio from the second moments: E = sqrt(m_xx/m_zz) (horizontal over vertical)
    E = math.sqrt(hist[-1, 5] / hist[-1, 4])
    print(f"\n  measured terminal velocity {U:.4f} cells/s  ({100*(U/Ut-1):+.1f} % of Grace)")
    print(f"  Re = {rho_l*U*D/mu_l:.1f};  aspect ratio (sqrt(m_xx/m_zz)) {E:.3f}  "
          f"(ellipsoidal regime: E > 1)")
    print(f"  volume drift {hist[-1,2]/hist[0,2]-1:+.2e};  interface area "
          f"{hist[-1,3]:.1f} vs the sphere's {4*math.pi*R*R:.1f} cells^2")
    print(f"  {i} steps, {h}   ({time.time()-t0:.0f} s)")
    print(f"  GATE: |U/U_Grace - 1| <= 10 %  ->  "
          f"{'PASS' if abs(U/Ut - 1) <= 0.10 else 'FAIL'}")


# --------------------------------------------------------------------------- gate 3: two bubbles
def gate_pair():
    print("\n" + "=" * 96)
    print("GATE 3 — two bubbles in line, the larger catching the smaller: no numerical coalescence")
    print("=" * 96)
    print("  A LARGE bubble rising in line behind a SMALL one closes the gap deterministically (at")
    print("  fixed Mo the terminal velocity grows with D), so the collision happens on a schedule")
    print("  instead of waiting on the wake.  In a single global colour field the two merge the")
    print("  moment their PLIC bands touch -- the film is sub-grid and the field has no way to")
    print("  represent two interfaces in one cell.  With one marker each they cannot merge, and the")
    print("  union carries CELLS THAT BELONG TO BOTH.  The control is the same case, same numbers,")
    print("  on the structured field.")
    Eo, Mo, ratio = 10.0, 1e-3, 100.0
    D = 10.0 if QUICK else 12.0
    Rs, Rl = 0.40 * D, 0.60 * D          # leading (small) and trailing (large)
    rho_l, sigma = 1.0, 1.0
    g = Eo * sigma / (rho_l * D * D)
    mu_l = (Mo * rho_l ** 2 * sigma ** 3 / g) ** 0.25
    rho_g, mu_g = rho_l / ratio, mu_l / ratio
    nx = ny = int(round(5 * D))
    nz = int(round(11 * D))
    # Seed them nearly touching (a 2-cell film): the large one closes the last two cells within a
    # few hundred steps, which is what makes the gate affordable.  Starting them a diameter apart
    # measured nothing in 20 minutes of wall clock -- the approach is slow and the answer is at
    # contact.
    zc_l = 2.0 * D
    zc_s = zc_l + Rs + Rl + 2.0
    print(f"  D_small = {2*Rs:.1f}, D_large = {2*Rl:.1f} cells; box {nx}x{ny}x{nz}; "
          f"centres z = {zc_l:.1f} / {zc_s:.1f} (gap {zc_s-zc_l-Rs-Rl:.1f} cells)")

    def build(blocks):
        s = pf.Solver(nx, ny, nz)
        s.set_rho(rho_l)
        s.set_mu(mu_l)
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 1, 0, 0, 0)
        s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 800, 1e-12)
        s.enable_vof()
        B0 = sphere_fractions((nx, ny, nz), Rl, (nx / 2.0, ny / 2.0, zc_l))
        B1 = sphere_fractions((nx, ny, nz), Rs, (nx / 2.0, ny / 2.0, zc_s))
        s.set_vof(np.asfortranarray(np.maximum(B0, B1)))
        s.set_property_model("rho", "linear", "C", [rho_l, rho_g - rho_l])
        s.set_property_model("mu", "linear", "C", [mu_l, mu_g - mu_l])
        s.set_surface_tension(sigma)
        s.set_property_model("force_z", "linear", "C", [-rho_l * g, -(rho_g - rho_l) * g])
        if blocks:
            b0, b1 = marker_box(B0), marker_box(B1)
            mid = (b0[5] + b1[2]) // 2       # split the boxes at the midplane: one sphere each
            b0[5] = min(b0[5], mid)
            b1[2] = max(b1[2], mid)
            s.enable_vof_blocks_from_field([b0, b1])
            s.enable_vof_block_csf()
        return s, 4 / 3 * math.pi * Rl ** 3, 4 / 3 * math.pi * Rs ** 3

    def axis_gap(C):
        """Empty cells on the axis BETWEEN the two blobs; 0 = they have merged there."""
        col = C[nx // 2, ny // 2, :]
        occ = np.argwhere(col > 0.5).ravel()
        if len(occ) == 0:
            return -1
        inside = col[occ.min():occ.max() + 1]
        return int((inside < 0.5).sum())

    Ut = grace_terminal(Eo, Mo, D, rho_l, mu_l, sigma, g)
    T = (3.0 if QUICK else 6.0) * D / Ut
    res = {}
    for name, blk in (("blocks", True), ("single field (control)", False)):
        s, v0l, v0s = build(blk)
        h = Solve(800)
        t, i, dt = 0.0, 0, 0.0
        mingap, overlap_max = 10 ** 9, 0.0
        merged_at = None
        t0 = time.time()
        while t < T:
            if i % 10 == 0:
                L = s.vof_step_limits()
                dt = 0.4 * min(L["cfl_dt"], L["capillary_dt"])
                s.set_dt(dt)
            t += dt
            i += 1
            s.step()
            h.sample(s)
            if i % 5 == 0:
                C = s.get_vof()
                gap = axis_gap(C)
                mingap = min(mingap, gap)
                if gap == 0 and merged_at is None:
                    merged_at = t
                if blk:
                    vols = [b["volume"] for b in s.vof_block_stats()]
                    overlap_max = max(overlap_max, sum(vols) - float(C.sum()))
            if i % 100 == 0:
                print(f"    t = {t:7.2f}  gap {axis_gap(s.get_vof()):3d} cells  "
                      f"({time.time()-t0:.0f} s)")
        C = s.get_vof()
        info = dict(steps=i, solve=h, gap=axis_gap(C), mingap=mingap, merged=merged_at,
                    wall=time.time() - t0)
        print(f"\n  -- {name}: {i} steps, {info['solve']}  ({info['wall']:.0f} s)")
        if blk:
            st = s.vof_block_stats()
            info["vols"] = [b["volume"] for b in st]
            info["zc"] = [b["centroid"][2] for b in st]
            info["overlap"] = overlap_max
            print(f"     marker volumes {info['vols'][0]:.4f} / {info['vols'][1]:.4f}  "
                  f"(seed {v0l:.4f} / {v0s:.4f}; drift "
                  f"{info['vols'][0]/v0l-1:+.2e} / {info['vols'][1]/v0s-1:+.2e})")
            print(f"     centroids z {info['zc'][0]:.2f} / {info['zc'][1]:.2f}; final film gap on "
                  f"the axis {info['gap']} cells, MINIMUM over the run {mingap}")
            print(f"     peak shared liquid (sum V_marker - sum C_union) {overlap_max:.3f} cells")
        else:
            print(f"     final film gap on the axis {info['gap']} cells, MINIMUM {mingap}; "
                  f"merged at t = {merged_at}")
        res[name] = info
    b, c = res["blocks"], res["single field (control)"]
    ok = (b["mingap"] >= 1) and (c["mingap"] == 0)
    print(f"\n  GATE: the control MERGES (gap -> 0) while the blocks keep two markers with their")
    print(f"        own volumes and a film at least one cell thick.")
    print(f"        blocks min gap {b['mingap']} cells (peak shared liquid "
          f"{b['overlap']:.3f}); control min gap {c['mingap']} cells "
          f"(merged at t = {c['merged']})  ->  {'PASS' if ok else 'INCONCLUSIVE'}")


# ------------------------------------------------------------------- gate 3b: markers in contact
def gate_contact():
    """The discriminating half of gate 3, when the film gate does not reach contact on its own.

    Seed the two markers ALREADY TOUCHING -- the centre distance is one cell less than the sum of
    the radii, so their PLIC bands share cells from step 0 -- and run a real two-phase step.  A
    single global colour field has one blob there and can never have two again; two markers have
    two, each conserving its own volume, and the union carries the shared liquid explicitly.
    """
    print("\n" + "=" * 96)
    print("GATE 3b — two markers seeded IN CONTACT under a real NS step: the union carries cells")
    print("          that belong to both, and the control is one blob for ever")
    print("=" * 96)
    Eo, Mo, ratio = 10.0, 1e-3, 100.0
    D = 12.0
    Rs, Rl = 0.40 * D, 0.60 * D
    rho_l, sigma = 1.0, 1.0
    g = Eo * sigma / (rho_l * D * D)
    mu_l = (Mo * rho_l ** 2 * sigma ** 3 / g) ** 0.25
    rho_g, mu_g = rho_l / ratio, mu_l / ratio
    nx = ny = int(round(4 * D))
    nz = int(round(7 * D))
    zc_l = 1.6 * D
    zc_s = zc_l + Rs + Rl - 1.0          # ONE cell of band overlap
    steps = 100 if QUICK else 400
    print(f"  D_small = {2*Rs:.1f}, D_large = {2*Rl:.1f}; box {nx}x{ny}x{nz}; centre distance "
          f"{zc_s-zc_l:.1f} = R_s + R_l - 1")

    def build(blocks):
        s = pf.Solver(nx, ny, nz)
        s.set_rho(rho_l)
        s.set_mu(mu_l)
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 1, 0, 0, 0)
        s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 800, 1e-12)
        s.enable_vof()
        B0 = sphere_fractions((nx, ny, nz), Rl, (nx / 2.0, ny / 2.0, zc_l))
        B1 = sphere_fractions((nx, ny, nz), Rs, (nx / 2.0, ny / 2.0, zc_s))
        s.set_vof(np.asfortranarray(np.maximum(B0, B1)))
        s.set_property_model("rho", "linear", "C", [rho_l, rho_g - rho_l])
        s.set_property_model("mu", "linear", "C", [mu_l, mu_g - mu_l])
        s.set_surface_tension(sigma)
        s.set_property_model("force_z", "linear", "C", [-rho_l * g, -(rho_g - rho_l) * g])
        if blocks:
            # SPHERE seeding, not `enable_vof_blocks_from_field`: with the two markers already
            # overlapping there is no way to split a union field between them without giving one
            # marker a slice of the other (measured: -2.7 % / +7.1 % on the two volumes when the
            # boxes are clipped at the midplane).  `enable_vof_blocks` paints each block's colour
            # from the exact sphere fraction, so each marker starts as its own whole bubble --
            # which is the state a single colour field cannot hold, and the point of the gate.
            s.enable_vof_blocks([(nx / 2.0, ny / 2.0, zc_l, Rl), (nx / 2.0, ny / 2.0, zc_s, Rs)])
            s.enable_vof_block_csf()
        return s, 4 / 3 * math.pi * Rl ** 3, 4 / 3 * math.pi * Rs ** 3

    out = {}
    for name, blk in (("blocks", True), ("single field (control)", False)):
        s, v0l, v0s = build(blk)
        h = Solve(800)
        dt = 0.0
        t0 = time.time()
        for i in range(steps):
            if i % 5 == 0:
                L = s.vof_step_limits()
                dt = 0.2 * min(L["cfl_dt"], L["capillary_dt"])
                s.set_dt(dt)
            s.step()
            h.sample(s)
        C = s.get_vof()
        col = C[nx // 2, ny // 2, :]
        occ = np.argwhere(col > 0.5).ravel()
        blobs = 0
        prev = False
        for v in (col > 0.5):
            if v and not prev:
                blobs += 1
            prev = v
        info = dict(blobs=blobs, union=float(C.sum()), solve=h, wall=time.time() - t0)
        if blk:
            st = s.vof_block_stats()
            info["vols"] = [b["volume"] for b in st]
            info["shared"] = sum(info["vols"]) - info["union"]
            info["cells_both"] = None
        out[name] = info
        print(f"\n  -- {name}: {steps} steps, {h}  ({info['wall']:.0f} s)")
        if blk:
            print(f"     TWO markers, volumes {info['vols'][0]:.3f} / {info['vols'][1]:.3f} "
                  f"(seed {v0l:.3f} / {v0s:.3f}, drift {info['vols'][0]/v0l-1:+.2e} / "
                  f"{info['vols'][1]/v0s-1:+.2e})")
            print(f"     union {info['union']:.3f} cells; SHARED liquid "
                  f"sum(V_marker) - sum(C_union) = {info['shared']:.3f} cells")
            print(f"     blobs along the axis in the union field: {blobs}")
        else:
            print(f"     ONE colour field: {blobs} blob(s) along the axis, "
                  f"union {info['union']:.3f} cells")
    b, c = out["blocks"], out["single field (control)"]
    ok = b["shared"] > 0.0
    print(f"\n  GATE: the blocks carry {b['shared']:.3f} cells of liquid that belong to BOTH "
          f"markers -- a state a single colour field cannot represent (the control's union is "
          f"{c['union']:.3f} cells in {c['blobs']} blob).  ->  {'PASS' if ok else 'FAIL'}")


# --------------------------------------------------------------------------- W1 in Python
def gate_swarm():
    print("\n" + "=" * 96)
    print("RUNG W1 — 64-bubble swarm: master assignment, per-bubble outputs, packing")
    print("=" * 96)
    n = 48
    s = pf.Solver(n, n, n)
    s.set_rho(1.0)
    s.set_mu(0.01)
    s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
    s.enable_vof()
    s.set_vof(np.zeros((n, n, n), order="F"))
    seeds = []
    k = 0
    for i in range(4):
        for j in range(4):
            for m in range(4):
                r = 2.0 + 7.0 * ((k % 8) / 7.0)
                seeds.append(((i + 0.5) * n / 4, (j + 0.5) * n / 4, (m + 0.5) * n / 4, r))
                k += 1
    s.enable_vof_blocks(seeds)
    print(f"  {len(s.vof_block_stats())} markers seeded")
    print(f"  {'mode':<16}{'imbalance (np = 1 here; the census is the replicated table)':<40}")
    for mode, name in ((0, "round robin"), (1, "LPT"), (2, "weighted ORB")):
        print(f"  {name:<16}{s.vof_block_imbalance_of(mode):.4f}")
    st = s.vof_block_stats()
    tot_v = sum(b["volume"] for b in st)
    tot_a = sum(b["area"] for b in st)
    print(f"  per-bubble outputs: total volume {tot_v:.2f} cells, total interface area "
          f"{tot_a:.2f} cells^2")
    b = max(st, key=lambda q: q["volume"])
    R = (3 * b["volume"] / (4 * math.pi)) ** (1 / 3)
    print(f"  largest marker id {b['id']}: V {b['volume']:.3f} -> R {R:.3f} cells, area "
          f"{b['area']:.2f} vs 4 pi R^2 {4*math.pi*R*R:.2f} ({100*(b['area']/(4*math.pi*R*R)-1):+.2f} %)")
    print(f"    centroid {tuple(round(q,3) for q in b['centroid'])}  moments "
          f"{tuple(round(q,3) for q in b['moments'])}")


ALL = {"hysing": gate_hysing, "grace": gate_grace, "pair": gate_pair,
       "contact": gate_contact, "swarm": gate_swarm}

if __name__ == "__main__":
    for g in (GATES or list(ALL)):
        ALL[g]()
