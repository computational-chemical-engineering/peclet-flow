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
def grace_terminal(Eo, Mo, d, rho_l, mu_l, sigma, g):
    """Grace (1973) correlation, as tabulated by Clift, Grace & Weber (1978) ch. 7 eq. (7-10):
       U = (mu_l / (rho_l d)) Mo^-0.149 (J - 0.857),  J from the Eotvos-Morton chart."""
    H = (4.0 / 3.0) * Eo * Mo ** -0.149          # (mu_l/mu_w)^-0.14 = 1 for a water-like reference
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
    D = 16.0 if QUICK else 20.0
    R = 0.5 * D
    rho_l, sigma = 1.0, 1.0
    g = Eo * sigma / (rho_l * D * D)                       # Eo = rho g D^2 / sigma (dRho ~ rho_l)
    mu_l = (Mo * rho_l ** 2 * sigma ** 3 / g) ** 0.25      # Mo = g mu^4 / (rho sigma^3)
    rho_g, mu_g = rho_l / ratio, mu_l / ratio
    nx = int(round(6 * D))
    nz = int(round(10 * D))
    ny = nx
    print(f"  D = {D:g} cells, box {nx}x{ny}x{nz}, rho_l {rho_l}, rho_g {rho_g}, mu_l {mu_l:.4e}, "
          f"sigma {sigma}, g {g:.4e}")
    Ut = grace_terminal(Eo, Mo, D, rho_l, mu_l, sigma, g)
    Ub = math.sqrt(g * D)
    print(f"  Grace/Clift correlation: U_T = {Ut:.4f} cells/s  (Re = {rho_l*Ut*D/mu_l:.1f}, "
          f"Fr = U/sqrt(gD) = {Ut/Ub:.3f})")
    s = pf.Solver(nx, ny, nz)
    s.set_rho(rho_l)
    s.set_mu(mu_l)
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)
    s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 800, 1e-12)
    s.enable_vof()
    B = sphere_fractions((nx, ny, nz), R, (nx / 2.0, ny / 2.0, 2.5 * D))
    s.set_vof(np.asfortranarray(B))
    s.set_property_model("rho", "linear", "C", [rho_l, rho_g - rho_l])
    s.set_property_model("mu", "linear", "C", [mu_l, mu_g - mu_l])
    s.set_surface_tension(sigma)
    s.set_property_model("force_z", "linear", "C",
                         [-rho_l * g, -(rho_g - rho_l) * g])
    s.enable_vof_blocks_from_field([marker_box(B)])
    s.enable_vof_block_csf()
    T = (2.0 if QUICK else 6.0) * D / Ut
    h = Solve(800)
    t, i = 0.0, 0
    hist = []
    t0 = time.time()
    while t < T:
        if i % 10 == 0:
            L = s.vof_step_limits()
            s.set_dt(0.4 * min(L["cfl_dt"], L["capillary_dt"]))
        dt = s.dt()
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
    print("GATE 3 — two bubbles in line: the film drains to one cell and the blocks stay TWO")
    print("=" * 96)
    print("  The trailing bubble rises into the leading one's wake and catches it.  In a single")
    print("  global colour field the two merge the moment their bands touch; with one marker each")
    print("  they cannot, and the union carries CELLS THAT BELONG TO BOTH.  The control run is the")
    print("  same case on the structured field.")
    Eo, Mo, ratio = 10.0, 1e-3, 100.0
    D = 12.0 if QUICK else 16.0
    R = 0.5 * D
    rho_l, sigma = 1.0, 1.0
    g = Eo * sigma / (rho_l * D * D)
    mu_l = (Mo * rho_l ** 2 * sigma ** 3 / g) ** 0.25
    rho_g, mu_g = rho_l / ratio, mu_l / ratio
    nx = ny = int(round(4 * D))
    nz = int(round(10 * D))
    sep = 2.2 * D
    zc0, zc1 = 2.0 * D, 2.0 * D + sep

    def build(blocks):
        s = pf.Solver(nx, ny, nz)
        s.set_rho(rho_l)
        s.set_mu(mu_l)
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 1, 0, 0, 0)
        s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 800, 1e-12)
        s.enable_vof()
        B0 = sphere_fractions((nx, ny, nz), R, (nx / 2.0, ny / 2.0, zc0))
        B1 = sphere_fractions((nx, ny, nz), R, (nx / 2.0, ny / 2.0, zc1))
        s.set_vof(np.asfortranarray(np.maximum(B0, B1)))
        s.set_property_model("rho", "linear", "C", [rho_l, rho_g - rho_l])
        s.set_property_model("mu", "linear", "C", [mu_l, mu_g - mu_l])
        s.set_surface_tension(sigma)
        s.set_property_model("force_z", "linear", "C", [-rho_l * g, -(rho_g - rho_l) * g])
        if blocks:
            s.enable_vof_blocks_from_field([marker_box(B0), marker_box(B1)])
            s.enable_vof_block_csf()
        return s

    Ut = grace_terminal(Eo, Mo, D, rho_l, mu_l, sigma, g)
    T = (1.5 if QUICK else 4.0) * sep / Ut
    res = {}
    for name, blk in (("blocks", True), ("single field (control)", False)):
        s = build(blk)
        h = Solve(800)
        t, i = 0.0, 0
        t0 = time.time()
        while t < T:
            if i % 10 == 0:
                L = s.vof_step_limits()
                s.set_dt(0.4 * min(L["cfl_dt"], L["capillary_dt"]))
            t += s.dt()
            i += 1
            s.step()
            h.sample(s)
        C = s.get_vof()
        # the neck: the minimum of C along the axis between the two centroids' final positions
        col = C[nx // 2, ny // 2, :]
        occupied = np.argwhere(col > 0.5).ravel()
        gap = 0
        if len(occupied):
            inside = col[occupied.min():occupied.max() + 1]
            gap = int((inside < 0.5).sum())
        info = dict(steps=i, solve=h, gap=gap, wall=time.time() - t0)
        if blk:
            st = s.vof_block_stats()
            info["vols"] = [b["volume"] for b in st]
            info["zc"] = [b["centroid"][2] for b in st]
            # cells carried by BOTH markers: sum of the marker volumes minus the union
            info["overlap"] = sum(info["vols"]) - float(C.sum())
        else:
            info["union"] = float(C.sum())
        res[name] = info
        print(f"\n  -- {name}: {i} steps, {h}  ({info['wall']:.0f} s)")
        if blk:
            print(f"     marker volumes {info['vols'][0]:.4f} / {info['vols'][1]:.4f}  "
                  f"(seed {4/3*math.pi*R**3:.4f} each)")
            print(f"     centroids z {info['zc'][0]:.2f} / {info['zc'][1]:.2f}   film gap on the "
                  f"axis {gap} cells   shared liquid {info['overlap']:.3f} cells")
        else:
            print(f"     film gap on the axis {gap} cells (0 = the two have MERGED)")
    b, c = res["blocks"], res["single field (control)"]
    print(f"\n  GATE: the blocks keep two markers with their own volumes; the control merges.")
    print(f"        blocks gap {b['gap']} cells, control gap {c['gap']} cells  ->  "
          f"{'PASS' if b['gap'] >= 1 else 'FAIL (the markers themselves lost the film)'}")


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


ALL = {"hysing": gate_hysing, "grace": gate_grace, "pair": gate_pair, "swarm": gate_swarm}

if __name__ == "__main__":
    for g in (GATES or list(ALL)):
        ALL[g]()
