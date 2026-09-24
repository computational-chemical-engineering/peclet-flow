#!/usr/bin/env python3
"""Overlapping markers in the block VoF container: what actually goes wrong when two bubbles'
markers interpenetrate (the channel_18 blow-up, WO-W3 section 7).

    static    two spheres at rest, zero gravity, centre distance d swept from separated to
              overlapping by several cells.  If the summed per-marker CSF were NOT balanced
              against the union-colour projection (the W3 hypothesis), the parasitic current
              would jump at d < 2R.  If it stays at the single-bubble level, the static balance
              holds and the blow-up has to be dynamic.
    squeeze   the same pair driven into each other by opposite body forces on the two markers'
              regions -- the controlled version of a turbulent collision.

The case numbers mirror channel_18 in cell units: D = 10 cells, rho_g/rho_l = 0.1,
mu_g = mu_l, sigma = 320 (S^3 * 5e-3 at S = 40 cells per unit length).

Usage:  PYTHONPATH=<build> python tests/study/vof_blocks_overlap.py [static|squeeze ...]
"""
import math
import sys
import time

import numpy as np

import os

import peclet.flow as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GATES = [a for i, a in enumerate(sys.argv[1:], 1)
         if not a.startswith("--") and not sys.argv[i - 1].startswith("--")] or ["static"]


def arg(name, default, cast=float):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


R = arg("--R", 5.0)
RHO_L, RHO_G = 1.0, arg("--rho-g", 0.1)
MU = arg("--mu", 0.5333)
MU_G = arg("--mu-g", MU)          # gas viscosity (default: the liquid's, as channel_18)
TENT = "--tent" in sys.argv       # vof_overlap_design 5.6: the opt-in overlap density
REPICK = "--repick" in sys.argv   # re-pick dt every step from vof_step_limits() (static gate)
SIGMA = arg("--sigma", 320.0)
STEPS = arg("--steps", 400, int)


def solver(n, seeds, blocks=True, csf=True):
    nx, ny, nz = n
    s = pf.Solver(nx, ny, nz)
    s.set_rho(RHO_L)
    s.set_mu(MU)
    s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 800, 1e-12)
    s.enable_vof()
    s.set_vof(np.zeros((nx, ny, nz), order="F"))
    s.set_property_model("rho", "linear", "C", [RHO_L, RHO_G - RHO_L])
    s.set_property_model("mu", "linear", "C", [MU, MU_G - MU])
    s.set_surface_tension(SIGMA)
    s.enable_vof_blocks(seeds)
    if csf:
        s.enable_vof_block_csf()
    if TENT:
        s.diagnostics.set_vof_block_overlap_density(True)
    return s


def counters(s):
    """The overlap-design counters: clip (block cascade), debris census, overlap census."""
    st = s.diagnostics.vof_block_stats()
    cs = s.diagnostics.vof_block_curvature_stats()
    return {"clipped": cs.get("clipped", 0),
            "debris_cells": sum(b.get("debris_cells", 0) for b in st),
            "debris_volume": sum(b.get("debris_volume", 0.0) for b in st),
            "returned": [b.get("debris_returned", 0.0) for b in st],
            "lost": sum(b.get("debris_lost", 0.0) for b in st),
            "unresolved": sum(b.get("debris_unresolved", 0) for b in st),
            "overlap_cells": st[0].get("overlap_cells", 0) if st else 0,
            "overlap_excess": st[0].get("overlap_excess", 0.0) if st else 0.0,
            "bound": bool(st[0].get("overlap_bound_active", False)) if st else False}


def umax(s):
    return max(np.abs(s.get_u()).max(), np.abs(s.get_v()).max(), np.abs(s.get_w()).max())


RUNSTAT = {}


def run(s, steps, dt=None, every=50, tag="", safety=None):
    L = s.vof_step_limits()
    if dt is None:
        dt = 0.25 * min(L["cfl_dt"], L["capillary_dt"])
    s.set_dt(dt)
    hist = []
    RUNSTAT.update(bound_steps=0, dt_min=dt, clipped=0, debris=0, ov_cells=0)
    for i in range(steps):
        if REPICK and safety is not None:
            L = s.vof_step_limits()
            dt = safety * L["capillary_dt"]
            s.set_dt(dt)
            RUNSTAT["dt_min"] = min(RUNSTAT["dt_min"], dt)
        try:
            s.step()
            if hasattr(s.diagnostics, "vof_block_stats"):
                k = counters(s)
                RUNSTAT["bound_steps"] += int(k["bound"])
                RUNSTAT["clipped"] += k["clipped"]
                RUNSTAT["debris"] = max(RUNSTAT["debris"], k["debris_cells"])
                RUNSTAT["ov_cells"] = max(RUNSTAT["ov_cells"], k["overlap_cells"])
        except Exception as exc:
            print(f"    {tag} step {i}: THREW {exc}")
            hist.append(float("nan"))
            break
        u = umax(s)
        hist.append(u)
        if not math.isfinite(u):
            break
    return dt, np.array(hist)


def gate_static():
    print("=" * 96)
    print("STATIC — two spheres at rest, no gravity: parasitic current vs centre distance d")
    print("=" * 96)
    n = (int(8 * R), int(5 * R), int(5 * R))
    cy, cz = n[1] / 2, n[2] / 2
    cx = n[0] / 2
    cap = math.sqrt((RHO_L + RHO_G) / (4 * math.pi * SIGMA))
    print(f"  R = {R} cells, box {n}, rho_g/rho_l = {RHO_G}, mu = {MU}, sigma = {SIGMA}; "
          f"Brackbill dt = {cap:.4e}; Laplace dp = {2*SIGMA/R:.2f}")
    rows = []
    ds = [float(x) for x in arg("--d", "14,10.5,10,9,8,6,4", str).split(",")]
    cases = ([("single", None)] if "--no-single" not in sys.argv else []) + \
        [(f"d={d:g}", d) for d in ds]
    for name, d in cases:
        if d is None:
            seeds = [(cx, cy, cz, R)]
        else:
            seeds = [(cx - d / 2, cy, cz, R), (cx + d / 2, cy, cz, R)]
        s = solver(n, seeds)
        v0 = [b["volume"] for b in s.diagnostics.vof_block_stats()]
        t0 = time.time()
        dt, h = run(s, STEPS, dt=arg("--cap-safety", 0.25) * cap, tag=name,
                    safety=arg("--cap-safety", 0.25))
        v1 = [b["volume"] for b in s.diagnostics.vof_block_stats()]
        C = s.get_field("C")
        dv = max(abs(a / b - 1) for a, b in zip(v1, v0))
        rows.append((name, h))
        print(f"  {name:8s} max|u| @50 {h[min(49,len(h)-1)]:.3e} @end {h[-1]:.3e} peak {np.nanmax(h):.3e}  "
              f"maxC {float(C.max()):.6f}  dV {dv:.1e}  ({time.time()-t0:.0f} s)  "
              f"bound-steps {RUNSTAT['bound_steps']} dt_min/Brackbill {RUNSTAT['dt_min']/cap:.3f} "
              f"clipped {RUNSTAT['clipped']} debris<= {RUNSTAT['debris']} "
              f"ov_cells<= {RUNSTAT['ov_cells']}", flush=True)
        if "--dump" in sys.argv:
            np.savez(f"{arg('--dump', 'static', str)}_{name}.npz", h=h, u=s.get_u(), v=s.get_v(),
                     w=s.get_w(), C=C)
    return rows


def gate_shear():
    """Two bubbles in a Couette flow, offset in y by b < D so they collide glancingly -- the
    deterministic stand-in for the channel's turbulent collisions.  Per step: max|u|, union max C,
    marker overlap (sum V_k - sum C_union), and the step at which anything goes non-finite."""
    print("=" * 96)
    print("SHEAR — two bubbles colliding in Couette flow (walls at +-y moving at +-U)")
    print("=" * 96)
    U = arg("--U", 16.0)
    b = arg("--b", 6.0)
    gap0 = arg("--gap", 20.0)
    blocks = "--single-field" not in sys.argv
    n = (int(round(13 * R)), int(round(8 * R)), int(round(6 * R)))
    nx, ny, nz = n
    s = pf.Solver(nx, ny, nz)
    s.set_rho(RHO_L)
    s.set_mu(MU)
    s.set_domain_bc("-y", "wall", (-U, 0, 0))
    s.set_domain_bc("+y", "wall", (U, 0, 0))
    s.set_pressure_geometry(np.full(n, 10.0, order="F"))
    s.set_pressure_chebyshev(True, 800, 1e-12)
    y = (np.arange(ny) + 0.5) / ny
    u0 = np.broadcast_to((-U + 2 * U * y)[None, :, None], n)
    s.set_velocity(0, np.asfortranarray(u0))
    s.enable_vof()
    s.set_vof(np.zeros(n, order="F"))
    s.set_property_model("rho", "linear", "C", [RHO_L, RHO_G - RHO_L])
    s.set_property_model("mu", "linear", "C", [MU, MU_G - MU])
    s.set_surface_tension(SIGMA)
    seeds = [(nx / 2 - gap0 / 2, ny / 2 + b / 2, nz / 2, R), (nx / 2 + gap0 / 2, ny / 2 - b / 2, nz / 2, R)]
    if blocks:
        s.enable_vof_blocks(seeds)
        s.enable_vof_block_csf()
        if TENT:
            s.diagnostics.set_vof_block_overlap_density(True)
    else:
        from vof_surface_tension import sphere_fractions
        C = np.zeros(n)
        for q in seeds:
            C = np.maximum(C, sphere_fractions(n, q[3], q[:3]))
        s.set_vof(np.asfortranarray(C))
    rel = 2 * U / ny * b
    print(f"  box {n}, R = {R}, b = {b}, U = {U} -> approach speed {rel:.2f} cells/time; "
          f"We = {RHO_L*rel**2*2*R/SIGMA:.2f}; {'BLOCKS' if blocks else 'single field'}")
    T = arg("--T", 3.0 * gap0 / max(rel, 1e-9))
    t, i = 0.0, 0
    t0 = time.time()
    V0 = [q["volume"] for q in s.diagnostics.vof_block_stats()] if blocks else None
    agg = dict(umax=0.0, bound=0, contact=0, ovmax=0.0, clipped=0, debris=0, dvol=0.0,
               first_contact=None, last_contact=None, debris_steps=0, debris_vol_max=0.0)
    while t < T:
        L = s.vof_step_limits()
        dt = 0.25 * min(L["cfl_dt"], L["capillary_dt"])
        if not (dt > 0 and math.isfinite(dt)):
            print(f"  step {i}: limits non-finite {L}")
            break
        s.set_dt(dt)
        try:
            s.step()
        except Exception as exc:
            print(f"  step {i} t={t:.3f}: THREW {exc}")
            break
        t += dt
        i += 1
        if blocks and hasattr(s.diagnostics, "vof_block_curvature_stats"):
            k = counters(s)
            agg["umax"] = max(agg["umax"], umax(s))
            agg["bound"] += int(k["bound"])
            if k["overlap_cells"] > 0:
                agg["contact"] += 1
                agg["first_contact"] = agg["first_contact"] or i
                agg["last_contact"] = i
            agg["ovmax"] = max(agg["ovmax"], k["overlap_excess"])
            agg["clipped"] += k["clipped"]
            agg["debris"] = max(agg["debris"], k["debris_cells"])
            agg["debris_steps"] += int(k["debris_cells"] > 0)
            agg["debris_vol_max"] = max(agg["debris_vol_max"], k["debris_volume"])
        if i % 20 == 0:
            C = s.get_field("C")
            um = umax(s)
            if blocks:
                st = s.diagnostics.vof_block_stats()
                vols = [q["volume"] for q in st]
                ov = sum(vols) - float(C.sum())
                cen = [q["centroid"] for q in st]
                dx = cen[1][0] - cen[0][0]
                extra = f"V {vols[0]:.4f}/{vols[1]:.4f} overlap {ov:8.4f}  dx {dx:6.2f}"
            else:
                extra = f"sumC {float(C.sum()):.4f}"
            print(f"  step {i:5d} t={t:7.3f} dt={dt:.2e} max|u| {um:8.3f} maxC {float(C.max()):.6f} "
                  f"{extra}  ({time.time()-t0:.0f} s)", flush=True)
            if not math.isfinite(um) or um > 20 * U:
                print("  BLOW-UP")
                break
    if blocks:
        V1 = [q["volume"] for q in s.diagnostics.vof_block_stats()]
        agg["dvol"] = max(abs(a / b - 1) for a, b in zip(V1, V0))
        print(f"  SUMMARY steps {i} t {t:.3f}/{T:.3f}  max|u| {agg['umax']:.3f}  "
              f"max rel dV {agg['dvol']:.2e}  contact steps {agg['contact']} "
              f"({agg['first_contact']}..{agg['last_contact']})  max overlap excess "
              f"{agg['ovmax']:.4f}  phantom-bound steps {agg['bound']}  clipped cells (sum) "
              f"{agg['clipped']}  debris: max cells {agg['debris']}, steps with debris "
              f"{agg['debris_steps']}, max volume {agg['debris_vol_max']:.3e}; ledger returned per "
              f"marker {[f'{r:.3e}' for r in k['returned']]}, lost {k['lost']:.3e}, unresolved "
              f"{k['unresolved']}", flush=True)


ALL = {"static": gate_static, "shear": gate_shear}

if __name__ == "__main__":
    for g in GATES:
        ALL[g]()
