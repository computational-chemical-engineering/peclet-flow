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
    s.set_property_model("mu", "linear", "C", [MU, 0.0])
    s.set_surface_tension(SIGMA)
    s.enable_vof_blocks(seeds)
    if csf:
        s.enable_vof_block_csf()
    return s


def umax(s):
    return max(np.abs(s.get_u()).max(), np.abs(s.get_v()).max(), np.abs(s.get_w()).max())


def run(s, steps, dt=None, every=50, tag=""):
    L = s.vof_step_limits()
    if dt is None:
        dt = 0.25 * min(L["cfl_dt"], L["capillary_dt"])
    s.set_dt(dt)
    hist = []
    for i in range(steps):
        try:
            s.step()
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
    cases = [("single", None)] + [(f"d={d:g}", d) for d in (14.0, 10.5, 10.0, 9.0, 8.0, 6.0, 4.0)]
    for name, d in cases:
        if d is None:
            seeds = [(cx, cy, cz, R)]
        else:
            seeds = [(cx - d / 2, cy, cz, R), (cx + d / 2, cy, cz, R)]
        s = solver(n, seeds)
        v0 = [b["volume"] for b in s.diagnostics.vof_block_stats()]
        t0 = time.time()
        dt, h = run(s, STEPS, dt=0.25 * cap, tag=name)
        v1 = [b["volume"] for b in s.diagnostics.vof_block_stats()]
        C = s.get_field("C")
        dv = max(abs(a / b - 1) for a, b in zip(v1, v0))
        rows.append((name, h))
        print(f"  {name:8s} max|u| @50 {h[min(49,len(h)-1)]:.3e} @end {h[-1]:.3e} peak {np.nanmax(h):.3e}  "
              f"maxC {float(C.max()):.6f}  dV {dv:.1e}  ({time.time()-t0:.0f} s)", flush=True)
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
    s.set_property_model("mu", "linear", "C", [MU, 0.0])
    s.set_surface_tension(SIGMA)
    seeds = [(nx / 2 - gap0 / 2, ny / 2 + b / 2, nz / 2, R), (nx / 2 + gap0 / 2, ny / 2 - b / 2, nz / 2, R)]
    if blocks:
        s.enable_vof_blocks(seeds)
        s.enable_vof_block_csf()
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


ALL = {"static": gate_static, "shear": gate_shear}

if __name__ == "__main__":
    for g in GATES:
        ALL[g]()
