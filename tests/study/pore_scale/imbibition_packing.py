#!/usr/bin/env python3
"""WO-V7 case 2 — IMBIBITION INTO A SPHERE PACKING.

A random loose packing of ~36 SDF spheres sits in a laterally periodic column that is gas-filled at
t = 0.  Liquid is fed through the bottom face at a fixed superficial velocity and leaves through the
top.  Measured: the liquid saturation of the bed against time, the breakthrough time, and the
fraction of the pore space left as gas DISCONNECTED from the outlet once the front has broken
through (the trapped-gas fraction).

The packing is generated in this file (a soft-sphere settling in numpy, deterministic seed), so the
script has no dependency on `dem`; the SDF grain radius is 0.85 of the contact radius, the same
device the `bubble-through-packing` gallery page uses, because grains that touch leave throats the
grid cannot resolve.

CONVENTIONS AND CAVEATS (WO-V7 preamble; stated on every run):
  * STATIC contact angle (V5b).  The capillary number is APPARENT: mu_l U_superficial / sigma.
  * density ratio 100 (V2b's float-momentum rating; WO-R's G3 caveat at 1000).
  * inflow/outflow, never a periodic net body force (VOF_PLAN 13.2 item 6).
  * the packing is kept clear of the inflow and outflow planes (flow's open-BC x cut-cell defect).
  * a capped pressure solve makes the run INVALID (rule 3b).

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/pore_scale/imbibition_packing.py [--quick]
        [--theta 30,60] [--ca 1e-3] [--probe]        # --probe = ms/step only, 60 steps
"""
import argparse
import heapq
import json
import math
import sys
import time

import numpy as np

import peclet.flow as pf

# ------------------------------------------------------------------------------------- the scene
NX = NY = 48
NZ = 96
Z_BED = 16.0                     # bottom of the bed (inlet plenum below it)
N_GRAIN = 36
R_CONTACT = 8.0                  # radius the settling uses
F_SDF = 0.85                     # SDF radius / contact radius -> resolvable throats
R_SDF = F_SDF * R_CONTACT

SIGMA = 100.0
RHO_L, RHO_G = 100.0, 1.0
MU_L, MU_G = 4.0, 0.04
CAP_CFL = 0.5
PRESS_CAP = 400


# ------------------------------------------------------------------------------- packing (numpy)
def settle(n=N_GRAIN, L=float(NX), R=R_CONTACT, seed=7, iters=4000, dz=0.02):
    """A soft-sphere deposit: gravity + pairwise overlap resolution + a floor, x/y periodic."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n, 3))
    P[:, :2] = rng.uniform(0.0, L, (n, 2))
    P[:, 2] = R + 2.1 * R * (np.arange(n) // 6) + rng.uniform(-0.4, 0.4, n)
    for _ in range(iters):
        P[:, 2] -= dz
        for _ in range(4):
            d = P[None, :, :] - P[:, None, :]
            d[:, :, 0] -= L * np.round(d[:, :, 0] / L)
            d[:, :, 1] -= L * np.round(d[:, :, 1] / L)
            r = np.sqrt((d ** 2).sum(-1))
            np.fill_diagonal(r, 1e9)
            ov = np.maximum(2.0 * R - r, 0.0)
            u = d / np.maximum(r, 1e-9)[:, :, None]
            P -= 0.5 * (ov[:, :, None] * u).sum(axis=1)
            P[:, 2] = np.maximum(P[:, 2], R)
        P[:, :2] %= L
    d = P[None, :, :] - P[:, None, :]
    d[:, :, 0] -= L * np.round(d[:, :, 0] / L)
    d[:, :, 1] -= L * np.round(d[:, :, 1] / L)
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, 1e9)
    return P, float(np.maximum(2.0 * R - r, 0.0).max())


def bed_sdf(P, nx=NX, ny=NY, nz=NZ, Rg=R_SDF, z0=Z_BED):
    L = float(nx)
    Q = P.copy()
    Q[:, 2] += z0 - R_CONTACT      # the lowest grain CENTRE sits at z0 + (R_contact - R_contact)
    g = np.arange(nx) + 0.5
    gz = np.arange(nz) + 0.5
    X, Y, Z = np.meshgrid(g, np.arange(ny) + 0.5, gz, indexing="ij")
    phi = np.full((nx, ny, nz), 1e30)
    for k in range(len(Q)):
        dx = X - Q[k, 0]; dx -= L * np.round(dx / L)
        dy = Y - Q[k, 1]; dy -= L * np.round(dy / L)
        dz = Z - Q[k, 2]
        phi = np.minimum(phi, np.sqrt(dx * dx + dy * dy + dz * dz) - Rg)
    return np.asfortranarray(phi)


def percolation_throat(phi, klo, khi):
    """Max over bottom->top paths of the minimum SDF along the path (26-connected, x/y periodic):
    the radius of the largest sphere that can be pushed through the bed."""
    sub = phi[:, :, klo:khi]
    nx, ny, nk = sub.shape
    val = np.full(sub.shape, -np.inf)
    h = []
    for i in range(nx):
        for j in range(ny):
            if sub[i, j, 0] > 0:
                val[i, j, 0] = sub[i, j, 0]
                h.append((-sub[i, j, 0], i, j, 0))
    heapq.heapify(h)
    best = -np.inf
    while h:
        nv, i, j, k = heapq.heappop(h)
        v = -nv
        if v < val[i, j, k] - 1e-12:
            continue
        if k == nk - 1:
            best = max(best, v)
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == dj == dk == 0:
                        continue
                    a, b, c = (i + di) % nx, (j + dj) % ny, k + dk
                    if not (0 <= c < nk):
                        continue
                    w = min(v, sub[a, b, c])
                    if w > val[a, b, c] + 1e-12:
                        val[a, b, c] = w
                        heapq.heappush(h, (-w, a, b, c))
    return best


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


def trapped_gas(C, eps, klo, khi):
    """Fraction of the bed pore volume held as gas NOT connected to the outlet plenum.
    6-connected labelling with x/y periodic wrap; the outlet plenum is the seed."""
    from scipy import ndimage
    gas = (C < 0.5) & (eps > 0.0)
    lab, n = ndimage.label(gas)
    # stitch the x and y periodic seams
    eq = set()
    for a, b in ((lab[0], lab[-1]), (lab[:, 0], lab[:, -1])):
        m = (a > 0) & (b > 0)
        eq.update(zip(a[m].ravel().tolist(), b[m].ravel().tolist()))
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in eq:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    root = np.array([find(i) if i else 0 for i in range(n + 1)])
    lab = root[lab]
    open_roots = set(np.unique(lab[:, :, khi:]).tolist()) - {0}
    bed = slice(klo, khi)
    pore = float(eps[:, :, bed].sum())
    gsel = gas[:, :, bed] & ~np.isin(lab[:, :, bed], list(open_roots))
    trapped = float((eps[:, :, bed] * (1.0 - C[:, :, bed]) * gsel).sum())
    return trapped / max(pore, 1e-30), len(open_roots)


def run(theta_deg, ca, sdf, klo, khi, steps, probe=False, after=0.25, sample_every=None,
        budget=None, npy_dir=""):
    U = ca * SIGMA / MU_L
    Re = RHO_L * U * 2 * R_SDF / MU_L
    print("\n" + "=" * 100)
    print(f"IMBIBITION INTO THE PACKING   theta = {theta_deg:g} deg   Ca(apparent) = {ca:g}")
    print("=" * 100)
    print(f"  grid {NX}x{NY}x{NZ}; bed core z = {klo} .. {khi}; sigma {SIGMA:g}, "
          f"rho {RHO_L:g}/{RHO_G:g}, mu {MU_L:g}/{MU_G:g}")
    print(f"  inlet superficial U = Ca*sigma/mu_l = {U:.6g} cells/s -> grain Reynolds "
          f"rho_l U D / mu_l = {Re:.4g}")

    s = pf.Solver(NX, NY, NZ)
    s.set_rho(RHO_L)
    s.set_mu(MU_L)
    s.set_dt(0.05)
    s.set_domain_bc(4, 2, 0.0, 0.0, U)     # -z inflow (liquid)
    s.set_domain_bc(5, 3, 0.0, 0.0, 0.0)   # +z outflow
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=8)
    s.set_pressure_solver_params(80)
    s.set_solid(sdf, cutcell_pressure=True)
    s.enable_vof()
    # t = 0 is the front at the BED FACE: the inlet plenum below the bed is already liquid.  It is
    # not a shortcut for its own sake - filling the plenum from the inlet costs more steps than the
    # bed itself and carries no physics (the plenum is an open channel).
    c0 = np.zeros((NX, NY, NZ), order="F")
    c0[:, :, : klo - 2] = 1.0
    s.set_vof(c0)
    s.set_property_model("rho", "linear", "C", [RHO_G, RHO_L - RHO_G])
    s.set_property_model("mu", "linear", "C", [MU_G, MU_L - MU_G])
    s.enable_vof_momentum(RHO_G, RHO_L)
    s.set_surface_tension(SIGMA)
    s.set_capillary_cfl(CAP_CFL)
    s.set_contact_angle(theta_deg)
    s.set_pressure_fcg(True, PRESS_CAP, 1e-11)     # driver LAST (set_density_mode reselects)
    s.set_vof_inflow(4, 1.0)
    s.set_vof_backflow(5, 0.0)

    eps = np.asarray(s.vof_geometry(0))
    pore = float(eps[:, :, klo:khi].sum())
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
    dt = CAP_CFL * L0["capillary_dt"]
    sample_every = sample_every or max(steps // 150, 5)
    t0 = time.time()
    n = 0
    stop_at = None
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
        if (i + 1) % sample_every == 0 or i == steps - 1:
            C = np.asarray(s.get_vof())
            S = float((C[:, :, klo:khi] * eps[:, :, klo:khi]).sum() / pore)
            top = float((C[:, :, khi - 1] * eps[:, :, khi - 1]).max())
            hist.append((t, S, top))
            if len(hist) % 10 == 1:
                um = max(float(np.abs(np.asarray(f())).max())
                         for f in (s.get_u, s.get_v, s.get_w))
                print(f"    step {i+1:7d}  t {t:10.4g}  dt {dt:9.3g}  S {S:.4f}  top {top:.3f}"
                      f"  max|u| {um:9.2e}  ({time.time()-t0:.0f} s, "
                      f"{1000*(time.time()-t0)/(i+1):.0f} ms/step)", flush=True)
            if bt is None and top > 0.5:
                bt = (t, i + 1, S)
                stop_at = i + 1 + int(after * (i + 1))    # run `after` x t_bt further
                print(f"    breakthrough at t = {t:.5g} (step {i+1}), bed saturation "
                      f"{S:.4f}; continuing to step {stop_at}")
            if stop_at is not None and i + 1 >= stop_at:
                break
            if budget and time.time() - t0 > budget:
                print(f"    *** wall-clock budget {budget:.0f} s reached at step {i+1} "
                      f"(t = {t:.5g}, S = {S:.4f}) - PARTIAL ***")
                break
    wall = time.time() - t0
    C = np.asarray(s.get_vof())
    S = float((C[:, :, klo:khi] * eps[:, :, klo:khi]).sum() / pore)
    trap, nclust = trapped_gas(C, eps, klo, khi)
    if npy_dir:
        import os
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, f"packing_C_theta{int(theta_deg)}.npy"), C)
        np.save(os.path.join(npy_dir, "packing_eps.npy"), eps)
    d = s.vof_diagnostics()
    print(f"\n  ran {n} steps to t = {t:.5g} s in {wall:.0f} s "
          f"({1000*wall/max(n,1):.2f} ms/step, SHARED GPU)")
    print(f"  dt final {dt:.5g}; the WY CFL limit displaced the capillary one on {ncfl} of "
          f"{n} steps ({100.0*ncfl/max(n,1):.1f} %)")
    print(f"  breakthrough: " + (f"t = {bt[0]:.5g} s at step {bt[1]}, saturation at "
                                 f"breakthrough S_bt = {bt[2]:.4f}"
                                 if bt else "NOT reached"))
    print(f"  final bed saturation S = {S:.4f}; trapped (outlet-disconnected) gas "
          f"{trap:.4f} of the pore volume, in {nclust} outlet-connected cluster(s)")
    print(f"  colour: solid_sum {d['solid_sum']:.3e}, min/max over uncut fluid "
          f"{d['min_fluid']:.3e}/{d['max_fluid']:.6f}, clipped {d['clipped_volume']:.3e}")
    print(f"  {h}")
    return dict(theta=theta_deg, ca=ca, U=U, Re=Re, steps=n, t=t, wall=wall,
                ms_per_step=1000 * wall / max(n, 1),
                t_bt=bt[0] if bt else None, step_bt=bt[1] if bt else None,
                S_bt=bt[2] if bt else None, S_final=S, trapped=trap,
                iters=h.iters, capped=h.capped, div=h.div, valid=h.valid,
                cfl_fraction=ncfl / max(n, 1), hist=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--probe", action="store_true", help="60 steps, ms/step only")
    ap.add_argument("--theta", default="30,60")
    ap.add_argument("--ca", default="1e-3")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--npy", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    print("WO-V7 case 2 — imbibition into an SDF sphere packing.  Static contact angle (V5b);")
    print("the capillary numbers are APPARENT (mu_l U_superficial / sigma).")
    t0 = time.time()
    P, ov = settle()
    sdf = bed_sdf(P)
    top = P[:, 2].max() - R_CONTACT + Z_BED + R_SDF
    bot = Z_BED + (R_CONTACT - R_SDF) - R_CONTACT + P[:, 2].min()
    klo, khi = int(math.ceil(bot + R_SDF)), int(math.floor(top - R_SDF))
    eps_bed = float((sdf[:, :, klo:khi] > 0).mean())
    thr = percolation_throat(sdf, klo, khi)
    print(f"\n  {N_GRAIN} grains settled in {time.time()-t0:.0f} s, max contact overlap "
          f"{ov:.4f} cells (contact radius {R_CONTACT}, SDF radius {R_SDF})")
    print(f"  bed surfaces span z = {bot:.1f} .. {top:.1f} ({(top-bot)/(2*R_SDF):.1f} SDF grain "
          f"diameters); bed CORE z = {klo} .. {khi}")
    print(f"  porosity over the bed core: eps = {eps_bed:.4f}")
    print(f"  percolation throat radius (max-min path, bottom -> top): {thr:.3f} cells "
          f"-> the entry pressure scale 2 sigma cos(theta)/r_throat is "
          f"{2*SIGMA/max(thr,1e-9):.3g} x cos(theta)")
    print(f"  inlet plenum z = 0 .. {int(bot)} and outlet plenum z = {int(top)} .. {NZ} keep the "
          f"solid clear of both open faces")

    if a.probe:
        r = run(30.0, 1e-3, sdf, klo, khi, 60, probe=True, sample_every=30)
        print(f"\n  PROBE: {r['ms_per_step']:.2f} ms/step on {NX}x{NY}x{NZ}")
        return 0

    thetas = [float(x) for x in a.theta.split(",")]
    cas = [float(x) for x in a.ca.split(",")]
    rows = []
    for th in thetas:
        for ca in cas:
            U = ca * SIGMA / MU_L
            ttrav = (khi - klo + 6) / (U / max(eps_bed, 0.2))
            dt0 = CAP_CFL * math.sqrt((RHO_L + RHO_G) / (4 * math.pi * SIGMA))
            steps = int(2.5 * ttrav / dt0)
            if a.quick:
                steps = min(steps, 600)
            rows.append(run(th, ca, sdf, klo, khi, steps, budget=a.budget or None,
                            npy_dir=a.npy))
    print("\n" + "=" * 100)
    print("SUMMARY — imbibition into the packing")
    print("=" * 100)
    print(f"{'theta':>6} {'Ca':>8} {'Re':>7} {'t_bt':>10} {'S_bt':>7} {'S_final':>8} "
          f"{'trapped':>8} {'iters':>6} {'ms/step':>8} {'valid':>6}")
    for r in rows:
        tb = f"{r['t_bt']:.4g}" if r["t_bt"] else "-"
        sb = f"{r['S_bt']:.4f}" if r["S_bt"] else "-"
        print(f"{r['theta']:6g} {r['ca']:8g} {r['Re']:7.3g} {tb:>10} {sb:>7} {r['S_final']:8.4f} "
              f"{r['trapped']:8.4f} {r['iters']:6d} {r['ms_per_step']:8.2f} "
              f"{'yes' if r['valid'] else 'NO':>6}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\n  wrote {a.out}")
    return 0 if all(r["valid"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
