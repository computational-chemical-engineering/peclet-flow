#!/usr/bin/env python3
"""WO-Q (rung V5a) — VoF transport in cut cells: the gate battery, in Python.

Every gate prints its numbers; nothing is summarised into an adjective. The three that matter:

  G2  CONSERVATION THROUGH A PACKING (kinematic). The single-phase solver is run to a Stokes steady
      state on a periodic sphere array, that PROJECTED velocity is frozen, and a liquid slab is
      advected by `advect_vof` alone. The conserved functional of the openness-weighted scheme is
      `sum eps_eff * C` (`src/vof/cutcell.hpp` rule 1) and its floor is the projection's own
      `max_open_divergence()` — both are reported, together with the raw `sum eps * C`, the colour
      in solid cells (0 by construction), the boundedness in UNCUT fluid cells and the clipped
      volume.

  G4  COUPLED DRAINING. The same packing at density ratio 10 with a ZERO-MEAN buoyancy force,
      `enable_vof` + the LinearMix closures, optionally `enable_vof_momentum`. Gates: no NaN, the
      colour drift against the projection floor, zero colour in solid cells, `max|u|` bounded, and
      the pressure iteration count against its cap (a capped solve makes a run INVALID).

  G5  THE 90-DEGREE NEUTRAL FILL. A liquid cap on a flat SDF wall at a HALF-INTEGER z, so the wall
      cells are genuinely cut, with surface tension. The apparent contact angle is measured from the
      cap height on the axis and the contact radius on the first fluid plane
      (`theta = 2 atan(h/a)`, the spherical-cap relation), and the Young-Laplace jump against
      `2 sigma / R`.

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_cutcell.py [g2 g4 g5 ...]
"""
import math
import sys

import numpy as np

import peclet.flow as pf

PRESS_MAXIT = 300


def sphere_array_sdf(n, centres, radius):
    """Periodic array of spheres; > 0 in fluid (flow's SDF convention), cell centres at i + 1/2."""
    ax = (np.arange(n) + 0.5)[:, None, None]
    ay = (np.arange(n) + 0.5)[None, :, None]
    az = (np.arange(n) + 0.5)[None, None, :]
    best = np.full((n, n, n), 1e30)
    for cx, cy, cz in centres:
        for px in (-n, 0, n):
            for py in (-n, 0, n):
                for pz in (-n, 0, n):
                    d = np.sqrt((ax - (cx + px)) ** 2 + (ay - (cy + py)) ** 2
                                + (az - (cz + pz)) ** 2) - radius
                    best = np.minimum(best, d)
    return np.asfortranarray(best)


DEFAULT_CENTRES = [(6.0, 7.0, 8.0), (20.0, 9.0, 23.0), (11.0, 24.0, 19.0), (26.0, 22.0, 6.0)]


def packing(n):
    f = n / 32.0
    return sphere_array_sdf(n, [(a * f, b * f, c * f) for a, b, c in DEFAULT_CENTRES], 6.0 * f)


def stokes_state(n, steps=60, mu=0.2, force=(2e-3, 1e-3, 5e-4)):
    s = pf.Solver(n, n, n)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_dt(1.0)
    s.set_body_force(*force)
    s.set_solid(packing(n), cutcell_pressure=True)
    it = 0
    for _ in range(steps):
        s.step()
        it = max(it, s.last_pressure_iterations())
    return s, it


# --------------------------------------------------------------------------------------- G2
def g2(n=48, nadv=500, cfl=0.2):
    s, it = stokes_state(n)
    div = s.max_open_divergence()
    s.enable_vof()
    c0 = np.zeros((n, n, n), order="F")
    c0[:, :, : n // 2] = 1.0
    s.set_vof(c0)
    d0 = s.vof_diagnostics()
    dt = cfl / s.vof_max_courant()
    clip = 0.0
    mn, mx = 1e30, -1e30
    solid = 0.0
    for _ in range(nadv):
        s.advect_vof(dt)
        d = s.vof_diagnostics()
        clip += d["clipped_volume"]
        mn = min(mn, d["min_fluid"])
        mx = max(mx, d["max_fluid"])
        solid = max(solid, abs(d["solid_sum"]))
    d1 = s.vof_diagnostics()
    drift = (d1["volume"] - d0["volume"]) / d0["volume"]
    raw = (d1["raw_volume"] - d0["raw_volume"]) / d0["raw_volume"]
    print(f"G2 packing {n}^3, {nadv} kinematic steps at interface CFL {cfl} (dt = {dt:.5g})")
    print(f"   max|div(open u)| {div:.3e}   pressure iters {it}")
    print(f"   sum eps_eff C {d0['volume']:.15e} -> {d1['volume']:.15e}   drift {drift:.3e}")
    print(f"   sum eps     C {d0['raw_volume']:.15e} -> {d1['raw_volume']:.15e}   drift {raw:.3e}")
    print(f"   solid cells {d1['solid_cells']}, cut cells {d1['cut_cells']}, "
          f"clamped faces last step {d1['clamped_faces']}")
    print(f"   max|sum C over solid cells| {solid:.3e}   (must be exactly 0)")
    print(f"   min/max C over UNCUT fluid cells {mn:.3e} / {mx:.17g}")
    print(f"   clipped liquid volume over the run {clip:.6e} "
          f"({clip / d0['volume']:.3e} of the liquid volume)")
    ok = abs(drift) <= 1e-11 and solid == 0.0 and mn >= -1e-12 and mx <= 1 + 1e-12
    print(f"   G2 {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------------------- G4
def g4(n=48, steps=400, ratio=10.0, grav=2e-3, momentum=True):
    s = pf.Solver(n, n, n)
    s.set_rho(1.0)
    s.set_mu(0.05)
    s.set_dt(0.5)
    s.set_solid(packing(n), cutcell_pressure=True)
    s.enable_vof()
    c0 = np.zeros((n, n, n), order="F")
    c0[:, :, 3 * n // 4:] = 1.0          # a liquid layer resting on top of the packing
    s.set_vof(c0)
    liq = 0.25
    rho_mean = liq * ratio + (1.0 - liq)
    s.set_property_model("rho", "linear", "C", [1.0, ratio - 1.0])
    s.set_property_model("mu", "linear", "C", [0.05, 0.45])
    # ZERO-MEAN buoyancy: a periodic box has no way to absorb a net body force, so `-g rho` would
    # accelerate the whole fluid without bound and there would be no state to measure a drift in.
    s.set_property_model("force_z", "linear", "rho", [grav * rho_mean, -grav])
    if momentum:
        s.enable_vof_momentum(1.0, ratio)
    v0 = s.vof_diagnostics()["volume"]
    maxit, capped, clip = 0, 0, 0.0
    for _ in range(steps):
        s.step()
        it = s.last_pressure_iterations()
        maxit = max(maxit, it)
        capped += it >= PRESS_MAXIT
        clip += s.vof_diagnostics()["clipped_volume"]
    d1 = s.vof_diagnostics()
    umax = max(abs(np.asarray(g())).max() for g in (s.get_u, s.get_v, s.get_w))
    drift = (d1["volume"] - v0) / v0
    print(f"G4 packing {n}^3 draining, ratio {ratio}, {steps} coupled steps, "
          f"momentum consistency {'ON' if momentum else 'off'}")
    print(f"   colour drift {drift:.3e} ({drift / steps:.3e} per step); "
          f"max|div(open u)| {s.max_open_divergence():.3e}")
    print(f"   pressure iters max {maxit} (cap {PRESS_MAXIT}, capped steps {capped})")
    print(f"   min/max C over uncut fluid {d1['min_fluid']:.3e} / {d1['max_fluid']:.17g}; "
          f"clipped volume total {clip:.3e}")
    print(f"   max|u| {umax:.4e}; sum C over solid cells {d1['solid_sum']:.3e}")
    ok = (not math.isnan(umax) and capped == 0 and abs(drift) / steps <= 1e-10
          and d1["solid_sum"] == 0.0)
    print(f"   G4 {'PASS' if ok else 'FAIL (a capped solve makes the run INVALID)'}")
    return ok


# --------------------------------------------------------------------------------------- G5
def g5(diam=24, steps=200, sigma=1.0, mu=0.05, cfl=0.5):
    nx = ny = int(1.7 * diam)
    nz = int(1.2 * diam)
    zw = 3.5                                   # HALF-INTEGER: the wall cells are genuinely cut
    r = 0.5 * diam
    z = (np.arange(nz) + 0.5)[None, None, :]
    sdf = np.asfortranarray(np.broadcast_to(z - zw, (nx, ny, nz)).astype(float).copy())
    cx, cy = nx * 0.5, ny * 0.5
    # 4^3 midpoint subsampling of the FLUID part of each cell (C is a fraction of the fluid volume)
    sub = (np.arange(4) + 0.5) / 4.0
    c0 = np.zeros((nx, ny, nz))
    for a in sub:
        for b in sub:
            for c in sub:
                px = (np.arange(nx) + a)[:, None, None]
                py = (np.arange(ny) + b)[None, :, None]
                pz = (np.arange(nz) + c)[None, None, :]
                fluid = pz >= zw
                inside = ((px - cx) ** 2 + (py - cy) ** 2 + (pz - zw) ** 2 < r * r) & fluid
                c0 += np.where(inside, 1.0, 0.0)
    tot = np.zeros((nx, ny, nz))
    for c in sub:
        pz = (np.arange(nz) + c)[None, None, :]
        tot += np.broadcast_to(np.where(pz >= zw, 16.0, 0.0), (nx, ny, nz))
    c0 = np.asfortranarray(np.where(tot > 0, c0 / np.maximum(tot, 1e-30), 0.0))

    s = pf.Solver(nx, ny, nz)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_solid(sdf, cutcell_pressure=True)
    s.enable_vof()
    s.set_vof(c0)
    s.set_surface_tension(sigma)
    dts = s.capillary_dt()
    s.set_dt(cfl * dts)
    v0 = s.vof_diagnostics()["volume"]
    maxit, capped = 0, 0
    for _ in range(steps):
        s.step()
        it = s.last_pressure_iterations()
        maxit = max(maxit, it)
        capped += it >= PRESS_MAXIT
    cc = np.asarray(s.get_vof())
    d1 = s.vof_diagnostics()
    z0 = int(math.ceil(zw))
    # h: the liquid column on the axis, counting only the FLUID part of the cut wall cell
    frac = np.clip(np.arange(nz) + 1 - zw, 0.0, 1.0)
    h = float((cc[nx // 2, ny // 2, :] * frac).sum())
    a = math.sqrt(float(cc[:, :, z0].sum()) / math.pi)
    theta = math.degrees(2.0 * math.atan2(h, a))
    rc = (a * a + h * h) / (2.0 * h)
    p = np.asarray(s.get_field("p"))
    px = (np.arange(nx) + 0.5)[:, None, None] - cx
    py = (np.arange(ny) + 0.5)[None, :, None] - cy
    pz = (np.arange(nz) + 0.5)[None, None, :] - zw
    rr = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    inner = (rr < 0.55 * r) & (pz > 0.5)
    outer = (rr > 1.5 * r) & (rr < 1.8 * r) & (pz > 0.5)
    dp = float(p[inner].mean() - p[outer].mean())
    umax_open = max(abs(np.asarray(g())[:, :, z0 + 1:]).max() for g in (s.get_u, s.get_v, s.get_w))
    umax_all = max(abs(np.asarray(g())).max() for g in (s.get_u, s.get_v, s.get_w))
    print(f"G5 liquid cap on a CUT flat wall, D/dx = {diam}, {nx}x{ny}x{nz}, {steps} steps at "
          f"{cfl} dt_sigma = {cfl * dts:.5g}")
    print(f"   volume {v0:.6f} -> {d1['volume']:.6f} (drift {(d1['volume'] - v0) / v0:.3e}); "
          f"analytic hemisphere {2 * math.pi * r ** 3 / 3:.6f}")
    print(f"   h {h:.4f}  a {a:.4f}  ->  apparent theta {theta:.3f} deg (target 90)")
    print(f"   cap sphere radius {rc:.4f} (target {r}); Young-Laplace dP {dp:.6f} vs "
          f"2 sigma/R {2 * sigma / rc:.6f} (rel {abs(dp - 2 * sigma / rc) / (2 * sigma / rc):.3e})")
    print(f"   spurious currents: max|u| in the OPEN fluid {umax_open:.3e} -> "
          f"Ca = mu max|u|/sigma = {mu * umax_open / sigma:.3e}; "
          f"max|u| including the wall band {umax_all:.3e}")
    print(f"   (V4 free-droplet reference at the same D/dx: Ca 1.4e-5 ... 2.6e-5)")
    print(f"   pressure iters max {maxit} (capped steps {capped}); "
          f"sum C over solid cells {d1['solid_sum']:.3e}")
    ok = (abs(theta - 90.0) <= 3.0
          and abs(dp - 2 * sigma / rc) / (2 * sigma / rc) <= 0.01
          and d1["solid_sum"] == 0.0 and capped == 0)
    print(f"   G5 {'PASS' if ok else 'FAIL'}")
    return ok


GATES = {"g2": g2, "g4": g4, "g5": g5}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(GATES)
    quick = "--quick" in sys.argv
    results = {}
    for nm in names:
        fn = GATES[nm]
        if quick and nm == "g2":
            results[nm] = fn(n=32, nadv=200)
        elif quick and nm == "g4":
            results[nm] = fn(n=32, steps=150)
        elif quick and nm == "g5":
            results[nm] = fn(diam=16, steps=100)
        else:
            results[nm] = fn()
        print()
    print("summary: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    sys.exit(0 if all(results.values()) else 1)
