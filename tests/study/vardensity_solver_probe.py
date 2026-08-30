#!/usr/bin/env python
"""WO-B (VoF rung S0) -- pressure-driver measurement battery on STATIC two-phase coefficients.

Purpose (see `doc/vof_workorders.md` WO-B and `suite/docs/VOF_PLAN.md` section 5, the S-ladder):
decide, from measurement rather than from the single anecdote in
`doc/variable_density_projection.md` section 2, how far up the S-ladder the variable-density
pressure solver work has to climb.  NO VoF is needed: the density field is manufactured and
static, so this probes the *linear solver* on the exact coefficient operator a VoF interface
would present.

What is swept
-------------
  geometry  x  rho-shape  x  edge  x  density ratio  x  driver  x  forcing case

  geometry   `box`   open box, no immersed solid (`set_pressure_geometry`), walls +-z
             `cyl`   immersed cylinder along y, R = 0.25 N (cut-cell IBM)
             `rings` 3 Raschig rings (the regression suite's `hollow_rings` bed) -- a genuinely
                     tortuous cut-cell pore geometry
             `pack`  `data/packing_ring.vti` strided onto the run grid (opt-in, N = 64)
  rho-shape  `slab`  flat film normal to z, heavy below  (the only true hydrostatic rest state)
             `blob`  sphere of heavy fluid at the domain centre
             `tilt`  film with normal (1,0,1)/sqrt2 -- deliberately NOT aligned with the MG
                     coarsening axes
             `const` THE CONTROL: constant density, `set_density_mode` never called, gravity as
                     a uniform body force.  Same geometry, same BCs, same driver, unit
                     coefficient -- so `pack/const` vs `pack/slab@1e4` is exactly the
                     "geometry alone" vs "geometry x coefficient" separation.  Recorded with
                     ratio 1.
  edge       `smooth` tanh over ~2 cells       `sharp` a one-cell step (what a real VoF
                     interface presents to the operator)
  ratio      1e2, 1e3, 1e4
  driver     `cheb` (`set_pressure_chebyshev`)   `pcg` (`set_pressure_pcg`)
             `fcg` (`set_pressure_fcg`, added by WO-C / rung S1) -- the SAME V-cycle-preconditioned
                     CG as `pcg` with the same cap/tolerance/stopping estimate, differing only in
                     the beta recurrence (Polak-Ribiere instead of Fletcher-Reeves).  `fcg` is NOT
                     in the default `--drivers` set, so the WO-B battery reproduces unchanged; ask
                     for it with `--drivers pcg,fcg` (or `cheb,pcg,fcg`).

**Driver-selection trap -- read before changing `run_one`.**  `set_pressure_pcg(on, maxit, rtol)`
**ignores its `on` flag**: `IbmSolver::setPressurePcg(bool /*on*/, ...)` only stores
`pcgMaxit_`/`pcgRtol_` and never clears `useChebyshev_`.  So after `set_density_mode("variable")`
-- which sets `useChebyshev_ = true` -- a `set_pressure_pcg(True, ...)` call leaves the solve on
**Chebyshev**, contradicting the comment at `setDensityMode` ("an explicit set_pressure_pcg AFTER
set_density_mode still wins") and `CLAUDE.md`'s "last set wins".  The only spelling that actually
selects PCG is `set_pressure_chebyshev(False, ...)` (which does write `useChebyshev_`), so that is
what this script does, followed by `set_pressure_pcg` for the cap/tolerance.  The tell that you got
it wrong: the "PCG" iteration counts cap at exactly **120**, which is `chebMaxit_`'s default, not
whatever cap you passed.
  case       `hydro`  quiescent + gravity closure force_z = -g*rho (the hydrostatic setup);
                      periodic x/y, walls +-z
             `lid`    lid-driven flow: walls -x/+x/-z, the +z face translating at U in x,
                      periodic y.
             `per`    THE PRODUCTION CONFIGURATION: fully periodic + a uniform body force in x
                      -- pore-scale flow through the packing, no domain BC anywhere.  This case
                      is where the two drivers swap places, so it is not optional.  It is
                      skipped for (box, const) and (box, slab), where a z-only stratification in
                      a periodic box leaves u* already x-uniform, div(u*) = 0 identically, and
                      the pressure solve returns after 0 iterations -- a degenerate cell, not a
                      measurement.

What is recorded, per configuration
-----------------------------------
  * pressure iterations per step (`last_pressure_iterations()`) -- median / max / total;
  * "final residual" = `max_open_divergence()` = max|div(open*u)| after the step.  This IS the
    residual of the pressure system in the divergence normalisation: the projection solves
    div(open*(rho0/rho_f) grad phi) = div(open*u*) and then subtracts the gradient, so whatever
    divergence survives is exactly what the linear solve failed to remove;
  * wall time: whole step and the `projection` phase alone (device-fenced per-phase timers);
  * converged: whether the driver stayed strictly below its iteration cap on every step.

Both drivers get the SAME rtol (1e-8) and the SAME cap (200), so "hit the cap" means the same
thing for both and a stalling configuration terminates instead of burning the device.

Run:
    OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=<build> \
        python tests/study/vardensity_solver_probe.py --out tests/study/vardensity_solver_probe.json
    ... --pack                 add the `data/packing_ring.vti` sub-sweep (N=64)
    ... --trace                print the PCG residual history of one stalling configuration
    ... --geoms box,cyl        restrict the sweep
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

# --------------------------------------------------------------------------- knobs
N_DEFAULT = 32
N_PACK = 64
STEPS = 20
MAXIT = 200          # same cap for both drivers -- a stall terminates instead of running away
RTOL = 1e-8          # same tolerance for both drivers
MU = 0.01
DT = 1.0
GRAV = 0.1           # hydrostatic case: force_z = -g*rho
ULID = 1.0           # lid case: +z face translating at U in x
FBODY = 1e-3         # per case: uniform body force in x (creeping pore-scale flow)
VEL_SWEEPS = 40      # RB-GS momentum sweeps (velocity MG is disabled under varRho)

RATIOS = (1e2, 1e3, 1e4)
SHAPES = ("slab", "blob", "tilt")
GEOM_ORDER = ("box", "cyl", "rings", "pack")
SHAPE_ORDER = ("const", "slab", "blob", "tilt")
EDGES = ("smooth", "sharp")
DRIVERS = ("cheb", "pcg", "fcg")
DRIVERS_DEFAULT = ("cheb", "pcg")   # WO-B's battery; `fcg` (WO-C) is opt-in via --drivers
CASES = ("hydro", "lid", "per")


# --------------------------------------------------------------------------- geometry (sdf[x,y,z], <0 solid)
def _grid(N):
    g = np.arange(N) + 0.5
    return np.meshgrid(g, g, g, indexing="ij")


def _minimg(d, N):
    return d - N * np.round(d / N)


def sdf_box(N):
    """No immersed solid: an all-fluid field for `set_pressure_geometry`."""
    return None


def sdf_cyl(N):
    """Immersed cylinder along y, centred, R = 0.25 N."""
    X, _, Z = _grid(N)
    c, R = N / 2.0, 0.25 * N
    return np.sqrt((X - c) ** 2 + (Z - c) ** 2) - R


def _hollow_cyl_sdf(X, Y, Z, c, axis, r_out, r_in, H, N):
    """One Raschig ring (hollow cylinder): annulus [r_in,r_out] intersect the slab |axial| <= H/2."""
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    dx = _minimg(X - c[0], N)
    dy = _minimg(Y - c[1], N)
    dz = _minimg(Z - c[2], N)
    z = dx * ax[0] + dy * ax[1] + dz * ax[2]
    rx, ry, rz = dx - z * ax[0], dy - z * ax[1], dz - z * ax[2]
    rho = np.sqrt(rx * rx + ry * ry + rz * rz)
    return np.maximum(np.maximum(r_in - rho, rho - r_out), np.abs(z) - 0.5 * H)


def sdf_rings(N):
    """The regression suite's `hollow_rings` bed: 3 Raschig rings, fixed positions/orientations."""
    rO, rI, H = 0.22 * N, 0.12 * N, 0.34 * N
    rings = [((0.30 * N, 0.32 * N, 0.30 * N), (1, 0, 0)),
             ((0.70 * N, 0.68 * N, 0.55 * N), (0, 1, 0)),
             ((0.45 * N, 0.50 * N, 0.78 * N), (0, 0, 1))]
    X, Y, Z = _grid(N)
    sdf = np.full((N, N, N), 1e30)
    for c, axis in rings:
        sdf = np.minimum(sdf, _hollow_cyl_sdf(X, Y, Z, c, axis, rO, rI, H, N))
    return sdf


def sdf_pack(N):
    """`data/packing_ring.vti` (256^3, physical units) strided onto an N^3 grid, in CELL units."""
    import pyvista as pv
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "data", "packing_ring.vti")
    g = pv.read(path)
    dims = g.dimensions
    a = g.point_data["SDF"].reshape(dims, order="F")
    if dims[0] % N:
        raise SystemExit(f"packing_ring.vti is {dims[0]}^3; N={N} must divide it")
    st = dims[0] // N
    return np.ascontiguousarray(a[::st, ::st, ::st]) / (g.spacing[0] * st)


GEOMS = {"box": sdf_box, "cyl": sdf_cyl, "rings": sdf_rings, "pack": sdf_pack}


# --------------------------------------------------------------------------- manufactured rho
def level_set(shape, N):
    """psi > 0 in the HEAVY fluid, in units of cells (so `psi/w` is a cell-width argument)."""
    X, Y, Z = _grid(N)
    if shape == "slab":            # flat film normal to z, heavy below -> a true rest state
        return N / 2.0 - Z
    if shape == "blob":            # sphere of heavy fluid at the centre
        c, R = N / 2.0, 0.25 * N
        return R - np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2)
    if shape == "tilt":            # film whose normal is NOT a grid/coarsening axis
        c = N / 2.0
        return ((c - X) + (c - Z)) / np.sqrt(2.0)
    raise ValueError(shape)


def rho_field(shape, edge, ratio, N):
    psi = level_set(shape, N)
    if edge == "smooth":           # tanh over ~2 cells (psi/w, w = 1 cell)
        h = 0.5 * (1.0 + np.tanh(psi / 1.0))
    elif edge == "sharp":          # a one-cell step: what a real VoF interface presents
        h = (psi > 0.0).astype(np.float64)
    else:
        raise ValueError(edge)
    return 1.0 + (ratio - 1.0) * h


# --------------------------------------------------------------------------- one configuration
def run_one(geom, shape, edge, ratio, driver, case, N, quiet=True, bottom=None, levels=None):
    from peclet import flow as F

    sdf = GEOMS[geom](N)
    levels = levels or max(2, int(np.floor(np.log2(N))) - 1)

    s = F.Solver(N, N, N)
    s.set_rho(1.0)
    s.set_mu(MU)
    s.set_dt(DT)
    s.set_advection(False)                       # creeping Stokes: isolate the pressure solve
    s.set_velocity_solver_params(VEL_SWEEPS)
    if case == "hydro":                          # periodic x/y, walls +-z
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 1, 0, 0, 0)
    elif case == "lid":                          # walls -x/+x/-z, lid at +z, periodic y
        s.set_domain_bc(0, 1, 0, 0, 0)
        s.set_domain_bc(1, 1, 0, 0, 0)
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 2, ULID, 0.0, 0.0)
    elif case == "per":                          # fully periodic + uniform body force in x
        s.set_body_force(FBODY, 0.0, 0.0)
    else:
        raise ValueError(case)
    s.set_pressure_multigrid(True, levels=levels)
    if bottom:
        s.set_pressure_bottom(bottom)

    if sdf is None:
        s.set_pressure_geometry(np.asfortranarray(np.full((N, N, N), 10.0)))
    else:
        s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True,
                    pressure_coarse="rediscretized")

    if shape == "const":                         # the constant-density control: varRho never on
        ratio = 1.0
        if case == "hydro":
            s.set_body_force(0.0, 0.0, -GRAV)    # uniform gravity replaces the rho closure
    else:
        s.add_field("rho")
        s.set_field("rho", np.asfortranarray(rho_field(shape, edge, ratio, N)))
        s.set_density_mode("variable")           # (defaults the driver to Chebyshev)
        if case == "hydro":
            s.set_property_model("force_z", "linear", "rho", [0.0, -GRAV])

    # Driver selection AFTER set_density_mode. `set_pressure_chebyshev` is the switch (it is the
    # only setter that writes useChebyshev_); `set_pressure_pcg` only carries the cap/tolerance.
    # See the "Driver-selection trap" in the module docstring.
    if driver == "cheb":
        s.set_pressure_pcg(True, MAXIT, RTOL)
        s.set_pressure_chebyshev(True, MAXIT, RTOL)
    elif driver == "pcg":
        s.set_pressure_chebyshev(False, MAXIT, RTOL)
        s.set_pressure_pcg(True, MAXIT, RTOL)
    elif driver == "fcg":
        # set_pressure_fcg DOES honour its flag (it clears useChebyshev_ itself), so unlike the
        # `pcg` branch above no set_pressure_chebyshev(False) dance is needed.
        s.set_pressure_fcg(True, MAXIT, RTOL)
    else:
        raise ValueError(driver)

    iters, divs, t_step, t_proj = [], [], [], []
    t_wall0 = time.perf_counter()
    for _ in range(STEPS):
        s.step()
        iters.append(int(s.last_pressure_iterations()))
        tm = s.last_step_timers()
        t_step.append(float(tm["step"]))
        t_proj.append(float(tm["projection"]))
        divs.append(float(s.max_open_divergence()))
    wall = time.perf_counter() - t_wall0
    umax = max(float(np.abs(s.get_u()).max()), float(np.abs(s.get_v()).max()),
               float(np.abs(s.get_w()).max()))

    # A low iteration count does NOT imply a good solve: PCG's breakdown guard (`pAp <= 1e-300`
    # or a non-finite recurrence scalar in `CutcellMG::solvePCG`) exits early and keeps the last
    # finite iterate.  So health is judged on the PHYSICS: max|div(open*u)| after the step,
    # normalised by the case's own velocity scale.
    it = np.asarray(iters, float)
    uscale = max(umax, {"lid": ULID, "hydro": GRAV * DT, "per": FBODY * DT}[case])
    div_rel = divs[-1] / uscale
    rec = dict(
        geom=geom, shape=shape, edge=edge, ratio=ratio, driver=driver, case=case, N=N,
        levels=levels, bottom=bottom or "auto", steps=STEPS, maxit=MAXIT, rtol=RTOL,
        iters=iters,
        it_median=float(np.median(it)), it_max=int(it.max()), it_min=int(it.min()),
        it_total=int(it.sum()),
        capped_steps=int((it >= MAXIT).sum()),
        converged=bool((it < MAXIT).all()),
        div_final=divs[-1], div_max=float(max(divs)), div_rel=float(div_rel),
        healthy=bool((it < MAXIT).all() and np.isfinite(umax) and div_rel < 1e-8),
        wall_s=wall,
        t_step_median=float(np.median(t_step)), t_proj_median=float(np.median(t_proj)),
        t_proj_min=float(np.min(t_proj)), t_proj_total=float(np.sum(t_proj)),
        umax=umax,
    )
    if not quiet:
        print(f"  {geom:5s} {shape:4s} {edge:6s} r={ratio:<7.0e} {driver:4s} {case:6s} "
              f"it med {rec['it_median']:6.1f} max {rec['it_max']:4d} "
              f"{'CAPPED' if not rec['converged'] else ('ok    ' if rec['healthy'] else 'BADDIV')} "
              f"div/u {rec['div_rel']:.1e}  proj {rec['t_proj_min']*1e3:7.1f} ms "
              f"(med {rec['t_proj_median']*1e3:7.1f})", flush=True)
    return rec


# --------------------------------------------------------------------------- reporting
def markdown(records):
    out = []
    out.append("`proj ms` is the FASTEST of the 20 steps' projection times, not the median: this "
               "host is shared with other agents, and a median picks up their load.\n")
    out.append("`ok` = stayed below the iteration cap AND left max|div(open u)|/u_scale < 1e-8; "
               "`CAP` = hit the cap; `BAD` = finished but the projection did not close.\n")
    # Columns follow whichever drivers are present, in DRIVERS order (cheb, pcg, fcg): a cheb+pcg
    # run reproduces WO-B's table exactly, a --drivers pcg,fcg run gets the FCG trio instead.
    label = {"cheb": "Cheb", "pcg": "PCG", "fcg": "FCG"}
    present = [d for d in DRIVERS if any(r["driver"] == d for r in records)]
    head = ["geom", "shape", "edge", "ratio", "case"]
    for d in present:
        head += [f"{label[d]} its (med/max)", label[d], f"{label[d]} div/u", f"{label[d]} proj ms"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    key = lambda r: (r["geom"], r["shape"], r["edge"], r["ratio"], r["case"])
    idx = {}
    for r in records:
        idx.setdefault(key(r), {})[r["driver"]] = r
    for k in sorted(idx, key=lambda k: (GEOM_ORDER.index(k[0]), SHAPE_ORDER.index(k[1]),
                                        ("smooth", "sharp").index(k[2]), k[3],
                                        CASES.index(k[4]))):
        g, sh, e, ra, ca = k
        def cell(r):
            if r is None:
                return ["-", "-", "-", "-"]
            state = ("ok" if r["healthy"]
                     else (f"**CAP** ({r['capped_steps']}/{r['steps']})"
                           if not r["converged"] else "**BAD**"))
            return [f"{r['it_median']:.0f}/{r['it_max']}", state,
                    f"{r['div_rel']:.1e}", f"{r['t_proj_min']*1e3:.1f}"]
        row = [g, sh, e, f"{ra:.0e}", ca]
        for d in present:
            row += cell(idx[k].get(d))
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def summarize(records):
    """The two questions the S-ladder decision turns on."""
    lines = []
    for ca in CASES:
        for drv in DRIVERS:
            lines.append(f"\ncase {ca}, driver {drv}: median-of-medians / worst-step iterations, "
                         f"aggregated over rho-shape")
            lines.append("| geom | edge | ratio 1e2 | ratio 1e3 | ratio 1e4 |")
            lines.append("|---|---|---|---|---|")
            for g in ("box", "cyl", "rings", "pack"):
                for e in EDGES:
                    cells, any_here = [], False
                    for ra in RATIOS:
                        sel = [r for r in records if r["driver"] == drv and r["geom"] == g
                               and r["edge"] == e and r["ratio"] == ra and r["case"] == ca]
                        if not sel:
                            cells.append("-")
                            continue
                        any_here = True
                        med = np.median([r["it_median"] for r in sel])
                        mx = max(r["it_max"] for r in sel)
                        bad = sum(0 if r["healthy"] else 1 for r in sel)
                        cells.append(f"{med:.0f} / {mx}"
                                     + (f" (**{bad}/{len(sel)} unhealthy**)" if bad else ""))
                    if any_here:
                        lines.append("| " + " | ".join([g, e] + cells) + " |")

    # The separation the S-ladder decision turns on: how much of the iteration count is the
    # GEOMETRY (constant-density control on the same grid) and how much is the rho jump on top.
    lines.append("\ngeometry vs coefficient: median iterations, constant-rho control vs the "
                 "sharp-edge varRho runs")
    lines.append("| geom | case | driver | const rho | sharp 1e2 | sharp 1e3 | sharp 1e4 | "
                 "jump cost (1e4 / const) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for g in GEOM_ORDER:
        for ca in CASES:
            for drv in DRIVERS:
                ctrl = [r for r in records if r["geom"] == g and r["shape"] == "const"
                        and r["case"] == ca and r["driver"] == drv]
                if not ctrl:
                    continue
                c0 = np.median([r["it_median"] for r in ctrl])
                cells = []
                for ra in RATIOS:
                    sel = [r for r in records if r["geom"] == g and r["shape"] != "const"
                           and r["edge"] == "sharp" and r["case"] == ca
                           and r["driver"] == drv and r["ratio"] == ra]
                    cells.append(f"{np.median([r['it_median'] for r in sel]):.0f}" if sel else "-")
                last = [r for r in records if r["geom"] == g and r["shape"] != "const"
                        and r["edge"] == "sharp" and r["case"] == ca and r["driver"] == drv
                        and r["ratio"] == RATIOS[-1]]
                fac = (np.median([r["it_median"] for r in last]) / c0) if (last and c0) else 0.0
                lines.append("| " + " | ".join([g, ca, drv, f"{c0:.0f}"] + cells
                                               + [f"{fac:.2f}x" if fac else "-"]) + " |")
    return "\n".join(lines)


def cheb_overhead(geom, N, ratio=1e3):
    """Quantify what rung S2 (bound amortization) would actually buy.

    Under varRho the coefficient operator is rebuilt every step and `chebBoundsSet_` is
    invalidated with it (`flow_ibm.hpp` project()), so `estimateEigenvalues(..., iters=15, ...)`
    runs EVERY step -- two 15-step power loops, each applying M^{-1}A once, i.e. **30 extra
    V-cycles per step** on top of the ~15-25 the solve itself needs.  Constant density estimates
    the bounds once (step 1) and reuses them, so the A/B below isolates the cost.
    """
    from peclet import flow as F

    def run(varrho, driver):
        sdf = GEOMS[geom](N)
        levels = max(2, int(np.floor(np.log2(N))) - 1)
        s = F.Solver(N, N, N)
        s.set_rho(1.0)
        s.set_mu(MU)
        s.set_dt(DT)
        s.set_advection(False)
        s.set_velocity_solver_params(VEL_SWEEPS)
        s.set_domain_bc(0, 1, 0, 0, 0)
        s.set_domain_bc(1, 1, 0, 0, 0)
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 2, ULID, 0.0, 0.0)
        s.set_pressure_multigrid(True, levels=levels)
        if sdf is None:
            s.set_pressure_geometry(np.asfortranarray(np.full((N, N, N), 10.0)))
        else:
            s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True,
                        pressure_coarse="rediscretized")
        if varrho:
            s.add_field("rho")
            s.set_field("rho", np.asfortranarray(rho_field("slab", "sharp", ratio, N)))
            s.set_density_mode("variable")
        if driver == "cheb":
            s.set_pressure_chebyshev(True, MAXIT, RTOL)
        else:
            s.set_pressure_chebyshev(False, MAXIT, RTOL)   # the actual switch (see the docstring)
            s.set_pressure_pcg(True, MAXIT, RTOL)
        tp, it = [], []
        for _ in range(STEPS):
            s.step()
            tp.append(float(s.last_step_timers()["projection"]))
            it.append(int(s.last_pressure_iterations()))
        return np.asarray(tp), np.asarray(it)

    out = {}
    for varrho in (False, True):
        for drv in ("cheb", "pcg"):
            tp, it = run(varrho, drv)
            k = f"{'varrho' if varrho else 'constrho'}_{drv}"
            out[k] = dict(t_first=float(tp[0]), t_median=float(np.median(tp[1:])),
                          it_first=int(it[0]), it_median=float(np.median(it[1:])))
    c = out["constrho_cheb"]
    est = c["t_first"] - c["t_median"]     # the one-off bound estimate, isolated
    v = out["varrho_cheb"]
    print("\n--- Chebyshev bound-estimation overhead (rung S2's target) ---")
    for k, r in out.items():
        print(f"  {k:14s} step1 {r['t_first']*1e3:7.1f} ms ({r['it_first']:3d} it)   "
              f"steady {r['t_median']*1e3:7.1f} ms ({r['it_median']:5.1f} it)")
    print(f"  isolated bound estimate (const-rho step1 - steady): {est*1e3:.1f} ms")
    print(f"  varRho Chebyshev pays it EVERY step: {est/max(v['t_median'],1e-12)*100:.0f} % of "
          f"its steady projection time; amortizing it (S2) would leave "
          f"{(v['t_median']-est)*1e3:.1f} ms")
    out["estimate_cost_s"] = est
    out["geom"] = geom
    out["N"] = N
    out["ratio"] = ratio
    return out


def trace_pcg(geom, shape, edge, ratio, case, N):
    """Re-run one configuration with the MG residual trace on (PCG only -- Chebyshev has none)."""
    os.environ["PECLET_FLOW_MG_DEBUG"] = "2"
    os.environ["PECLET_FLOW_MG_DEBUG_SOLVES"] = "3"
    print(f"\n--- PCG residual trace: {geom}/{shape}/{edge}/ratio {ratio:.0e}/{case} ---",
          flush=True)
    run_one(geom, shape, edge, ratio, "pcg", case, N, quiet=False)
    os.environ.pop("PECLET_FLOW_MG_DEBUG", None)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "vardensity_solver_probe.json"))
    ap.add_argument("--md", default=None, help="also write the markdown table to this path")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--geoms", default="box,cyl,rings")
    ap.add_argument("--shapes", default="const," + ",".join(SHAPES))
    ap.add_argument("--edges", default=",".join(EDGES))
    ap.add_argument("--ratios", default=",".join(f"{r:.0e}" for r in RATIOS))
    ap.add_argument("--drivers", default=",".join(DRIVERS_DEFAULT),
                    help="cheb, pcg, fcg (comma-separated); default cheb,pcg = WO-B's battery")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--pack", action="store_true",
                    help="append the data/packing_ring.vti sub-sweep at N=64 (shapes slab,blob)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--cheb-overhead", action="store_true",
                    help="also measure the per-step Chebyshev bound-estimation cost (rung S2)")
    ap.add_argument("--overhead-geoms", default="box,rings",
                    help="geometries for --cheb-overhead (independent of --geoms)")
    ap.add_argument("--tag", default="", help="label stored in the JSON (e.g. the backend)")
    ap.add_argument("--bottom", default=None,
                    help="set_pressure_bottom mode ('auto'/'smoother'/'agglomerated'); "
                         "default = the shipped 'auto'")
    ap.add_argument("--levels", type=int, default=None,
                    help="pressure MG depth (default floor(log2 N) - 1)")
    args = ap.parse_args()

    geoms = [g for g in args.geoms.split(",") if g]
    shapes = [s for s in args.shapes.split(",") if s]
    edges = [e for e in args.edges.split(",") if e]
    ratios = [float(r) for r in args.ratios.split(",") if r]
    drivers = [d for d in args.drivers.split(",") if d]
    cases = [c for c in args.cases.split(",") if c]

    def expand(gs, shs, n):
        # `const` is the unit-coefficient control: edge and ratio are meaningless there, so it is
        # emitted once per (geom, driver, case) instead of once per (edge, ratio).
        out = []
        for g in gs:
            for sh in shs:
                ee = [edges[0]] if sh == "const" else edges
                rr = [1.0] if sh == "const" else ratios
                cc = [c for c in cases
                      # degenerate: a z-only rho in a periodic box gives div(u*) == 0 exactly
                      if not (c == "per" and g == "box" and sh in ("const", "slab"))]
                out += [(g, sh, e, ra, d, ca, n)
                        for e in ee for ra in rr for d in drivers for ca in cc]
        return out

    plan = expand(geoms, shapes, args.n)
    if args.pack:
        plan += expand(["pack"], [s for s in ("const", "slab", "blob") if s in shapes], N_PACK)

    print(f"{len(plan)} configurations x {STEPS} steps", flush=True)
    records = []
    t0 = time.perf_counter()
    for i, (g, sh, e, ra, d, ca, n) in enumerate(plan):
        print(f"[{i+1}/{len(plan)}]", end=" ", flush=True)
        records.append(run_one(g, sh, e, ra, d, ca, n, quiet=False,
                               bottom=args.bottom, levels=args.levels))
    wall = time.perf_counter() - t0

    overhead = None
    if args.cheb_overhead:
        overhead = [cheb_overhead(g, args.n)
                    for g in args.overhead_geoms.split(",") if g]

    md = markdown(records)
    print("\n" + md)
    print(summarize(records))
    print(f"\ntotal wall {wall:.1f} s")

    meta = dict(tag=args.tag, steps=STEPS, maxit=MAXIT, rtol=RTOL, mu=MU, dt=DT,
                g=GRAV, u_lid=ULID, vel_sweeps=VEL_SWEEPS, wall_s=wall,
                python=sys.version.split()[0], host=platform.node(),
                argv=" ".join(sys.argv[1:]))
    if os.path.exists(args.out):
        with open(args.out) as f:
            blob = json.load(f)
    else:
        blob = {"runs": []}
    blob.setdefault("runs", []).append({"meta": meta, "records": records,
                                        "cheb_overhead": overhead})
    with open(args.out, "w") as f:
        json.dump(blob, f, indent=1)
    print(f"wrote {args.out} ({len(blob['runs'])} run(s))")
    if args.md:
        with open(args.md, "w") as f:
            f.write(md + "\n" + summarize(records) + "\n")

    if args.trace:
        trace_pcg("box", "slab", "sharp", 1e3, "hydro", args.n)


if __name__ == "__main__":
    main()
