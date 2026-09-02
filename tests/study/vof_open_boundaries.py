#!/usr/bin/env python3
"""VoF rung V-BC (WO-R) — two-phase open boundaries: the physics battery.

The *algebraic* gates of this rung (the out-of-domain mask, the bitwise inertness of the boundary
flux rule, the backflow rule, the inflow property ghost, the variable-density outflow correction
and the exact colour budget under a prescribed divergence-free field) live in
`tests/kokkos/test_vof_bc.cpp` and run in the `vof_bc` ctest. What is here is the part that needs a
COUPLED solve and a number somebody else published:

    budget      WO-R gate G1, COUPLED: the same slug injection driven by step(), so the advecting
                field is the projection's own output and the budget's floor is the projection's
                divergence residual rather than zero. Records the pressure iteration count.
    nusselt     WO-R gate G2: a falling liquid film fed by a prescribed Nusselt inlet profile,
                against the analytical flow rate. Run at density ratio 100 (V2a regime) and at
                1000 with momentum consistency. This is the gate on the 1/rho_f outflow
                correction: `PECLET_FLOW_OUTFLOW_RHO=0` restores the pre-WO-R correction and the
                script reports BOTH columns.
    pool        WO-R gate G3: gas blown over a resting liquid pool at ratio 1000. Gates the pool's
                volume, the velocity it is allowed to pick up, and the inflow ghost DENSITY.

Units: cell size 1, time in seconds, exactly as `vof_surface_tension.py`; the cases here are
stated directly in solver units because the reference (Nusselt) is a similarity solution and only
the dimensionless groups matter.

EVERY gate that reports a functional records the max pressure-iteration count against its cap and
marks a capped run INVALID rather than merely flagging it (shared preamble rule 3b).

Usage:  PYTHONPATH=<build> python tests/study/vof_open_boundaries.py [gate ...] [--quick]
"""
import os
import sys
import time

import numpy as np

import peclet.flow as pf

QUICK = "--quick" in sys.argv
GATES = [a for a in sys.argv[1:] if not a.startswith("--")]


class Solve:
    """A run's solver health, per the shared preamble's rule 3b."""

    def __init__(self, cap):
        self.cap = cap
        self.iters = 0
        self.div = 0.0

    def sample(self, s):
        self.iters = max(self.iters, s.last_pressure_iterations())
        # max_open_divergence() RE-IMPOSES the zero-gradient outflow face before measuring, which
        # both destroys bcCorrectOutflow's correction (so calling it once per step changes the run)
        # and reports a field the solver never used. On an open boundary use the projected sibling.
        self.div = max(self.div, s.max_open_divergence_projected())

    @property
    def valid(self):
        return self.iters < self.cap

    def __str__(self):
        return (f"pressure {self.iters}/{self.cap} "
                f"{'OK' if self.valid else '*** CAPPED -> RUN INVALID ***'}, "
                f"max|div(open u)| {self.div:.2e}")


def to_host(view):
    """`field_view` returns the PADDED buffer — NumPy on a host build, DLPack on a device one."""
    try:
        a = np.asarray(view)
        if a.ndim == 3:
            return a
    except Exception:
        pass
    import cupy as cp
    return cp.asnumpy(cp.from_dlpack(view))


def maxvel(s):
    return max(np.abs(s.get_u()).max(), np.abs(s.get_v()).max(), np.abs(s.get_w()).max())


# --------------------------------------------------------------------------- G1 coupled budget
def gate_budget():
    print("\n" + "=" * 96)
    print("G1 COLOUR BUDGET, COUPLED (WO-R gate G1 driven by step(), not by a prescribed field)")
    print("=" * 96)
    print("  A liquid slug is injected through a uniform -z inlet and flushed out of the +z")
    print("  outlet, with the full momentum + projection step in between. The gate is")
    print("      sum(C)(t) - integral(inflow liquid) + integral(outflow liquid) = const,")
    print("  the fluxes being the ADVECTOR's own boundary face fluxes (vof_bc_volumes). Unlike the")
    print("  kinematic ctest twin, the advecting field here is the projection's output, so the")
    print("  floor is the projection's discrete divergence residual (WO-J finding 3).\n")
    nx = ny = 32 if not QUICK else 16
    nz = 64 if not QUICK else 32
    nslug, nafter = (60, 740) if not QUICK else (20, 120)
    W, dt = 0.5, 0.2
    s = pf.Solver(nx, ny, nz)
    s.set_rho(1.0)
    s.set_mu(0.5)
    s.set_dt(dt)
    for f in range(4):
        s.set_domain_bc(f, 1, 0, 0, 0)
    s.set_domain_bc(4, 2, 0.0, 0.0, W)
    s.set_domain_bc(5, 3, 0, 0, 0)
    # the open-boundary duct settings the validated channel case uses
    # (scripts/verify_channel_sdflow.py): a deep semi-coarsening hierarchy, enough momentum
    # sweeps, and the all-fluid SDF far from every cell. With the defaults this configuration
    # diverges — see the findings entry.
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=8)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((nx, ny, nz), 1e30, order="F"))
    s.set_pressure_chebyshev(True, 400, 1e-12)
    s.enable_vof()
    s.set_vof(np.zeros((nx, ny, nz), order="F"))
    s.set_vof_inflow(4, 1.0)
    s.set_vof_backflow(5, 0.0)
    h = Solve(400)
    ledger, budget0, drift = 0.0, None, 0.0
    cmin, cmax = 1.0, 0.0
    t0 = time.time()
    for i in range(nslug + nafter):
        if i == nslug:
            s.set_vof_inflow(4, 0.0)
        s.step()
        h.sample(s)
        d = s.vof_diagnostics()
        ledger += sum(s.vof_bc_volumes())
        b = d["sum"] - ledger
        if budget0 is None:
            budget0 = b
        drift = max(drift, abs(b - budget0))
        cmin, cmax = min(cmin, d["min"]), max(cmax, d["max"])
    tot = s.vof_bc_volumes_total()
    injected, left = tot[4], -tot[5]
    d = s.vof_diagnostics()
    print(f"  injected {injected:.10g}, left {left:.10g}, remaining sum(C) {d['sum']:.3e}")
    print(f"  |budget drift| {drift:.3e}  (relative to the injected volume {drift/injected:.3e})")
    print(f"  C in [{cmin:.3e}, {cmax:.17g}]")
    print(f"  {h}   ({time.time()-t0:.0f} s)")
    ok = (drift / injected < 1e-10) and h.valid and cmin > -1e-12 and cmax < 1 + 1e-12
    print(f"  GATE: {'PASS' if ok else 'FAIL'} (budget <= 1e-10 relative, C bounded, no cap)")
    return ok


# ------------------------------------------------------------------------ G2 Nusselt falling film
# Both legs keep rho_l = 100, mu_l = 50, delta = 8 and the SAME film Reynolds number 5 and the
# same Nusselt flow rate Q: only the gas is made lighter and the body force is rescaled by
# 99/drho, so the two rows differ in the density RATIO and in nothing else. dt is set by the WY
# Courant cap on the film's own velocity (the same u_max in both rows) with margin for the gas.
NUSSELT = {
    100: dict(rho_l=100.0, rho_g=1.0, mu_l=50.0, mu_g=1.0, dt=0.5, mom=True, cap=400),
    1000: dict(rho_l=100.0, rho_g=0.1, mu_l=50.0, mu_g=1.0, dt=0.25, mom=True, cap=2000),
}


def gate_nusselt(ratio=100, steps=None, outflow_rho=True):
    p = NUSSELT[ratio]
    nx, ny, nz = 32, 4, 64
    delta = 8.0
    drho = p["rho_l"] - p["rho_g"]
    p = dict(p, g=4.93e-3 * 99.0 / drho)
    umax = drho * p["g"] * delta ** 2 / (2 * p["mu_l"])
    Q = drho * p["g"] * delta ** 3 / (3 * p["mu_l"])  # flow rate per unit width, cells^2/s
    Re = p["rho_l"] * umax * delta / p["mu_l"]
    steps = steps or (900 if QUICK else 2400)
    print("\n" + "=" * 96)
    print(f"G2 NUSSELT FALLING FILM, density ratio {ratio}, mu ratio {p['mu_l']/p['mu_g']:.0f}"
          f"{'' if outflow_rho else '   [ABLATION: PECLET_FLOW_OUTFLOW_RHO=0]'}")
    print("=" * 96)
    print("  A liquid film of thickness delta = 8 cells runs down the -x wall of a quasi-2D duct,")
    print("  fed at the +z inlet with the exact Nusselt profile and leaving through the -z outlet.")
    print("  The body force is the REDUCED gravity -(rho_l - rho_g) g C: the gas is then exactly")
    print("  force-free and hydrostatic, which is what the Nusselt similarity solution assumes")
    print("  (with gravity on the gas too, the Dirichlet p=0 outlet would accelerate the gas")
    print("  column instead of supporting it — a different problem).")
    print(f"  u_max {umax:.5g} cells/s, Q {Q:.5g} cells^2/s, film Reynolds {Re:.2f}\n")
    s = pf.Solver(nx, ny, nz)
    s.set_rho(p["rho_l"])
    s.set_mu(p["mu_l"])
    s.set_dt(p["dt"])
    s.set_domain_bc(0, 1, 0, 0, 0)
    s.set_domain_bc(1, 1, 0, 0, 0)  # walls +-x (the film wall and the far wall)
    s.set_domain_bc(4, 3, 0, 0, 0)  # outflow at the BOTTOM
    # inlet velocity profile on face 5 (+z): axes (b, c) = (x, y)
    xc = np.arange(nx) + 0.5
    wprof = np.where(xc < delta, -(drho * p["g"] / p["mu_l"]) * (delta * xc - 0.5 * xc ** 2), 0.0)
    prof = np.zeros((nx, ny, 3))
    prof[:, :, 2] = wprof[:, None]
    s.set_domain_bc_profile(5, np.ascontiguousarray(prof))
    # the open-boundary duct settings the validated channel case uses
    # (scripts/verify_channel_sdflow.py): a deep semi-coarsening hierarchy, enough momentum
    # sweeps, and the all-fluid SDF far from every cell. With the defaults this configuration
    # diverges — see the findings entry.
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=8)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((nx, ny, nz), 1e30, order="F"))
    s.enable_vof()
    # the initial state IS the developed film, so the run only has to hold it
    C = np.zeros((nx, ny, nz), order="F")
    frac = np.clip(delta - np.arange(nx), 0.0, 1.0)  # exact fractions of the x < delta slab
    C[:, :, :] = frac[:, None, None]
    s.set_vof(np.asfortranarray(C))
    s.set_property_model("rho", "linear", "C", [p["rho_g"], p["rho_l"] - p["rho_g"]])
    s.set_property_model("mu", "linear", "C", [p["mu_g"], p["mu_l"] - p["mu_g"]])
    s.set_property_model("force_z", "linear", "C", [0.0, -drho * p["g"]])
    if p["mom"]:
        s.enable_vof_momentum(p["rho_g"], p["rho_l"])
    # THE DRIVER IS SELECTED LAST, AND IT IS FCG. Two measured reasons, both recorded in the
    # findings: (i) `set_property_model("rho", ...)` fires `set_density_mode`, which reselects
    # Chebyshev — a driver chosen BEFORE the closure is silently discarded (this is the WO-H
    # "capped at 120" tell, alive and well); (ii) on this configuration Chebyshev DIVERGES (the
    # film accelerates until the WY cap throws) and MG-PCG burns any cap, while FCG converges.
    s.set_pressure_fcg(True, p["cap"], 1e-11)
    # the inlet carries the same film: colour profile 1 in the film, 0 in the gas
    s.set_vof_inflow_profile(5, np.ascontiguousarray(np.repeat(frac[:, None], ny, axis=1)))
    s.set_vof_backflow(4, 0.0)
    # seed the interior with the analytical velocity so the transient is short
    w0 = np.zeros((nx, ny, nz), order="F")
    w0[:, :, :] = wprof[:, None, None]
    s.set_field("w", np.asfortranarray(w0))

    h = Solve(p["cap"])
    hist = []
    t0 = time.time()
    for i in range(steps):
        s.step()
        h.sample(s)
        hist.append(s.vof_diagnostics()["sum"])
    C = s.get_vof()
    w = s.get_w()
    zmid = nz // 2
    dmeas = float(C[:, :, zmid].sum() / ny)
    qmeas = float(-(C[:, :, zmid] * w[:, :, zmid]).sum() / ny)
    tail = np.array(hist[-200:])
    steady = float(np.abs(tail - tail.mean()).max() / max(tail.mean(), 1e-300))
    print(f"  at z = nz/2:  film thickness {dmeas:.4f} cells (target {delta})  "
          f"[error {dmeas-delta:+.4f}]")
    print(f"                flow rate      {qmeas:.6g} (Nusselt {Q:.6g})  "
          f"[{100*(qmeas/Q-1):+.2f} %]")
    print(f"  sum(C) steadiness over the last 200 steps: {steady:.3e} relative")
    print(f"  {h}   ({time.time()-t0:.0f} s)")
    ok = (abs(qmeas / Q - 1) < 0.03 and abs(dmeas - delta) < 0.5 and steady < 1e-4 and h.valid)
    print(f"  GATE: {'PASS' if ok else 'FAIL'} (Q within 3 %, delta within 0.5 cell, "
          f"sum(C) steady to 1e-4, no cap)")
    return ok, qmeas / Q - 1, dmeas - delta, steady, h.iters


# ------------------------------------------------------------------------- G3 gas over a pool
def gate_pool():
    print("\n" + "=" * 96)
    print("G3 GAS OVER A RESTING POOL, density ratio 1000, momentum-consistent transport")
    print("=" * 96)
    print("  Gas enters at -x above a liquid pool that fills the lower half, and leaves at +x. The")
    print("  pool must not move and must not lose volume, and the inflow ghost DENSITY must be the")
    print("  INLET fluid's — gas above the pool, liquid at the pool's own inlet plane.")
    print("")
    print("  NOTE ON THE GATE AS WRITTEN. WO-R asks for the ghost density 'through get_field(rho)")
    print("  on the first inner plane'. That measures the wrong quantity: get_field returns the")
    print("  INNER cells, whose rho is rho(C_inner) whatever the ghost policy does. The ghost is")
    print("  what the inlet FACE density is built from, and it is reachable only through")
    print("  field_view('rho'), which returns the padded buffer. That is what is gated below.\n")
    nx = 64 if not QUICK else 32
    ny = 4
    nz = 32 if not QUICK else 16
    steps = 500 if not QUICK else 150
    rho_l, rho_g = 1000.0, 1.0
    mu_l, mu_g = 10.0, 0.1
    U = 0.4
    s = pf.Solver(nx, ny, nz)
    s.set_rho(rho_l)
    s.set_mu(mu_l)
    s.set_dt(0.2)
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)  # walls +-z (the pool floor and the lid)
    s.set_domain_bc(1, 3, 0, 0, 0)  # outflow at +x
    # inflow at -x: gas only above the pool. Face 0 -> (b, c) = (y, z).
    zc = np.arange(nz) + 0.5
    gas = zc >= nz / 2
    prof = np.zeros((ny, nz, 3))
    prof[:, :, 0] = np.where(gas, U, 0.0)[None, :]
    s.set_domain_bc_profile(0, np.ascontiguousarray(prof))
    # the open-boundary duct settings the validated channel case uses
    # (scripts/verify_channel_sdflow.py): a deep semi-coarsening hierarchy, enough momentum
    # sweeps, and the all-fluid SDF far from every cell. With the defaults this configuration
    # diverges — see the findings entry.
    s.set_velocity_solver_params(60)
    s.set_pressure_multigrid(True, levels=8)
    s.set_pressure_solver_params(80)
    s.set_pressure_geometry(np.full((nx, ny, nz), 1e30, order="F"))
    s.enable_vof()
    C = np.zeros((nx, ny, nz), order="F")
    C[:, :, : nz // 2] = 1.0
    s.set_vof(np.asfortranarray(C))
    s.set_property_model("rho", "linear", "C", [rho_g, rho_l - rho_g])
    s.set_property_model("mu", "linear", "C", [mu_g, mu_l - mu_g])
    s.enable_vof_momentum(rho_g, rho_l)
    s.set_pressure_fcg(True, 800, 1e-11)  # driver LAST — see gate_nusselt
    # the inlet colour is the pool's own: liquid below, gas above
    cprof = np.where(gas, 0.0, 1.0)
    s.set_vof_inflow_profile(0, np.ascontiguousarray(np.repeat(cprof[None, :], ny, axis=0)))
    s.set_vof_backflow(1, 0.0)

    h = Solve(800)
    v0 = s.vof_diagnostics()["sum"]
    t0 = time.time()
    for _ in range(steps):
        s.step()
        h.sample(s)
    v1 = s.vof_diagnostics()["sum"]
    C = s.get_vof()
    u, v, w = s.get_u(), s.get_v(), s.get_w()
    liq = C > 0.999
    umax_liq = float(np.max([np.abs(f[liq]).max() for f in (u, v, w)]))
    rho = to_host(s.field_view("rho"))
    G = (rho.shape[0] - nx) // 2
    ghost = rho[G - 1, G + ny // 2, :]          # the -x inflow ghost column
    gh_gas = float(np.mean(ghost[G + nz // 2: G + nz]))
    gh_liq = float(np.mean(ghost[G: G + nz // 2]))
    print(f"  pool volume {v0:.10g} -> {v1:.10g}  (relative change {abs(v1-v0)/v0:.3e})")
    print(f"  max|u| in the liquid {umax_liq:.4e}  ({umax_liq/U:.3e} of the gas inlet speed)")
    print(f"  inflow ghost rho: {gh_gas:.6g} above the pool (want {rho_g}), "
          f"{gh_liq:.6g} at the pool (want {rho_l})")
    print(f"  {h}   ({time.time()-t0:.0f} s)")
    ok = (abs(v1 - v0) / v0 < 1e-10 and umax_liq / U < 1e-3 and h.valid
          and abs(gh_gas - rho_g) < 1e-9 * rho_l and abs(gh_liq - rho_l) < 1e-9 * rho_l)
    print(f"  GATE: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    todo = GATES or ["budget", "nusselt", "pool"]
    results = {}
    if "budget" in todo:
        results["budget"] = gate_budget()
    if "nusselt" in todo:
        ablation = os.environ.get("PECLET_FLOW_OUTFLOW_RHO") == "0"
        rows = []
        for r in (100, 1000):
            ok, dq, dd, st, it = gate_nusselt(r, outflow_rho=not ablation)
            results[f"nusselt{r}"] = ok
            rows.append((r, dq, dd, st, it))
        print("\n  ratio   dQ/Q        d(delta)   sum(C) steadiness   pressure iters"
              f"   [1/rho_f outflow correction: {'OFF (ablation)' if ablation else 'ON'}]")
        for r, dq, dd, st, it in rows:
            print(f"  {r:5d}   {100*dq:+7.2f} %   {dd:+8.4f}   {st:.3e}           {it}")
    if "pool" in todo:
        results["pool"] = gate_pool()
    print("\n" + "=" * 96)
    for k, v in results.items():
        print(f"  {k:12s} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
