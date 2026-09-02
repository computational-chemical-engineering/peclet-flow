#!/usr/bin/env python3
"""WO-P01 (Part II, rungs P0/P1) — phase change in planar form: the gate battery, in Python.

Every gate prints its numbers; nothing is summarised into an adjective.

  p0a  REGRESSION ONLY (rho_g = rho_l = 1, no divergence source). A planar interface under a
       uniform prescribed mdot, 1000 kinematic steps. The interface position must follow
       x(t) = x0 - mdot t / rho_l EXACTLY (it is a plane shift by a constant volume per step,
       so the only error is round-off), the colour must stay in [0, 1], and the run must cross
       twenty cell boundaries — which is what exercises the clip-and-redistribute.

  p0b  THE DIVERGENCE SOURCE at density ratio 100. A CLOSED column (walls on +-x, periodic in
       y/z) with the net vapour production balanced by a prescribed sink plane, so the Poisson
       problem is compatible without an outflow face. WHY CLOSED AND NOT AN OUTFLOW: the
       variable-density outflow OPERATOR is inconsistent by the density ratio until WO-R2 lands
       (`flow/CLAUDE.md`, rung V-BC), so an outflow at ratio 100 would measure that defect, not
       this one. WHY WALLS AND NOT A PERIODIC BOX: in a periodic box nothing anchors the frame,
       the constraint is zero net momentum, and the liquid recoils at
       -u_g rho_g L_g / (rho_l L_l + rho_g L_g) -- a real effect, but it turns the "gas velocity"
       gate into a statement about a near-cancellation. With walls the liquid is at rest and the
       1-D solution is u = 0 up to the deposit cell, u = mdot(1/rho_g - 1/rho_l) beyond it, and
       0 again past the sink.

  p1   THE STEFAN PROBLEM, and the convergence ladder 64/128/256. A vapour layer growing from a
       superheated wall into saturated liquid; mdot from the one-sided pure-cell gradients of a
       transported temperature, interfacial cells pinned at T_sat by the per-cell Dirichlet mask.
       x(t) = 2 lambda sqrt(alpha_g t) with lambda e^{lambda^2} erf(lambda) = St/sqrt(pi).
       The ladder holds the PHYSICAL problem fixed (domain length 1, alpha = 1, St = 1, the layer
       growing from 0.10 to 0.25 of the domain) and refines h = 1/N with the diffusive number
       Fo = alpha dt/h^2 fixed, so dt ~ h^2 and the backward-Euler time error is O(h^2) --
       it cannot contaminate a first- or second-order spatial statement.
       At N = 256 the vapour layer is 64 cells thick at the gate time, which is the WO's rung.
       Run with rho_g = rho_l: the classical Stefan similarity solution has the vapour at rest
       against the wall and the LIQUID moving at mdot(1/rho_g - 1/rho_l), i.e. a ratio /= 1 needs
       an outlet at the liquid end -- the same WO-R2 dependence as p0b, and the thermal/regression
       kernels this rung exists to measure are identical either way.

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_stefan.py [p0a p0b p1 ...]
"""
import math
import sys

import numpy as np

import peclet.flow as pf


def stefan_lambda(st):
    """lambda e^{lambda^2} erf(lambda) = St / sqrt(pi)  (bisection; no scipy dependency)."""
    rhs = st / math.sqrt(math.pi)
    lo, hi = 1e-9, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * math.exp(mid * mid) * math.erf(mid) > rhs:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def planar_colour(nx, ny, nz, xg, liquid_low=True):
    c = np.zeros((nx, ny, nz), order="F")
    for i in range(nx):
        c[i, :, :] = min(1.0, max(0.0, xg - i)) if liquid_low else min(1.0, max(0.0, (i + 1) - xg))
    return c


# ------------------------------------------------------------------ p0a
def gate_p0a(nx=64, ny=4, nz=4, x0=32.25, mdot=0.02, dt=1.0, nsteps=1000):
    s = pf.Solver(nx, ny, nz)
    s.set_rho(1.0)
    s.set_mu(0.01)
    s.set_dt(dt)
    s.enable_vof()
    s.set_vof(planar_colour(nx, ny, nz, x0))
    s.enable_phase_change(1.0, 1.0, 1.0)
    s.set_mass_flux_uniform(mdot)
    worst, lo, hi, clips, redist = 0.0, 1e30, -1e30, 0, 0.0
    for k in range(1, nsteps + 1):
        s.apply_phase_change(dt)
        x = s.get_vof().sum() / (ny * nz)
        worst = max(worst, abs(x - (x0 - mdot * dt * k)))
        d = s.phase_change_diagnostics()
        lo, hi = min(lo, d["min_C"]), max(hi, d["max_C"])
        clips += d["deficit_cells"]
        redist += d["redistributed"]
    print(f"p0a  grid {nx}x{ny}x{nz}, mdot {mdot}, {nsteps} steps")
    print(f"     max |x_num - (x0 - mdot t/rho_l)| = {worst:.3e}      [gate 1e-12]")
    print(f"     C in [{lo!r}, {hi!r}]      cell-crossings clipped {clips}, "
          f"|redistributed| {redist:.6g}")
    return worst < 1e-12 and lo >= 0.0 and hi <= 1.0 and clips > 0


# ------------------------------------------------------------------ p0b
def gate_p0b(nx=64, ny=4, nz=4, rg=1.0, rl=100.0, mdot=0.01, x0=32.25, nsteps=20):
    s = pf.Solver(nx, ny, nz)
    s.set_rho(rg)
    s.set_mu(1e-3)
    s.set_dt(1.0)
    s.set_domain_bc(0, 1)
    s.set_domain_bc(1, 1)
    s.set_pressure_geometry(np.full((nx, ny, nz), 1.0, order="F"))
    s.enable_vof()
    s.set_vof(planar_colour(nx, ny, nz, x0))
    s.set_property_model("rho", "linear", "C", [rg, rl - rg])
    s.enable_phase_change(rg, rl, 1.0)
    s.set_mass_flux_uniform(mdot)
    ug = mdot * (1.0 / rg - 1.0 / rl)
    sink = np.zeros((nx, ny, nz), order="F")
    sink[nx - 6, :, :] = -ug          # A = 1 per interfacial cell on a grid-aligned plane
    s.set_divergence_source(sink)
    s.set_pressure_fcg(True, 400, 1e-12)
    for _ in range(nsteps):
        s.step()
    u, v, w = s.get_u(), s.get_v(), s.get_w()
    src = s.get_field("pc_source") + s.get_field("div_source")
    div = np.zeros_like(u)
    div[:-1] += u[1:] - u[:-1]
    div[:, :-1] += v[:, 1:] - v[:, :-1]
    div[:, :, :-1] += w[:, :, 1:] - w[:, :, :-1]
    dmax = np.abs(div - src)[:-1, :-1, :-1].max()
    plateau = u[35:56, 0, 0]
    liq = np.abs(u[2:30, 0, 0]).max()
    d = s.phase_change_diagnostics()
    print(f"p0b  ratio {rl/rg:g}, closed column + balanced sink, {nsteps} steps")
    print(f"     u_gas = {plateau.mean():.17g}  exact {ug:.17g}  rel {(plateau.mean()-ug)/ug:+.3e}"
          f"   [gate 1e-10]")
    print(f"     plateau spread {plateau.max()-plateau.min():.3e}, max|u_liquid| {liq:.3e}")
    print(f"     interfacial-cell faces u = {u[32,0,0]:.3e}, {u[33,0,0]:.3e}  (the LIQUID velocity)")
    print(f"     max|div(u) - S| = {dmax:.3e}   source sum {d['source_sum']:.6g} into "
          f"{d['source_cells']} cells, fallback {d['fallback_cells']}")
    print(f"     pressure iterations {s.last_pressure_iterations()} / cap 400 "
          f"(a capped solve makes the run INVALID)")
    return abs((plateau.mean() - ug) / ug) < 1e-10 and dmax < 1e-12 \
        and s.last_pressure_iterations() < 400


# ------------------------------------------------------------------ p1
def stefan_run(n, st=1.0, alpha=1.0, x0p=0.10, xep=0.25, fo=0.5, ny=4, nz=4):
    lam = stefan_lambda(st)
    t0 = (x0p / (2 * lam)) ** 2 / alpha
    te = (xep / (2 * lam)) ** 2 / alpha
    d = alpha * n * n                      # diffusivity in CELL units (h = 1/n physically)
    nsteps = int(round((te - t0) / (fo / d)))
    dt = (te - t0) / nsteps
    s = pf.Solver(n, ny, nz)
    s.set_rho(1.0)
    s.set_mu(1e-3)
    s.set_dt(dt)
    s.set_domain_bc(0, 1)
    s.set_domain_bc(1, 1)
    s.set_pressure_geometry(np.full((n, ny, nz), 1.0, order="F"))
    s.enable_vof()
    s.set_vof(planar_colour(n, ny, nz, x0p * n, liquid_low=False))
    t = np.zeros((n, ny, nz), order="F")
    for i in range(n):
        x = (i + 0.5) / n
        t[i, :, :] = (1.0 - math.erf(lam * x / x0p) / math.erf(lam)) if x < x0p else 0.0
    s.add_scalar("T", d, 1, 60)
    s.set_scalar_bc("T", 0, 2, 1.0)        # superheated wall,  T_w - T_sat = 1
    s.set_scalar_bc("T", 1, 2, 0.0)        # saturated far field
    s.set_field("T", t)
    s.enable_phase_change(1.0, 1.0, 1.0)   # h_lv = 1, c_p = 1  =>  St = 1
    s.set_phase_change_thermal("T", 0.0, d, d, 0.0)
    for _ in range(nsteps):
        s.apply_phase_change(dt)
        s.advance_scalars()
    layer = n - s.get_vof().sum() / (ny * nz)
    exact = 2 * lam * math.sqrt(alpha * te) * n
    return layer, exact, nsteps, lam, s.phase_change_diagnostics()


def gate_p1(ns=(64, 128, 256)):
    print("p1   Stefan problem, St = 1, layer 0.10 -> 0.25 of the domain, Fo = 0.5 (dt ~ h^2)")
    errs = []
    for n in ns:
        layer, exact, nsteps, lam, d = stefan_run(n)
        rel = (layer - exact) / exact
        errs.append(abs(rel))
        print(f"     N = {n:4d}  steps {nsteps:5d}  lambda {lam:.6f}  layer {layer:9.5f} cells "
              f"(exact {exact:9.5f})  rel {100*rel:+8.4f} %   C in [{d['min_C']:.2e}, "
              f"{d['max_C']:.6f}]  unresolved {d['unresolved']:.2e}  fallback {d['fallback_cells']}")
    for a, b, na, nb in zip(errs[:-1], errs[1:], ns[:-1], ns[1:]):
        print(f"     observed order {na} -> {nb}: {math.log2(a/b):.3f}")
    if len(errs) > 2:
        print(f"     observed order {ns[0]} -> {ns[-1]} (fit): "
              f"{math.log2(errs[0]/errs[-1])/math.log2(ns[-1]/ns[0]):.3f}")
    print(f"     GATE: |rel| at N = {ns[-1]} is {100*errs[-1]:.4f} %  [gate 0.5 %; "
          f"Malan et al. 2021 report 0.23 %]")
    return errs[-1] < 0.005


if __name__ == "__main__":
    want = sys.argv[1:] or ["p0a", "p0b", "p1"]
    ok = True
    for g in want:
        ok &= {"p0a": gate_p0a, "p0b": gate_p0b, "p1": gate_p1}[g]()
        print()
    print("ALL GATES PASSED" if ok else "SOME GATES FAILED")
    sys.exit(0 if ok else 1)
