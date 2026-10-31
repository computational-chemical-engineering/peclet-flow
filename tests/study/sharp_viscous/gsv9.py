"""G-SV9 (b) and (d) (doc/vof_sharp_viscous_stress.md §11): the 3-D homogeneous march of the sharp
viscous split with the production stepper.

Unit box, N = 16, no-slip walls on all six faces (homogeneous Dirichlet), Stokes (momentum
advection off, no body force), sharp viscous interface. The colour is FROZEN: it is reset to the
initial field after every step (the step builds mu, rho and the band from the colour it starts
with; its colour stage at the tail is discarded), so the march is the linear operator of §6.
  (b) plane:  the plane with normal n(22.5, 22.5 deg) through the centre, offset 0.3 cell along n
              (C = 1, phase A, on the side n.(x - x0) < 0);
  (d) film:   a grid-aligned 1-cell slab of the low-mu phase normal to x (cells x = N/2), both of
              its faces carrying virtual planes.
mu_A/mu_B = 1e-2, 1e2, 1e-3, 1e3 with rho_A/rho_B = 1e-3, 1e3, 1e-4, 1e4 (the high-mu phase has
mu = rho = 1); D_hi = mu_hi dt/(rho_hi h^2) in {0.1, 1, 10, 100, 1e4}; 2000 steps.
Initial field: the discrete MAC curl of a random edge vector potential (zero on the walls), so it
is exactly solenoidal with zero wall-normal velocity; its amplitude keeps the colour CFL <= 1e-3
(the problem is linear, so the scale is immaterial).
Kinetic energy: E = 1/2 sum_faces rho_F u^2 h^3, rho_F the solver's face density (the mean of the
two cells' rho(C), face_props.hpp) -- the M-norm of the §6 analysis.
PASS (per run): E_n <= E_0 for every n, and E_{n+100}/E_n <= 1 + 1e-10 for every n >= 200.

  nice -n 10 python gsv9.py [--geom plane|film|both] [--procs 6] [--mode sharp|averaged]   (one line per run + a verdict)
"""
import argparse
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

N, STEPS = 16, 2000
PAIRS = ((1e-2, 1e-3), (1e2, 1e3), (1e-3, 1e-4), (1e3, 1e4))
DS = (0.1, 1.0, 10.0, 100.0, 1e4)


def plane_fractions(n, nrm, x0, sub=8):
    off = (np.arange(sub) + 0.5) / sub
    g = (np.arange(n)[:, None] + off[None, :]).ravel()
    out = np.zeros((n, n, n))
    for k in range(n):
        X, Y, Z = np.meshgrid(g, g, k + off, indexing="ij")
        ins = (nrm[0] * (X - x0[0]) + nrm[1] * (Y - x0[1]) + nrm[2] * (Z - x0[2]) < 0).astype(np.float64)
        out[:, :, k] = ins.reshape(n, sub, n, sub, sub).mean(axis=(1, 3, 4))
    return np.asfortranarray(out)


def colour(geom, lowA):
    if geom == "plane":
        b = g = math.radians(22.5)
        nrm = (math.cos(b) * math.cos(g), math.cos(b) * math.sin(g), math.sin(b))
        return plane_fractions(N, nrm, [0.5 * N + 0.3 * c for c in nrm])
    C = np.zeros((N, N, N)) if lowA else np.ones((N, N, N))
    C[N // 2, :, :] = 1.0 if lowA else 0.0  # the slab is the low-mu phase
    return np.asfortranarray(C)


def solenoidal(seed):
    """u = curl(psi) on the MAC grid, psi on edges, zero on every wall edge (index units)."""
    r = np.random.default_rng(seed)
    px = r.standard_normal((N, N + 1, N + 1))  # psi_x at (i+1/2, j, k)
    py = r.standard_normal((N + 1, N, N + 1))  # psi_y at (i, j+1/2, k)
    pz = r.standard_normal((N + 1, N + 1, N))  # psi_z at (i, j, k+1/2)
    px[:, [0, N], :] = 0.0; px[:, :, [0, N]] = 0.0
    py[[0, N], :, :] = 0.0; py[:, :, [0, N]] = 0.0
    pz[[0, N], :, :] = 0.0; pz[:, [0, N], :] = 0.0
    u = (pz[:, 1:, :] - pz[:, :-1, :]) - (py[:, :, 1:] - py[:, :, :-1])  # (N+1, N, N) at x-faces
    v = (px[:, :, 1:] - px[:, :, :-1]) - (pz[1:, :, :] - pz[:-1, :, :])  # (N, N+1, N)
    w = (py[1:, :, :] - py[:-1, :, :]) - (px[:, 1:, :] - px[:, :-1, :])  # (N, N, N+1)
    div = (u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1]) + (w[:, :, 1:] - w[:, :, :-1])
    assert np.max(np.abs(div)) < 1e-12 and not u[0].any() and not u[N].any()
    return u[:N], v[:, :N], w[:, :, :N]  # the -c face of every cell; the +wall face is 0


def energy(vel, rho):
    e = 0.0
    for c in range(3):
        sl_hi = [slice(None)] * 3; sl_lo = [slice(None)] * 3
        sl_hi[c] = slice(1, N); sl_lo[c] = slice(0, N - 1)
        rf = 0.5 * (rho[tuple(sl_hi)] + rho[tuple(sl_lo)])
        e += 0.5 * float(np.sum(rf * vel[c][tuple(sl_hi)] ** 2))  # the wall faces (index 0) carry 0
    return e / N**3


def run(args):
    geom, muR, rhoR, Dn, mode = args
    import peclet.flow as pf
    lowA = muR < 1.0
    muA, muB = (muR, 1.0) if lowA else (1.0, 1.0 / muR)
    rA, rB = (rhoR, 1.0) if lowA else (1.0, 1.0 / rhoR)
    h = 1.0 / N
    dt = Dn * h * h  # mu_hi = rho_hi = 1
    s = pf.Solver((N, N, N), extent=(1.0, 1.0, 1.0))
    s.set_dt(dt)  # FIRST: the first set_dt pins the internal time scale (set_dt docstring)
    for f in ("-x", "+x", "-y", "+y", "-z", "+z"):
        s.set_domain_bc(f, "wall")
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_pressure_geometry(np.full((N, N, N), 1.0, order="F"))
    s.enable_vof()
    C0 = colour(geom, lowA)
    s.set_vof(C0)
    s.set_property_model("mu", "linear", "C", [muB, muA - muB])
    s.set_property_model("rho", "linear", "C", [rB, rA - rB])
    s.set_advection(False)
    s.set_pressure_pcg(True, 800, 1e-12)
    s.diagnostics.set_vof_viscous_interface(mode)
    u, v, w = solenoidal(12345)
    amp = 1e-3 * h / (dt * max(np.max(np.abs(a)) for a in (u, v, w)))  # colour CFL <= 1e-3
    for c, a in enumerate((u, v, w)):
        s.set_velocity(c, np.asfortranarray(amp * a))
    rho = rB + (rA - rB) * C0
    E = [energy([s.get_u(), s.get_v(), s.get_w()], rho)]
    cen = {"band_cells": 0, "virtual_cells": 0, "film_cells": 0, "fit_fallbacks": 0}
    capped = 0
    first_cap = None
    blow = None
    for n in range(STEPS):
        try:
            s.step()
        except RuntimeError as ex:  # the colour CFL cap: the field has grown ~1e3x in amplitude
            blow = f"step {n + 1}: {str(ex)[:60]}"
            break
        s.set_vof(C0)
        if s.diagnostics.last_pressure_iterations() >= 800:
            capped += 1
            first_cap = n + 1 if first_cap is None else first_cap
        if mode == "sharp":
            c = s.diagnostics.vof_viscous_interface_census()
            for k in cen:
                cen[k] += c[k]
        E.append(energy([s.get_u(), s.get_v(), s.get_w()], rho))
    E = np.array(E)
    up = np.nonzero(E[1:] > E[:-1])[0]
    first_grow = int(up[0]) + 1 if up.size else None
    tag = "GSV9" + ("b" if geom == "plane" else "d") + ("" if mode == "sharp" else "-AVG")
    timing = f"first_pcg_cap_step={first_cap} first_energy_growth_step={first_grow}"
    if blow is not None:
        grow = E[1:] / E[:-1]
        return (f"{tag} MISS geom={geom} muA/muB={muR:g} rhoA/rhoB={rhoR:g} "
                f"D_hi={Dn:g} BLOWUP at {blow} E_last/E0={E[-1] / E[0]:.3e} min_n E_n/E0={float(np.min(E) / E[0]):.3e} "
                f"last-100-step mean growth/step={float(np.exp(np.mean(np.log(grow[-100:])))):.6f} "
                f"pcg_capped_steps={capped} {timing}", False)
    above = int(np.sum(E[1:] > E[0]))
    lo = min(200, max(len(E) - 101, 0))  # 200 for the 2000-step run; shorter runs are smoke tests
    gmax = float(np.max(E[lo + 100:] / E[lo:-100])) if np.all(E[lo:-100] > 0) else float("nan")
    ok = above == 0 and gmax <= 1.0 + 1e-10
    return (f"{tag} {'PASS' if ok else 'MISS'} geom={geom} "
            f"muA/muB={muR:g} rhoA/rhoB={rhoR:g} D_hi={Dn:g} E_end/E0={E[-1] / E[0]:.3e} "
            f"max_n E_n/E0={float(np.max(E[1:]) / E[0]):.6f} steps_above_E0={above} "
            f"max E(n+100)/E(n),n>=200={gmax:.6f} band/step={cen['band_cells'] / STEPS:.0f} "
            f"virt/step={cen['virtual_cells'] / STEPS:.0f} film/step={cen['film_cells'] / STEPS:.0f} "
            f"fallbacks={cen['fit_fallbacks']} pcg_capped_steps={capped} {timing}", ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geom", choices=("plane", "film", "both"), default="both")
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--mode", choices=("sharp", "averaged"), default="sharp")
    a = p.parse_args()
    geoms = ("plane", "film") if a.geom == "both" else (a.geom,)
    jobs = [(g, m, r, d, a.mode) for g in geoms for (m, r) in PAIRS for d in DS]
    res = []
    with Pool(a.procs) as pool:
        for line, ok in pool.imap_unordered(run, jobs):
            print(line, flush=True)
            res.append((line, ok))
    nfail = sum(1 for _, ok in res if not ok)
    print(f"GSV9{'' if a.mode == 'sharp' else '-AVG'} {'PASS' if nfail == 0 else 'MISS'} {len(res) - nfail}/{len(res)} runs pass (geom={a.geom})")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
