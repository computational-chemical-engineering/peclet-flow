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
  (b') pplane: the periodic twin, reported only (§11 third-pass amendment, WO-6a): a TRIPLY PERIODIC
              box with the rational normal m = (2,1,1), n = m/|m|, period P = 1/|m| along n; two
              parallel planes per period, the first at offset 0.3 cell (s0 = n.(1/2,1/2,1/2) + 0.3 h),
              the second half a period further (layer fraction 0.5, coordinator 2026-11-01): phase A
              where frac((n.x - s0)/P) < 1/2, exact cell fractions (gsv1.layered_fractions). The
              initial field is the periodic MAC curl of a random edge potential with each
              component's rho_F-weighted mean removed (the momentum invariant of the periodic box,
              which no step can decay; DECISION, reported); E sums every face. WO-6a' (§6.5): on the
              periodic box the verdicts judge E' = E - sum_c (sum_F rho_F u_c)^2/(2 sum_F rho_F), the
              energy of the M-orthogonal complement of the translations (the inexact default momentum
              solve deposits momentum, which the undamped translations accumulate); E is reported.
              --frac sets the layer fraction, --D the D_hi list, --pairs the PAIRS indices (WO-6f).
mu_A/mu_B = 1e-2, 1e2, 1e-3, 1e3 with rho_A/rho_B = 1e-3, 1e3, 1e-4, 1e4 (the high-mu phase has
mu = rho = 1); D_hi = mu_hi dt/(rho_hi h^2) in {0.1, 1, 10, 100, 1e4}; 2000 steps.
Initial field: the discrete MAC curl of a random edge vector potential (zero on the walls), so it
is exactly solenoidal with zero wall-normal velocity; its amplitude keeps the colour CFL <= 1e-3
(the problem is linear, so the scale is immaterial).
Kinetic energy: E = 1/2 sum_faces rho_F u^2 h^3, rho_F the solver's face density (the mean of the
two cells' rho(C), face_props.hpp) -- the M-norm of the §6 analysis.
PASS (per run): E_n <= E_0 for every n, and E_{n+100}/E_n <= 1 + 1e-10 for every n >= 200.
Pressure (WO-6a, §6.2): the production driver the rho property model selects (Chebyshev), never a
forced CG driver, with rtol 1e-12 and a cap (PCAP) that must never bind: a capped step makes the
run NOT JUDGED. Two growth verdicts per run (§6.2, USER Q15 pending): 'as written' above, and
'floor-excluded', the growth ratio judged only over windows with E_n >= 1e-20 E_0 (the ratio
below that level is reported, not judged).

  nice -n 10 python gsv9.py [--geom plane|film|pplane|both] [--procs 6] [--mode sharp|averaged]   (one line per run + a verdict)
"""
import argparse
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

N, STEPS = 16, 2000
PCAP, PRTOL = 20000, 1e-12  # the Chebyshev cap must never bind (§6.2)
FLOOR = 1e-20  # the Q15 floor: windows with E_n < FLOOR * E_0 are reported, not judged
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


def colour(geom, lowA, frac=0.5):
    if geom == "pplane":
        from gsv1 import layered_fractions
        m = np.array((2.0, 1.0, 1.0))
        nrm = m / np.linalg.norm(m)
        P = 1.0 / float(np.linalg.norm(m))
        s0 = float(nrm @ np.full(3, 0.5)) + 0.3 / N
        return np.asfortranarray(layered_fractions(nrm, s0, P, frac * P, 0.0, N))
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


def solenoidal_periodic(seed):
    """u = curl(psi) on the periodic MAC grid (index units): psi_x at (i+1/2, j, k), psi_y at
    (i, j+1/2, k), psi_z at (i, j, k+1/2); face c index i is the -c face of cell i."""
    r = np.random.default_rng(seed)
    px, py, pz = (r.standard_normal((N, N, N)) for _ in range(3))
    d = lambda a, ax: np.roll(a, -1, axis=ax) - a
    u = d(pz, 1) - d(py, 2)
    v = d(px, 2) - d(pz, 0)
    w = d(py, 0) - d(px, 1)
    div = d(u, 0) + d(v, 1) + d(w, 2)
    assert np.max(np.abs(div)) < 1e-12
    return u, v, w


def rho_face_periodic(rho):
    """The solver's face density: the mean of the two cells, face c index i between cells i-1 and i."""
    return [0.5 * (rho + np.roll(rho, 1, axis=c)) for c in range(3)]


def energy_periodic(vel, rho):
    return sum(0.5 * float(np.sum(rf * np.asarray(vel[c]) ** 2)) for c, rf in enumerate(rho_face_periodic(rho))) / N**3


def energy_periodic_deflated(vel, rho):
    """WO-6a' (§6.5 finding 1): E' = E - sum_c (sum_F rho_F u_c)^2 / (2 sum_F rho_F), the energy of the
    M-orthogonal complement of the three translations (exact lambda = 1 eigenvectors of any
    momentum-conserving step on a periodic box). Same h^3 = 1/N^3 normalisation as energy_periodic."""
    e = energy_periodic(vel, rho)
    for c, rf in enumerate(rho_face_periodic(rho)):
        e -= float(np.sum(rf * np.asarray(vel[c]))) ** 2 / (2.0 * float(np.sum(rf))) / N**3
    return e


def energy(vel, rho):
    e = 0.0
    for c in range(3):
        sl_hi = [slice(None)] * 3; sl_lo = [slice(None)] * 3
        sl_hi[c] = slice(1, N); sl_lo[c] = slice(0, N - 1)
        rf = 0.5 * (rho[tuple(sl_hi)] + rho[tuple(sl_lo)])
        e += 0.5 * float(np.sum(rf * vel[c][tuple(sl_hi)] ** 2))  # the wall faces (index 0) carry 0
    return e / N**3


def run(args):
    geom, muR, rhoR, Dn, mode, frac = args
    import peclet.flow as pf
    lowA = muR < 1.0
    muA, muB = (muR, 1.0) if lowA else (1.0, 1.0 / muR)
    rA, rB = (rhoR, 1.0) if lowA else (1.0, 1.0 / rhoR)
    h = 1.0 / N
    dt = Dn * h * h  # mu_hi = rho_hi = 1
    s = pf.Solver((N, N, N), extent=(1.0, 1.0, 1.0))
    s.set_dt(dt)  # FIRST: the first set_dt pins the internal time scale (set_dt docstring)
    for f in ("-x", "+x", "-y", "+y", "-z", "+z"):
        s.set_domain_bc(f, "periodic" if geom == "pplane" else "wall")
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_pressure_geometry(np.full((N, N, N), 1.0, order="F"))
    s.enable_vof()
    C0 = colour(geom, lowA, frac)
    s.set_vof(C0)
    s.set_property_model("mu", "linear", "C", [muB, muA - muB])
    s.set_property_model("rho", "linear", "C", [rB, rA - rB])
    s.set_advection(False)
    # Keep the driver the rho model selected (Chebyshev); only its tolerance and cap are set.
    s.set_pressure_chebyshev(True, PCAP, PRTOL)
    s.set_vof_viscous_interface(mode)
    rho = rB + (rA - rB) * C0
    if geom == "pplane":
        u, v, w = solenoidal_periodic(12345)
        u, v, w = (a - float(np.sum(rf * a)) / float(np.sum(rf)) for a, rf in zip((u, v, w), rho_face_periodic(rho)))
        en = energy_periodic_deflated  # judged (WO-6a'); E itself is reported
    else:
        u, v, w = solenoidal(12345)
        en = energy
    amp = 1e-3 * h / (dt * max(np.max(np.abs(a)) for a in (u, v, w)))  # colour CFL <= 1e-3
    for c, a in enumerate((u, v, w)):
        s.set_velocity(c, np.asfortranarray(amp * a))
    E = [en([s.get_u(), s.get_v(), s.get_w()], rho)]
    Efull = [energy_periodic([s.get_u(), s.get_v(), s.get_w()], rho)] if geom == "pplane" else None
    cen = {"band_cells": 0, "virtual_cells": 0, "film_cells": 0, "fit_fallbacks": 0}
    capped = 0
    first_cap = None
    pmax = 0
    blow = None
    for n in range(STEPS):
        try:
            s.step()
        except RuntimeError as ex:  # the colour CFL cap: the field has grown ~1e3x in amplitude
            blow = f"step {n + 1}: {str(ex)[:60]}"
            break
        s.set_vof(C0)
        pit = s.diagnostics.last_pressure_iterations()
        pmax = max(pmax, pit)
        if pit >= PCAP:
            capped += 1
            first_cap = n + 1 if first_cap is None else first_cap
        if mode == "sharp":
            c = s.diagnostics.vof_viscous_interface_census()
            for k in cen:
                cen[k] += c[k]
        vel = [s.get_u(), s.get_v(), s.get_w()]
        E.append(en(vel, rho))
        if Efull is not None:
            Efull.append(energy_periodic(vel, rho))
    E = np.array(E)
    rep = ""
    if Efull is not None:  # E reported, E' judged (WO-6a')
        Ef = np.array(Efull)
        lo_ = min(200, max(len(Ef) - 101, 0))
        wf = Ef[lo_ + 100:] / Ef[lo_:-100] if len(Ef) > 101 else np.array([np.nan])
        rep = (f"[judged E'=E-translations; reported E: E_end/E0={Ef[-1] / Ef[0]:.6f} "
               f"max E(n+100)/E(n)={float(np.max(wf)):.6f}] frac={frac:g} ")
    up = np.nonzero(E[1:] > E[:-1])[0]
    first_grow = int(up[0]) + 1 if up.size else None
    tag = "GSV9" + {"plane": "b", "film": "d", "pplane": "b'"}[geom] + ("" if mode == "sharp" else "-AVG")
    timing = (f"{rep}pdriver=chebyshev(rtol={PRTOL:g},cap={PCAP}) max_p_it={pmax} capped_steps={capped} "
              f"first_cap_step={first_cap} first_energy_growth_step={first_grow}")
    if capped:
        return (f"{tag} NOT_JUDGED geom={geom} muA/muB={muR:g} rhoA/rhoB={rhoR:g} D_hi={Dn:g} "
                f"(a capped pressure step) steps_run={len(E) - 1} {timing}", None, None)
    if blow is not None:
        grow = E[1:] / E[:-1]
        return (f"{tag} MISS/MISS geom={geom} muA/muB={muR:g} rhoA/rhoB={rhoR:g} "
                f"D_hi={Dn:g} BLOWUP at {blow} E_last/E0={E[-1] / E[0]:.3e} min_n E_n/E0={float(np.min(E) / E[0]):.3e} "
                f"last-100-step mean growth/step={float(np.exp(np.mean(np.log(grow[-100:])))):.6f} "
                f"{timing}", False, False)
    above = int(np.sum(E[1:] > E[0]))
    lo = min(200, max(len(E) - 101, 0))  # 200 for the 2000-step run; shorter runs are smoke tests
    En, Ew = E[lo:-100], E[lo + 100:]
    win = Ew / np.where(En > 0, En, np.nan)
    gmax = float(np.max(win)) if np.all(En > 0) else float("nan")
    keep = En >= FLOOR * E[0]
    gfl = float(np.max(win[keep])) if np.any(keep) else float("nan")
    gbel = float(np.nanmax(win[~keep])) if np.any(~keep) else float("nan")
    ok = above == 0 and gmax <= 1.0 + 1e-10
    okf = above == 0 and not (gfl > 1.0 + 1e-10)  # no window above the floor: nothing to judge
    vd = lambda b: "PASS" if b else "MISS"
    return (f"{tag} {vd(ok)}/{vd(okf)} geom={geom} "
            f"muA/muB={muR:g} rhoA/rhoB={rhoR:g} D_hi={Dn:g} E_end/E0={E[-1] / E[0]:.3e} "
            f"max_n E_n/E0={float(np.max(E[1:]) / E[0]):.6f} steps_above_E0={above} "
            f"max E(n+100)/E(n),n>=200={gmax:.6f} [E_n>=1e-20E0: {gfl:.6f}, below: {gbel:.6f}] "
            f"band/step={cen['band_cells'] / STEPS:.0f} "
            f"virt/step={cen['virtual_cells'] / STEPS:.0f} film/step={cen['film_cells'] / STEPS:.0f} "
            f"fallbacks={cen['fit_fallbacks']} {timing}", ok, okf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geom", choices=("plane", "film", "pplane", "both"), default="both")
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--mode", choices=("sharp", "averaged"), default="sharp")
    p.add_argument("--frac", type=float, default=0.5)  # pplane layer fraction of phase A (WO-6f: 0.5, 0.75)
    p.add_argument("--D", type=lambda t: tuple(float(x) for x in t.split(",")), default=DS)  # D_hi list
    p.add_argument("--pairs", type=lambda t: tuple(int(x) for x in t.split(",")), default=None)  # PAIRS indices
    a = p.parse_args()
    geoms = ("plane", "film") if a.geom == "both" else (a.geom,)
    pairs = PAIRS if a.pairs is None else tuple(PAIRS[i] for i in a.pairs)
    jobs = [(g, m, r, d, a.mode, a.frac) for g in geoms for (m, r) in pairs for d in a.D]
    res = []
    with Pool(a.procs) as pool:
        for line, ok, okf in pool.imap_unordered(run, jobs):
            print(line, flush=True)
            res.append((ok, okf))
    tag = f"GSV9{'' if a.mode == 'sharp' else '-AVG'}"
    nj = sum(1 for ok, _ in res if ok is None)
    rc = 0
    for k, nm in ((0, "as-written"), (1, "floor-excluded")):
        npass = sum(1 for r in res if r[k] is True)
        verdict = "PASS" if npass == len(res) else ("MISS" if npass + nj < len(res) else "NOT_JUDGED")
        rc |= verdict != "PASS"
        print(f"{tag} {nm} {verdict} {npass}/{len(res)} runs pass, {nj} not judged (geom={a.geom})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
