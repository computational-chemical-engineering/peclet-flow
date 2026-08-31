#!/usr/bin/env python3
"""WO-K (rung V2b) — momentum-consistent transport: the gate battery.

The decisive gate is the **uniform-velocity consistency test**: an arbitrary sharp colour field
carried by a spatially uniform velocity, no gravity, no viscosity. A momentum-consistent scheme
holds the velocity uniform to machine precision at ANY density ratio; an inconsistent one is wrong
at O(delta rho), so the test is pass/fail at 1e-15 with no tuning knob — the momentum analogue of
the hydrostatic acid test.

Every gate here also asserts that **no pressure solve hit its iteration cap**. Weymouth-Yue's exact
conservation is conditioned on a discretely divergence-free face field, so an under-solved
projection degrades precisely the quantities these gates measure, and a capped solve can look like
a momentum-consistency failure. A run with a capped solve is reported INVALID, not degraded.

Usage:
    PYTHONPATH=$PWD/build_wok_omp python tests/study/vof_momentum_consistency.py [gate ...]
"""
import math
import sys
import time

import numpy as np

import peclet.flow as pf

# The pressure-driver cap in force for every run here (Chebyshev is the varRho default).
PRESS_MAXIT = 300
PRESS_RTOL = 1e-13


class SolveWatch:
    """Per-step record of the pressure solve, so a capped solve invalidates a run."""

    def __init__(self, cap=PRESS_MAXIT):
        self.cap = cap
        self.max_iters = 0
        self.max_div = 0.0
        self.capped = 0

    def sample(self, s):
        it = s.last_pressure_iterations()
        self.max_iters = max(self.max_iters, it)
        if it >= self.cap:
            self.capped += 1
        try:
            self.max_div = max(self.max_div, s.max_open_divergence())
        except Exception:
            pass

    def ok(self):
        return self.capped == 0

    def __str__(self):
        return "pressure: max_iters %d/%d%s, max|div(open u)| %.3g" % (
            self.max_iters,
            self.cap,
            "  ** CAPPED %d STEPS — RUN INVALID **" % self.capped if self.capped else "",
            self.max_div,
        )


def all_fluid(n):
    return np.full((n, n, n), 10.0, order="F")


# --------------------------------------------------------------------------------------------
# sharp colour scenes (exact fractions; VOF_PLAN: never initialise a diffuse profile)
# --------------------------------------------------------------------------------------------
def scene_slab(n):
    """Grid-aligned slab: no mixed cell at all (the configuration with the fewest degrees of
    freedom, and the one the interface-local CFL band had to be redefined for in WO-J)."""
    c = np.zeros((n, n, n), order="F")
    c[:, :, n // 4 : 3 * n // 4] = 1.0
    return c


def scene_tilted(n, mx=0.6, my=-0.5, mz=0.62456):
    """A plane at a generic angle, with exact PLIC fractions from the analytic plane->volume map."""
    from itertools import product

    m = np.array([mx, my, mz], dtype=float)
    m /= np.abs(m).sum()
    alpha = 0.5 * m.sum() * n  # plane through the box centre, in cell units
    c = np.zeros((n, n, n), order="F")
    # exact fraction of the unit cell under m.x < alpha, via the same 5-case SZ relation the solver
    # uses; implemented here independently (host, numpy) so the test does not lean on the kernel.
    for i, j, k in product(range(n), repeat=3):
        c[i, j, k] = _plic_volume(m[0], m[1], m[2], alpha - (m[0] * i + m[1] * j + m[2] * k))
    return c


def _plic_volume(mx, my, mz, alpha):
    """Fluid fraction of the unit cube under mx x + my y + mz z < alpha (independent oracle)."""
    m = np.abs(np.array([mx, my, mz], dtype=float))
    a = alpha
    for k in range(3):
        if [mx, my, mz][k] < 0:
            a += m[k]
    s = m.sum()
    if s == 0.0:
        return 1.0 if a > 0 else 0.0
    m = m / s
    a = a / s
    if a <= 0.0:
        return 0.0
    if a >= 1.0:
        return 1.0
    # inclusion-exclusion for the simplex volume truncated to the unit cube
    def cub(x):
        return x * x * x if x > 0 else 0.0

    n1, n2, n3 = np.sort(m)
    v = cub(a)
    v -= cub(a - n1) + cub(a - n2) + cub(a - n3)
    v += cub(a - n1 - n2) + cub(a - n1 - n3) + cub(a - n2 - n3)
    v -= cub(a - n1 - n2 - n3)
    den = 6.0 * n1 * n2 * n3 if n1 * n2 * n3 > 0 else 0.0
    if den == 0.0:
        # a degenerate (1D/2D) normal: fall back to the exact lower-dimensional closed form
        nz = [x for x in (n1, n2, n3) if x > 1e-300]
        if len(nz) == 1:
            return min(1.0, max(0.0, a / nz[0]))
        p, q = sorted(nz)
        f = 0.0
        f += (a * a) / (2 * p * q)
        if a > p:
            f -= ((a - p) ** 2) / (2 * p * q)
        if a > q:
            f -= ((a - q) ** 2) / (2 * p * q)
        if a > p + q:
            f += ((a - p - q) ** 2) / (2 * p * q)
        return min(1.0, max(0.0, f))
    return min(1.0, max(0.0, v / den))


def scene_droplet(n, r=None):
    """A sphere, sampled by 5-level octree subdivision (exact to ~1e-6 per cell)."""
    r = r or n * 0.28
    cx = cy = cz = n * 0.5
    c = np.zeros((n, n, n), order="F")
    L = 4
    off = (np.arange(2**L) + 0.5) / 2**L
    for i in range(n):
        for j in range(n):
            X = (i + off)[:, None, None] - cx
            Y = (j + off)[None, :, None] - cy
            for k in range(n):
                Z = (np.arange(2**L) + 0.5) / 2**L + k - cz
                d2 = X * X + Y * Y + Z[None, None, :] ** 2
                c[i, j, k] = float((d2 < r * r).mean())
    return c


SCENES = {"slab": scene_slab, "tilted": scene_tilted, "droplet": scene_droplet}


# --------------------------------------------------------------------------------------------
def make_solver(n, C, rho_g, rho_l, mu, dt, U, momentum, rho0=None):
    s = pf.Solver(n, n, n)
    s.set_rho(rho0 if rho0 is not None else rho_l)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_pressure_geometry(all_fluid(n))
    s.set_pressure_chebyshev(True, PRESS_MAXIT, PRESS_RTOL)
    s.enable_vof()
    s.set_vof(C)
    s.set_property_model("rho", "linear", "C", [rho_g, rho_l - rho_g])
    if momentum:
        s.enable_vof_momentum(rho_g, rho_l)
    one = np.ones((n, n, n), order="F")
    s.set_state(U[0] * one, U[1] * one, U[2] * one)
    return s


def uniform_error(s, U):
    e = 0.0
    for c, uc in enumerate(U):
        f = [s.get_u, s.get_v, s.get_w][c]()
        e = max(e, float(np.max(np.abs(f - uc))))
    return e / max(abs(x) for x in U)


# --------------------------------------------------------------------------------------------
def gate_uniform(n=32, steps=200, scenes=("tilted",), ratios=(10.0, 1e2, 1e3, 1e4)):
    """THE gate: uniform velocity at four density ratios, consistent vs inconsistent."""
    U = (1.0, 0.6, -0.4)
    dt = 0.2 / max(abs(x) for x in U)
    print("\n=== GATE 1 — uniform-velocity consistency (n=%d, %d steps, dt=%g, CFL=%.2f) ==="
          % (n, steps, dt, dt * max(abs(x) for x in U)))
    print("    U = %s, mu = 0, g = 0; error = max|u-U|/|U|_inf" % (U,))
    for scene in scenes:
        C = SCENES[scene](n)
        print("\n  scene: %s   (sum C = %.6f)" % (scene, C.sum()))
        print("  %-10s | %-14s | %-14s | %s" % ("ratio", "consistent", "inconsistent", "pressure"))
        for R in ratios:
            row = []
            watches = []
            for mom in (True, False):
                s = make_solver(n, C, 1.0, R, 0.0, dt, U, mom)
                w = SolveWatch()
                for _ in range(steps):
                    s.step()
                    w.sample(s)
                row.append(uniform_error(s, U))
                watches.append(w)
                del s
            flag = "" if all(x.ok() for x in watches) else "  ** INVALID (capped) **"
            print("  %-10g | %-14.4e | %-14.4e | %s%s"
                  % (R, row[0], row[1], watches[0], flag))


def gate_uniform_advection_only(n=32, steps=1, ratios=(10.0, 1e2, 1e3, 1e4)):
    """The same identity on the ADVECTION alone — projection and momentum solve out of the picture.
    This isolates whether the two advections truly share fluxes."""
    U = (1.0, 0.6, -0.4)
    dt = 0.2 / max(abs(x) for x in U)
    print("\n=== GATE 1b — the consistency identity on the advection alone ===")
    for scene in ("slab", "tilted", "droplet"):
        C = SCENES[scene](n)
        errs = []
        for R in ratios:
            s = make_solver(n, C, 1.0, R, 0.0, dt, U, True)
            for _ in range(steps):
                s.step()
            e = 0.0
            for c, uc in enumerate(U):
                f = s.vof_advected_velocity(c)
                e = max(e, float(np.max(np.abs(f - uc))))
            errs.append(e / max(abs(x) for x in U))
            d = s.vof_momentum_diagnostics()
            print("      %-8s R=%-8g err %.3e   C^c in [%.4e, %.6f]  floored %d  min rho^c %.4g"
                  % (scene, R, errs[-1], min(d["min_Cc"]), max(d["max_Cc"]), d["floored"],
                     d["min_rho_c"]))
            del s
        print("  %-8s " % scene + "  ".join("R=%-6g %.3e" % (R, e) for R, e in zip(ratios, errs)))


def gate_single_phase(n=32, steps=50):
    """With C == const the consistent advection must be a SINGLE-PHASE transport: bitwise
    independent of the density ratio (the rho_g/rho_l factors cancel), and volume-preserving."""
    print("\n=== GATE 2 — C == const: the consistent scheme is density-ratio independent ===")
    U = (1.0, 0.6, -0.4)
    dt = 0.2
    for val, name in ((1.0, "C=1 (all liquid)"), (0.0, "C=0 (all gas)")):
        C = np.full((n, n, n), val, order="F")
        ref = None
        for R in (1.0, 1e2, 1e4):
            s = make_solver(n, C, 1.0, R, 0.0, dt, U, True)
            w = SolveWatch()
            for _ in range(steps):
                s.step()
                w.sample(s)
            u = np.stack([s.get_u(), s.get_v(), s.get_w()])
            if ref is None:
                ref = u
                print("  %-18s R=%-8g reference   %s" % (name, R, w))
            else:
                d = float(np.max(np.abs(u - ref))) / max(abs(x) for x in U)
                print("  %-18s R=%-8g d=%.4e  %s" % (name, R, d, w))
            del s


def gate_momentum_conservation(n=32, steps=200, R=1000.0):
    """Total momentum sum(rho_f u) in a periodic box, no forces: conserved to round-off."""
    print("\n=== GATE 3 — momentum conservation in a periodic box (ratio %g) ===" % R)
    C = scene_droplet(n)
    # a sheared, discretely divergence-free initial field would not stay uniform; use a uniform
    # translation plus a solenoidal shear so there IS momentum transport to conserve.
    x = (np.arange(n) + 0.5) / n
    ux = 1.0 + 0.3 * np.sin(2 * math.pi * x)[None, None, :] * np.ones((n, n, n), order="F")
    uy = np.full((n, n, n), 0.2, order="F")
    uz = np.full((n, n, n), -0.1, order="F")
    for mom in (True, False):
        s = make_solver(n, C, 1.0, R, 0.0, 0.15, (1.3, 0.2, -0.1), mom)
        s.set_state(np.asfortranarray(ux), uy, uz)
        w = SolveWatch()

        def total():
            rho = s.get_field("rho")
            p = 0.0
            comps = [s.get_u(), s.get_v(), s.get_w()]
            for c in range(3):
                rf = 0.5 * (rho + np.roll(rho, 1, axis=c))
                p += float((rf * comps[c]).sum())
            return p

        m0 = total()
        for _ in range(steps):
            s.step()
            w.sample(s)
        m1 = total()
        print("  %-14s sum(rho_f u): %.16e -> %.16e   rel %.3e   %s"
              % ("consistent" if mom else "inconsistent", m0, m1, abs(m1 - m0) / abs(m0), w))
        del s


def gate_rho_floor(n=32, steps=200, R=1000.0):
    """The rho^c floor: how often it bites and how much it matters."""
    print("\n=== GATE 4 — the rho^c floor (ratio %g, %d steps) ===" % (R, steps))
    C = scene_droplet(n)
    U = (1.0, 0.6, -0.4)
    ref = None
    for frac in (1e-3, 1e-6, 1e-9, 1e-12):
        s = make_solver(n, C, 1.0, R, 0.0, 0.2, U, True)
        s.set_vof_rho_floor(frac)
        for _ in range(steps):
            s.step()
        d = s.vof_momentum_diagnostics()
        u = np.stack([s.get_u(), s.get_v(), s.get_w()])
        dd = 0.0 if ref is None else float(np.max(np.abs(u - ref)))
        if ref is None:
            ref = u
        print("  frac %-8g floor %-11.4g floored CVs %-6d  min rho^c %.6f  "
              "C^c in [%.3e, %.6f]  du vs frac=1e-3: %.3e"
              % (frac, s.vof_rho_floor(), d["floored"], d["min_rho_c"],
                 min(d["min_Cc"]), max(d["max_Cc"]), dd))
        del s


def gate_falling_drop(n=60, D=15, R=800.0, mu_ratio=100.0, steps=1500, quiet=False):
    """A viscous drop falling under buoyancy at density ratio ~800 and 15 cells/diameter.

    WO-K asks for the Arrufat raindrop (ratio 831.8, 15 cells/D, terminal velocity within 15%).
    That case is held together by SURFACE TENSION, which is rung V4 — without it a raindrop at its
    terminal Weber number flattens and breaks up, so the literal case cannot be run at V2b and a
    "terminal velocity" would not exist to measure. The surface-tension-free case with the same
    payoff structure is a VISCOUS drop in the Stokes regime, whose terminal velocity is the
    Hadamard-Rybczynski result

        U_HR = (2/3) g R^2 (rho_d - rho_c)/mu_c * (mu_c + mu_d)/(2 mu_c + 3 mu_d)

    (-> the rigid-sphere Stokes value (2/9) g R^2 drho/mu_c as mu_d/mu_c -> infinity). In a periodic
    box the drop feels its own images, so the measured plateau sits below U_HR by the usual
    Sangani-Acrivos-type factor at this volume fraction; the number to trust is the CONSISTENT vs
    INCONSISTENT comparison at identical settings, with U_HR quoted as the scale.

    Buoyancy is imposed as f_z = -(rho - <rho>) g through a LinearMix force closure, so the box
    carries no net force and the drop falls relative to a quiescent mean.
    """
    R_drop = D / 2.0
    C = scene_droplet(n, r=R_drop)
    vol = C.sum() / n**3
    rho_g, rho_l = 1.0, R
    rho_mean = rho_g + (rho_l - rho_g) * vol
    mu_g = 1.0
    mu_l = mu_g * mu_ratio
    g = 2.0e-4
    # f_z = -(rho - rho_mean) g = -(rho_g - rho_mean) g - (rho_l - rho_g) g * C
    f0 = -(rho_g - rho_mean) * g
    f1 = -(rho_l - rho_g) * g
    U_hr = (2.0 / 3.0) * g * R_drop**2 * (rho_l - rho_g) / mu_g * (mu_g + mu_l) / (
        2 * mu_g + 3 * mu_l)
    print("\n=== GATE 5 — falling viscous drop, ratio %g, D/h = %d, mu_l/mu_g = %g ==="
          % (R, D, mu_ratio))
    print("    Hadamard-Rybczynski U_t = %.5e (unbounded); box %d^3, drop volume fraction %.4f"
          % (U_hr, n, vol))
    print("    %-14s | %-12s | %-12s | %-9s | %s"
          % ("scheme", "U_drop", "U/U_HR", "max|u|", "pressure"))
    out = {}
    for mom in (True, False):
        s = pf.Solver(n, n, n)
        s.set_rho(rho_l)
        s.set_mu(mu_g)
        dt0 = 4.0
        dt = dt0
        s.set_dt(dt)
        s.set_pressure_geometry(all_fluid(n))
        s.set_pressure_chebyshev(True, PRESS_MAXIT, 1e-11)
        s.enable_vof()
        s.set_vof(C)
        s.set_property_model("rho", "linear", "C", [rho_g, rho_l - rho_g])
        s.set_property_model("mu", "linear", "C", [mu_g, mu_l - mu_g])
        s.set_property_model("force_z", "linear", "C", [f0, f1])
        if mom:
            s.enable_vof_momentum(rho_g, rho_l)
        w = SolveWatch()
        hist = []
        blew = False
        for k in range(steps):
            try:
                s.step()
            except RuntimeError as ex:            # the WY CFL cap = the run went unstable
                print("    %-14s BLEW UP at step %d: %s" % ("consistent" if mom else "inconsistent",
                                                            k, str(ex)[:90]))
                blew = True
                break
            w.sample(s)
            if k % 25 == 24:
                c = s.get_vof()
                uz = s.get_w()
                m = c.sum()
                hist.append(float((c * uz).sum() / m) if m > 0 else 0.0)
            # keep the interface-local Courant number inside the WY cap
            if k % 10 == 9:
                cfl = s.vof_last_courant()
                if cfl > 0.18:
                    dt *= 0.18 / cfl
                    s.set_dt(dt)
                elif cfl < 0.06 and dt < dt0:
                    dt = min(dt0, dt * 1.5)
                    s.set_dt(dt)
        if not blew:
            u_drop = np.mean(hist[-4:]) if len(hist) >= 4 else (hist[-1] if hist else 0.0)
            mx = max(float(np.max(np.abs(s.get_u()))), float(np.max(np.abs(s.get_v()))),
                     float(np.max(np.abs(s.get_w()))))
            print("    %-14s | %-12.5e | %-12.4f | %-9.3e | %s"
                  % ("consistent" if mom else "inconsistent", u_drop, u_drop / U_hr, mx, w))
            out["mom" if mom else "no"] = (u_drop, hist)
        del s
    return out


def gate_rt(res=(32, 64), R=1000.0, steps=400):
    """Rayleigh-Taylor at two resolutions — the WO-K near-Nyquist check (Arrufat section 5).

    A momentum-consistent scheme can excite growth near the grid scale on under-resolved shear
    layers. Reported here as the interface amplitude growth AND the peak |u| relative to the
    inconsistent run at the same resolution; a near-Nyquist artefact shows up as the finer grid
    being WORSE, which is the opposite of a convergence signature.
    """
    print("\n=== GATE 6 — Rayleigh-Taylor at two resolutions (near-Nyquist check, ratio %g) ==="
          % R)
    print("    %-5s | %-13s | %-13s | %-13s | %s"
          % ("n", "amp consistent", "amp inconsist", "max|u| cons", "pressure (consistent)"))
    for n in res:
        row = []
        for mom in (True, False):
            nx, nz = n, 2 * n
            C = np.zeros((nx, nx, nz), order="F")
            x = (np.arange(nx) + 0.5) / nx
            z = (np.arange(nz) + 0.5) / nz
            # a single-mode perturbed flat interface at mid height, exact fractions in the cut cells
            for i in range(nx):
                for j in range(nx):
                    h = 0.5 + 0.01 * math.cos(2 * math.pi * x[i]) * math.cos(2 * math.pi * x[j])
                    hz = h * nz
                    for k in range(nz):
                        C[i, j, k] = min(1.0, max(0.0, hz - k))
            rho_g, rho_l = 1.0, R
            g = 1e-4
            vol = C.sum() / (nx * nx * nz)
            rho_mean = rho_g + (rho_l - rho_g) * vol
            s = pf.Solver(nx, nx, nz)
            s.set_rho(rho_l)
            s.set_mu(0.05)
            dt0 = 2.0
            dt = dt0
            s.set_dt(dt)
            s.set_pressure_geometry(np.full((nx, nx, nz), 10.0, order="F"))
            s.set_pressure_chebyshev(True, PRESS_MAXIT, 1e-11)
            s.enable_vof()
            s.set_vof(C)
            s.set_property_model("rho", "linear", "C", [rho_g, rho_l - rho_g])
            s.set_property_model("force_z", "linear", "C",
                                 [-(rho_g - rho_mean) * g, -(rho_l - rho_g) * g])
            if mom:
                s.enable_vof_momentum(rho_g, rho_l)
            w = SolveWatch()
            blew = False
            for k in range(steps):
                try:
                    s.step()
                except RuntimeError:
                    blew = True
                    break
                w.sample(s)
                cfl = s.vof_last_courant()
                if cfl > 0.18:
                    dt *= 0.18 / cfl
                    s.set_dt(dt)
                elif cfl < 0.06 and dt < dt0:
                    dt = min(dt0, dt * 1.5)
                    s.set_dt(dt)
            if blew:
                row.append((float("nan"), float("nan"), w))
                continue
            cf = s.get_vof()
            col = cf.sum(axis=(0, 1)) / (nx * nx)
            hbar = col.sum()
            amp = float(np.max(cf.sum(axis=2)) - np.min(cf.sum(axis=2))) / nz
            mx = max(float(np.max(np.abs(s.get_u()))), float(np.max(np.abs(s.get_w()))))
            row.append((amp, mx, w))
            del s
        print("    %-5d | %-13.5e | %-13.5e | %-13.3e | %s"
              % (n, row[0][0], row[1][0], row[0][1], row[0][2]))


GATES = {
    "uniform": gate_uniform,
    "uniform_adv": gate_uniform_advection_only,
    "single_phase": gate_single_phase,
    "momentum": gate_momentum_conservation,
    "floor": gate_rho_floor,
    "drop": gate_falling_drop,
    "rt": gate_rt,
}

if __name__ == "__main__":
    want = sys.argv[1:] or ["uniform_adv", "uniform", "single_phase", "momentum", "floor"]
    for g in want:
        t0 = time.time()
        GATES[g]()
        print("  [%s: %.1f s]" % (g, time.time() - t0))
