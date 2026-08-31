#!/usr/bin/env python3
"""VoF rung V4 (WO-P) — the balanced-force CSF physics battery.

The *algebraic* gates of this rung (the exactness gate, the operator ablation, Young-Laplace,
the capillary dt, inertness, the wisp guard) live in `tests/kokkos/test_vof_surface_tension.cpp`
and run in the ctest. What is here is the PHYSICS: the cases where surface tension has to produce
a number somebody else has already published.

    static      spurious currents of a stationary droplet: Ca = mu max|u| / sigma vs resolution,
                and the wisp-guard ablation at a resolution where it is destructive.
    wave        a small-amplitude standing capillary wave against the analytical dispersion
                relation omega^2 = sigma k^3 / (rho1 + rho2) and the 2 nu k^2 viscous decay rate
                (Prosperetti 1981; Popinet 2009 section 4.2).
    lamb        a droplet oscillating in mode n = 2 against Lamb's (1932, art. 275) frequency
                omega^2 = n(n-1)(n+1)(n+2) sigma / (R^3 ((n+1) rho_in + n rho_out)).
    hysing      the Hysing et al. (IJNMF 60:1259, 2009) rising-bubble benchmark, both cases,
                against the published reference quantities. READ THE CAVEAT it prints.
    falling     the falling-drop terminal velocity WO-K had to defer, re-run WITH surface tension
                and with the momentum solve driven to a tolerance instead of a fixed sweep count.
    limits      step economics: does the capillary dt or the Weymouth-Yue CFL bind, and at what
                cost in steps?

UNITS. The solver's cell size is 1 and its time unit is the physical second, so a physical
problem of size L on nx cells is mapped with s = nx / L cells per unit length:

    rho' = rho    mu' = s^2 mu    sigma' = s^3 sigma    (rho g)' = s rho g    u' = s u    t' = t

(derived by substituting x = x'/s, u = u'/s, t = t' into the momentum equation and requiring
every term to scale alike; `Scale` below is that map, and every case states its physical numbers).

EVERY gate that reports a functional records the max pressure-iteration count against its cap and
the flux divergence, and marks a capped run INVALID rather than merely flagging it (shared
preamble rule 3b).

Usage:  PYTHONPATH=<build> python tests/study/vof_surface_tension.py [gate ...] [--quick]
"""
import math
import sys
import time

import numpy as np

import peclet.flow as pf

QUICK = "--quick" in sys.argv
GATES = [a for a in sys.argv[1:] if not a.startswith("--")]


# --------------------------------------------------------------------------- helpers
class Scale:
    """Physical <-> solver (cell size 1, time in seconds) unit map."""

    def __init__(self, cells_per_length):
        self.s = float(cells_per_length)

    def mu(self, mu):
        return self.s ** 2 * mu

    def sigma(self, sigma):
        return self.s ** 3 * sigma

    def bodyforce(self, rho_g):  # a force per unit volume, e.g. rho*g
        return self.s * rho_g

    def vel_to_phys(self, u):
        return u / self.s

    def len_to_cells(self, x):
        return self.s * x


def maxvel(s):
    return max(np.abs(s.get_u()).max(), np.abs(s.get_v()).max(), np.abs(s.get_w()).max())


class Solve:
    """A run's solver health, per the shared preamble's rule 3b."""

    def __init__(self, cap):
        self.cap = cap
        self.iters = 0
        self.div = 0.0

    def sample(self, s):
        self.iters = max(self.iters, s.last_pressure_iterations())
        self.div = max(self.div, s.max_open_divergence())

    @property
    def valid(self):
        return self.iters < self.cap

    def __str__(self):
        return (f"pressure {self.iters}/{self.cap} "
                f"{'OK' if self.valid else '*** CAPPED -> RUN INVALID ***'}, "
                f"max|div(open u)| {self.div:.2e}")


def sphere_fractions(shape, R, c, sub=24):
    """Volume fractions of a sphere: exact in z, sub x sub subsampled in (x,y)."""
    nx, ny, nz = shape
    ax = (np.arange(nx)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub).ravel()
    ay = (np.arange(ny)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub).ravel()
    X, Y = ax[:, None], ay[None, :]
    half = np.sqrt(np.maximum(R * R - (X - c[0]) ** 2 - (Y - c[1]) ** 2, 0.0))
    z0, z1 = c[2] - half, c[2] + half
    C = np.zeros((nx, ny, nz))
    for k in range(nz):
        seg = np.maximum(np.minimum(z1, k + 1) - np.maximum(z0, k), 0.0)
        C[:, :, k] = seg.reshape(nx, sub, ny, sub).mean(axis=(1, 3))
    return np.asfortranarray(C)


def cylinder_fractions(shape, R, cx, cz, sub=32):
    """Volume fractions of a cylinder with its axis along y (a 2-D disc), exact in z."""
    nx, ny, nz = shape
    ax = (np.arange(nx)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub).ravel()
    half = np.sqrt(np.maximum(R * R - (ax - cx) ** 2, 0.0))
    z0, z1 = cz - half, cz + half
    C = np.zeros((nx, ny, nz))
    for k in range(nz):
        seg = np.maximum(np.minimum(z1, k + 1) - np.maximum(z0, k), 0.0)
        col = seg.reshape(nx, sub).mean(axis=1)
        C[:, :, k] = col[:, None]
    return np.asfortranarray(C)


def zero_crossings(t, y):
    """Times of the sign changes of y (linear interpolation)."""
    out = []
    for i in range(len(y) - 1):
        if y[i] == 0.0:
            out.append(t[i])
        elif y[i] * y[i + 1] < 0.0:
            out.append(t[i] + (t[i + 1] - t[i]) * y[i] / (y[i] - y[i + 1]))
    return np.array(out)


# --------------------------------------------------------------------------- static droplet
def gate_static():
    print("\n" + "=" * 96)
    print("STATIC DROPLET — the spurious capillary number Ca = mu max|u| / sigma")
    print("=" * 96)
    print("  The balanced-force identity itself is EXACT (the ctest's P1: max|u| ~ 1e-17 with a")
    print("  constant kappa, at every resolution and viscosity). What is left here is the")
    print("  curvature error, and Ca is essentially its cell-unit magnitude.\n")
    rungs = [(16, 4.0), (32, 8.0), (48, 12.0)] if QUICK else [(16, 4.0), (32, 8.0), (48, 12.0),
                                                              (64, 16.0)]
    prev = None
    for n, R in rungs:
        s = pf.Solver(n, n, n)
        s.set_rho(1.0)
        s.set_mu(0.1)
        s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 500, 1e-14)
        s.enable_vof()
        s.set_vof(sphere_fractions((n, n, n), R, (n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11)))
        s.set_property_model("rho", "linear", "C", [1.0, 0.0])
        s.set_surface_tension(1.0)
        s.set_dt(0.5 * s.capillary_dt())
        h = Solve(500)
        for _ in range(60):
            s.step()
            h.sample(s)
        ca = 0.1 * maxvel(s) / 1.0
        br, ka = s.vof_curvature_branch(), s.vof_curvature()
        d = (br > 0.5) & (br < 5.5)
        dk = np.abs(ka[d] - 2.0 / R)
        d2 = s.csf_diagnostics()
        print(f"  D/dx = {2*R:5.1f}   Ca = {ca:.3e}"
              + (f"   ({prev/ca:.2f} x the coarser rung)" if prev else "               ")
              + f"   dkappa rms {dk.std():.2e} max {dk.max():.2e}"
              f"   orphan faces {d2['orphan_faces']}")
        print(f"                 {h}")
        prev = ca

    print("\n  WISP-GUARD ABLATION at 64^3 (set_vof_interface_eps(0) — the rung-V3 predicate):")
    for eps in (1e-8, 0.0):
        n, R = 64, 16.0
        s = pf.Solver(n, n, n)
        s.set_rho(1.0)
        s.set_mu(0.1)
        s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 500, 1e-14)
        s.enable_vof()
        s.set_vof(sphere_fractions((n, n, n), R, (n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11)))
        s.set_property_model("rho", "linear", "C", [1.0, 0.0])
        s.set_surface_tension(1.0)
        s.set_vof_interface_eps(eps)
        s.set_dt(0.5 * s.capillary_dt())
        u = []
        try:
            for _ in range(60):
                s.step()
                u.append(maxvel(s))
        except RuntimeError as e:
            print(f"    eps = {eps:g}: DIVERGED at step {len(u)} — {str(e)[:70]}")
            continue
        br, ka = s.vof_curvature_branch(), s.vof_curvature()
        d = (br > 0.5) & (br < 5.5)
        print(f"    eps = {eps:<6g} max|u| step 1 {u[0]:.3e} -> step 60 {u[-1]:.3e}"
              f"   cells served {d.sum()}   max|kappa| {np.abs(ka[d]).max():.3e} (2/R = {2/R:.4f})")


# --------------------------------------------------------------------------- capillary wave
def gate_wave():
    print("\n" + "=" * 96)
    print("CAPILLARY WAVE — dispersion omega^2 = sigma k^3 / (rho1 + rho2)  (Prosperetti 1981)")
    print("=" * 96)
    print("  Quasi-2D: a standing wave of wavelength lambda = Lx along x on a flat interface at")
    print("  mid-height, periodic in x and y, walls +-z. Small amplitude (a0 = lambda/100) and")
    print("  matched kinematic viscosities, so the inviscid dispersion relation is the leading")
    print("  behaviour and the decay rate is 2 nu k^2. Solver units throughout (cell = 1).\n")
    print(f"  {'nx':>4} {'lambda':>7} {'nu':>8} {'omega_num':>11} {'omega_th':>11} {'err':>8}"
          f"  {'gamma_num':>10} {'gamma_th':>10} {'err':>8}")
    rungs = [(32, 0.005)] if QUICK else [(32, 0.005), (64, 0.005), (32, 0.02)]
    for nx, nu in rungs:
        nz = nx
        ny = 4
        rho = 1.0
        sigma = 1.0
        lam = float(nx)
        k = 2 * math.pi / lam
        a0 = lam / 100.0
        omega_th = math.sqrt(sigma * k ** 3 / (2 * rho))
        gamma_th = 2 * nu * k * k

        s = pf.Solver(nx, ny, nz)
        s.set_rho(rho)
        s.set_mu(nu * rho)
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 1, 0, 0, 0)
        s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 500, 1e-11)
        s.enable_vof()
        # exact fractions of a cosine interface z = nz/2 + a0 cos(k x)
        xe = np.arange(nx + 1)
        zi = nz / 2.0 + a0 * (np.sin(k * xe[1:]) - np.sin(k * xe[:-1])) / (k * 1.0)
        C = np.zeros((nx, ny, nz))
        for kk in range(nz):
            C[:, :, kk] = np.clip(zi - kk, 0.0, 1.0)[:, None]
        s.set_vof(np.asfortranarray(C))
        s.set_property_model("rho", "linear", "C", [rho, 0.0])
        s.set_surface_tension(sigma)
        dt = 0.5 * s.capillary_dt()
        s.set_dt(dt)

        period = 2 * math.pi / omega_th
        nsteps = int(2.5 * period / dt)
        h = Solve(500)
        t, amp = [], []
        for i in range(nsteps):
            s.step()
            h.sample(s)
            Cf = s.get_vof()
            # interface height from the column sum (exact for a single-valued interface)
            col = Cf.sum(axis=2)[:, 0]
            zint = col - (nz / 2.0)
            t.append((i + 1) * dt)
            amp.append(float(np.dot(zint, np.cos(k * (np.arange(nx) + 0.5))) * 2.0 / nx))
        t = np.array(t)
        amp = np.array(amp)
        zc = zero_crossings(t, amp)
        if len(zc) >= 3:
            omega_num = math.pi / np.mean(np.diff(zc))
        else:
            omega_num = float("nan")
        # decay rate from the successive extrema
        pk = [i for i in range(1, len(amp) - 1)
              if abs(amp[i]) > abs(amp[i - 1]) and abs(amp[i]) > abs(amp[i + 1])]
        if len(pk) >= 2:
            gamma_num = -math.log(abs(amp[pk[-1]]) / abs(amp[pk[0]])) / (t[pk[-1]] - t[pk[0]])
        else:
            gamma_num = float("nan")
        print(f"  {nx:>4} {lam:>7.0f} {nu:>8.4g} {omega_num:>11.5f} {omega_th:>11.5f}"
              f" {100*(omega_num/omega_th-1):>7.2f}% {gamma_num:>10.3e} {gamma_th:>10.3e}"
              f" {100*(gamma_num/gamma_th-1):>7.1f}%")
        print(f"       a0 = {a0:.3f} cells, {nsteps} steps over 2.5 periods; {h}")


# --------------------------------------------------------------------------- Lamb oscillation
def gate_lamb():
    print("\n" + "=" * 96)
    print("OSCILLATING DROPLET — Lamb (1932) art. 275 mode n = 2")
    print("=" * 96)
    print("  omega_n^2 = n(n-1)(n+1)(n+2) sigma / (R^3 ((n+1) rho_in + n rho_out))")
    print("  A prolate perturbation of a sphere, amplitude 5% of R, released from rest; the")
    print("  frequency is read off the zero crossings of the n = 2 moment of the colour field.\n")
    print(f"  {'n':>4} {'R':>5} {'phi':>7} {'ratio':>7} {'omega_num':>11} {'omega_th':>11}"
          f" {'err':>8}")
    # Two sweeps: (a) fixed drop-to-box ratio phi = 6.5 % at two resolutions -- the DISCRETIZATION
    # rung; (b) the same drop R = 8 in a bigger and bigger periodic box -- the CONFINEMENT rung.
    # Lamb's frequency is for a drop in an UNBOUNDED outer fluid, and the periodic images add
    # outer-fluid inertia, which lowers omega. The ratio-100 row is the same statement from the
    # other side: with a 100x lighter outer fluid there is much less added mass to confine.
    rungs = [(32, 8.0, 1.0), (48, 8.0, 1.0)] if QUICK else [(32, 8.0, 1.0), (48, 12.0, 1.0),
                                                            (48, 8.0, 1.0), (64, 8.0, 1.0),
                                                            (48, 12.0, 100.0)]
    for n, R, ratio in rungs:
        sigma = 1.0
        rho_in, rho_out = ratio, 1.0
        mu = 0.02
        eps = 0.05
        nn = 2
        omega_th = math.sqrt(nn * (nn - 1) * (nn + 1) * (nn + 2) * sigma
                             / (R ** 3 * ((nn + 1) * rho_in + nn * rho_out)))
        # prolate spheroid of the same volume: a = R(1+eps), b = c = R/sqrt(1+eps)
        a = R * (1 + eps)
        b = R / math.sqrt(1 + eps)
        cx = cy = cz = n / 2 + 0.137
        sub = 16
        ax = (np.arange(n)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub).ravel()
        X, Y = ax[:, None], ax[None, :]
        q = 1.0 - ((X - cx) ** 2 + (Y - cy) ** 2) / b ** 2
        half = a * np.sqrt(np.maximum(q, 0.0))
        C = np.zeros((n, n, n))
        for kk in range(n):
            seg = np.maximum(np.minimum(cz + half, kk + 1) - np.maximum(cz - half, kk), 0.0)
            C[:, :, kk] = seg.reshape(n, sub, n, sub).mean(axis=(1, 3))

        s = pf.Solver(n, n, n)
        s.set_rho(rho_in)
        s.set_mu(mu)
        s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 500, 1e-11)
        s.enable_vof()
        s.set_vof(np.asfortranarray(C))
        s.set_property_model("rho", "linear", "C", [rho_out, rho_in - rho_out])
        s.set_surface_tension(sigma)
        dt = 0.5 * s.capillary_dt()
        s.set_dt(dt)
        period = 2 * math.pi / omega_th
        nsteps = int(2.5 * period / dt)
        xs = (np.arange(n) + 0.5) - cx
        ys = (np.arange(n) + 0.5) - cy
        zs = (np.arange(n) + 0.5) - cz
        XX = xs[:, None, None] ** 2
        YY = ys[None, :, None] ** 2
        ZZ = zs[None, None, :] ** 2
        h = Solve(500)
        t, m2 = [], []
        for i in range(nsteps):
            s.step()
            h.sample(s)
            Cf = s.get_vof()
            v = Cf.sum()
            # the n = 2 shape moment <2z^2 - x^2 - y^2>
            m = float((Cf * (2 * ZZ - XX - YY)).sum() / v)
            t.append((i + 1) * dt)
            m2.append(m)
        t = np.array(t)
        m2 = np.array(m2) - np.mean(np.array(m2)[len(m2) // 2:])
        zc = zero_crossings(t, m2)
        omega_num = math.pi / np.mean(np.diff(zc)) if len(zc) >= 3 else float("nan")
        phi = (4 / 3) * math.pi * R ** 3 / n ** 3
        print(f"  {n:>4} {R:>5.1f} {100*phi:>6.2f}% {ratio:>7.0f} {omega_num:>11.5f}"
              f" {omega_th:>11.5f} {100*(omega_num/omega_th-1):>7.2f}%")
        print(f"       {nsteps} steps over 2.5 periods, dt = {dt:.4g}; {h}")


# --------------------------------------------------------------------------- Hysing bubble
HYSING = {
    1: dict(rho1=1000.0, rho2=100.0, mu1=10.0, mu2=1.0, g=0.98, sigma=24.5,
            ref_vmax=0.2417, ref_tvmax=0.9215, ref_yc=1.081, ref_circ=0.9013),
    2: dict(rho1=1000.0, rho2=1.0, mu1=10.0, mu2=0.1, g=0.98, sigma=1.96,
            ref_vmax=0.2502, ref_tvmax=0.7317, ref_yc=1.1376, ref_circ=0.5869),
}


def gate_hysing(case=1, nx=64):
    p = HYSING[case]
    print("\n" + "=" * 96)
    print(f"HYSING RISING BUBBLE, case {case} (Hysing et al., IJNMF 60:1259, 2009), nx = {nx}")
    print("=" * 96)
    print("  LATERAL BC. The benchmark prescribes FREE-SLIP side walls, which this solver does not")
    print("  have (mac_bc.hpp offers periodic / no-slip / Dirichlet / outflow). The lateral")
    print("  condition here is PERIODIC — and for a bubble on the centreline that is not an")
    print("  approximation: mirroring about x = 0 and x = 1 puts images at -0.5 and 1.5, i.e. at")
    print("  spacing 1, and the mirror of a laterally symmetric bubble IS its translate. Case 1")
    print("  stays symmetric, so the two conditions are equivalent there; case 2 develops skirts")
    print("  and filaments that break the symmetry, so for case 2 it is an approximation.")
    print("  NOT reported: circularity. It needs the PLIC interface length, which the solver does")
    print("  not expose today; y_c(3) and max v_rise are the two the reference tabulates that can")
    print("  be computed from the colour field alone.\n")
    Lx, Lz = 1.0, 2.0
    nz = 2 * nx
    ny = 4
    sc = Scale(nx / Lx)
    R = sc.len_to_cells(0.25)
    s = pf.Solver(nx, ny, nz)
    s.set_rho(p["rho1"])
    s.set_mu(sc.mu(p["mu1"]))
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)  # no-slip top/bottom, as the benchmark prescribes
    s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 600, 1e-12)
    s.enable_vof()
    # C = 1 in the HEAVY fluid (the surrounding liquid), 0 in the bubble
    C = 1.0 - cylinder_fractions((nx, ny, nz), R, nx / 2.0, sc.len_to_cells(0.5))
    s.set_vof(np.asfortranarray(C))
    s.set_property_model("rho", "linear", "C", [p["rho2"], p["rho1"] - p["rho2"]])
    s.set_property_model("mu", "linear", "C", [sc.mu(p["mu2"]), sc.mu(p["mu1"] - p["mu2"])])
    s.set_surface_tension(sc.sigma(p["sigma"]))
    # Momentum-consistent transport (V2b): the same geometric fluxes carry rho^c u_c. At case 1's
    # density ratio 10 it moves the peak rise velocity by 15 % (see the findings entry) and at
    # case 2's ratio 1000 it is what the rung is FOR.
    s.enable_vof_momentum(p["rho2"], p["rho1"])
    # buoyancy f_z = -rho(C) g, as a LinearMix on C (the +-z walls carry the net force)
    s.set_property_model("force_z", "linear", "C",
                         [-sc.bodyforce(p["rho2"] * p["g"]),
                          -sc.bodyforce((p["rho1"] - p["rho2"]) * p["g"])])
    T = 3.0
    dt = 0.5 * s.capillary_dt()
    s.set_dt(dt)
    lim = s.vof_step_limits()
    print(f"  h = 1/{nx}, dt_sigma = {s.capillary_dt():.4e} s")
    print(f"  step limits at t=0: WY CFL dt {lim['cfl_dt']:.4g}, capillary dt "
          f"{lim['capillary_dt']:.4g}  ->  {'CAPILLARY' if lim['capillary_binds'] else 'CFL'} binds")
    print(f"  ADAPTIVE dt = 0.4 x min(WY CFL, capillary), re-picked every 10 steps: the bubble")
    print(f"  accelerates, so the WY cap takes over from the capillary limit part-way through.")
    zs = (np.arange(nz) + 0.5) / sc.s
    h = Solve(600)
    vmax, tvmax, yc = 0.0, 0.0, 0.0
    t0 = time.time()
    t = 0.0
    i = 0
    ncap, ncfl = 0, 0
    while t < T:
        if i % 10 == 0:
            L = s.vof_step_limits()
            dt = 0.4 * min(L["cfl_dt"], L["capillary_dt"])
            s.set_dt(dt)
            ncap += int(L["capillary_binds"])
            ncfl += int(not L["capillary_binds"])
        t += dt
        i += 1
        nsteps = i
        s.step()
        h.sample(s)
        Cf = s.get_vof()
        B = 1.0 - Cf  # the bubble indicator
        vb = B.sum()
        ycm = float((B.sum(axis=(0, 1)) * zs).sum() / vb)
        w = s.get_w()
        vrise = float((B * w).sum() / vb) / sc.s
        if vrise > vmax:
            vmax, tvmax = vrise, t
        yc = ycm
        if i % 100 == 0:
            print(f"    t = {t:5.3f}  dt = {dt:.3e}  y_c = {ycm:.4f}  v_rise = {vrise:.4f}  "
                  f"V/V0 = {vb/ (math.pi*0.25**2*sc.s**2*ny):.6f}")
    print(f"  measured: v_rise max {vmax:.4f} at t = {tvmax:.3f};  y_c(3) = {yc:.4f}")
    print(f"  reference: v_rise max {p['ref_vmax']:.4f} at t = {p['ref_tvmax']:.3f};  "
          f"y_c(3) = {p['ref_yc']:.4f}")
    print(f"  deviation: v_rise {100*(vmax/p['ref_vmax']-1):+.1f} %, "
          f"y_c {100*(yc/p['ref_yc']-1):+.1f} %")
    print(f"  {nsteps} steps to t = {T}; the binding limit was CAPILLARY on {ncap} of the "
          f"{ncap+ncfl} dt re-picks and the WY CFL on {ncfl}")
    print(f"  {h}   ({time.time()-t0:.0f} s)")


# --------------------------------------------------------------------------- falling drop
def gate_falling():
    print("\n" + "=" * 96)
    print("FALLING DROP — the terminal velocity WO-K deferred, re-run with surface tension")
    print("=" * 96)
    print("  WO-K reached 9 % of Hadamard-Rybczynski and named an under-resolved momentum solve")
    print("  as the suspected cause. BOTH halves of that turn out to be wrong, and the gate as")
    print("  written measured the wrong quantity:")
    print("")
    print("  (1) A periodic box driven by a ZERO-MEAN body force conserves total MOMENTUM, not")
    print("      total volume flux. The box's mean velocity is a free mode of the projection, and")
    print("      with rho_l/rho_g = 800 at phi = 1.6 % the light ambient recoils at ~13 x the")
    print("      drop's speed (rho_g (1-phi) U_a = -rho_l phi U_d). So the LAB-FRAME drop velocity")
    print("      is a near-cancellation of two much larger numbers and is not the settling")
    print("      velocity. The quantity Hadamard-Rybczynski predicts is U_drop - U_ambient.")
    print("  (2) Driving the momentum smoother to a tolerance instead of a fixed sweep count does")
    print("      NOT rescue the lab-frame number -- it moves it further, precisely because that")
    print("      number is a cancellation. Both are reported below.")
    print("")
    print("  Also corrected here: Hadamard-Rybczynski is U = (2/3)(d rho) g R^2/mu_o *")
    print("  (mu_o + mu_i)/(2 mu_o + 3 mu_i) -- the denominator carries 3 mu_i, and the rigid")
    print("  limit mu_i -> inf must return Stokes' 2 (d rho) g R^2/(9 mu_o).\n")
    rho_l, rho_g = 800.0, 1.0
    mu_out = 50.0
    mu_ratio = 100.0
    sigma = 16.0
    g = 3.4e-4

    def hr(R, mo, lam):
        return (2.0 / 3.0) * (rho_l - rho_g) * g * R * R / mo * (1 + lam) / (2 + 3 * lam)

    def run(n, D, cap, rtol, label):
        R = D / 2
        phi = (4 / 3) * math.pi * R ** 3 / n ** 3
        s = pf.Solver(n, n, n)
        s.set_rho(rho_l)
        s.set_mu(mu_out)
        s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 400, 1e-12)
        s.enable_vof()
        s.set_vof(sphere_fractions((n, n, n), R, (n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11)))
        s.set_property_model("rho", "linear", "C", [rho_g, rho_l - rho_g])
        s.set_property_model("mu", "linear", "C", [mu_out, mu_ratio * mu_out - mu_out])
        s.set_surface_tension(sigma)
        s.enable_vof_momentum(rho_g, rho_l)
        s.set_property_model("force_z", "linear", "C",
                             [g * (rho_l - rho_g) * phi, -g * (rho_l - rho_g)])
        if rtol > 0:
            s.set_velocity_solver_params(cap, rtol, 2)
        else:
            s.set_velocity_solver_params(cap)
        dt = 0.5 * s.capillary_dt()
        s.set_dt(dt)
        tau = 2.0 * R * R * rho_l / (9.0 * mu_out)
        nsteps = int((2.0 if QUICK else 6.0) * tau / dt)
        h = Solve(400)
        sweeps = 0
        hist = []
        for i in range(nsteps):
            s.step()
            h.sample(s)
            sweeps = max(sweeps, s.last_step_timers()["momentum_sweeps"])
            C = s.get_vof()
            w = s.get_w()
            ud = float((C * w).sum() / C.sum())
            ua = float(((1 - C) * w).sum() / (1 - C).sum())
            hist.append((ud, ua))
        tail = hist[-max(10, nsteps // 20):]
        Ud = float(np.mean([a for a, _ in tail]))
        Ua = float(np.mean([b for _, b in tail]))
        Urel = Ud - Ua
        U0 = hr(R, mu_out, mu_ratio)
        # Hasimoto (1959) simple-cubic array drag correction, in its DENOMINATOR form
        #   F = 6 pi mu a U / (1 - 1.7601 c^{1/3} + c - 1.5593 c^2)
        # -- not the linearised 1 + 1.7601 c^{1/3}, which under-corrects by 20 % already at
        # c = 0.016 (1.443 against 1.746) and is what WO-K's substitute quoted.
        K = 1.0 / (1.0 - 1.7601 * phi ** (1 / 3) + phi - 1.5593 * phi ** 2)
        print(f"  {label}")
        print(f"     D/h {D:.0f}  phi {phi:.4f}  U_HR {U0:.4e}  K {K:.3f}  U_HR/K {U0/K:.4e}")
        print(f"     U_drop(lab) {Ud:.4e} -> {Ud/(-U0/K):.4f} x     "
              f"U_ambient {Ua:.4e}   recoil ratio {-Ua/Ud:.1f}")
        print(f"     U_rel = U_drop - U_ambient = {Urel:.4e}  ->  "
              f"**{Urel/(-U0/K):.4f} x (U_HR/K)**")
        print(f"     dt {dt:.3g} = 0.5 dt_sigma, {nsteps} steps = "
              f"{nsteps*dt/tau:.1f} tau, max sweeps/step {sweeps};  {h}")
        return Urel / (-U0 / K)

    print("  (a) the momentum-solve hypothesis, at D/h = 15:")
    run(48, 15.0, 20, 0.0, "fixed 20 sweeps (WO-K's setting)")
    run(48, 15.0, 400, 1e-4, "tolerance 1e-4, cap 400")
    if not QUICK:
        run(48, 15.0, 2000, 1e-6, "tolerance 1e-6, cap 2000")
    print("\n  (b) resolution, with the momentum solve converged (tolerance 1e-4):")
    for n, D in ((48, 10.0), (48, 15.0), (64, 20.0)) if not QUICK else ((48, 10.0), (48, 15.0),):
        run(n, D, 400, 1e-4, f"n = {n}, D/h = {D:.0f}")


# --------------------------------------------------------------------------- step economics
def gate_limits():
    print("\n" + "=" * 96)
    print("STEP ECONOMICS — does the capillary dt or the Weymouth-Yue CFL bind?")
    print("=" * 96)
    print("  A pore-scale configuration in physical units: water/air (sigma = 0.072 N/m,")
    print("  rho 1000/1.2, mu 1e-3/1.8e-5), a pore of diameter d driven at a capillary number")
    print("  Ca = mu U / sigma. The two explicit limits are")
    print("      capillary   dt < sqrt((rho1+rho2) h^3 / (4 pi sigma))")
    print("      transport   dt < CFL_WY h / |u|, CFL_WY = 0.25 (Weymouth's 3-D bound)")
    print("  so their ratio is  dt_cfl/dt_sigma = (CFL/|u'|) sqrt(4 pi sigma' / (rho_sum))")
    print("  dt_sigma shrinks as h^{3/2} and dt_cfl only as h, so REFINING makes the capillary")
    print("  limit MORE binding, not less: their ratio grows as h^{-1/2}.\n")
    rho1, rho2 = 1000.0, 1.2
    sigma_p = 0.072
    mu1 = 1e-3
    print(f"  {'d [um]':>8} {'cells/d':>8} {'Ca':>8} {'U [m/s]':>10} {'dt_sig [s]':>12}"
          f" {'dt_cfl [s]':>12} {'binds':>10} {'steps for 1 pore':>18}")
    for d_um in (50.0, 200.0):
        for cells in (16, 32, 64):
            for Ca in (1e-6, 1e-4, 1e-2):
                d = d_um * 1e-6
                sc = Scale(cells / d)
                U = Ca * sigma_p / mu1
                sig_s = sc.sigma(sigma_p)
                dt_sig = math.sqrt((rho1 + rho2) * 1.0 / (4 * math.pi * sig_s))
                dt_cfl = 0.25 / (sc.s * U)
                binds = "capillary" if dt_sig < dt_cfl else "CFL"
                t_pore = d / U
                nst = t_pore / min(dt_sig, dt_cfl)
                print(f"  {d_um:>8.0f} {cells:>8d} {Ca:>8.0e} {U:>10.3e} {dt_sig:>12.3e}"
                      f" {dt_cfl:>12.3e} {binds:>10} {nst:>18.3e}")
    print("\n  And the same read-off from the solver itself on the static-droplet case:")
    n, R = 48, 12.0
    s = pf.Solver(n, n, n)
    s.set_rho(1000.0)
    s.set_mu(1.0)
    s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 400, 1e-12)
    s.enable_vof()
    s.set_vof(sphere_fractions((n, n, n), R, (n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11)))
    s.set_property_model("rho", "linear", "C", [1.0, 999.0])
    s.set_surface_tension(50.0)
    s.set_dt(0.5 * s.capillary_dt())
    U = 0.05
    s.set_state(np.full((n, n, n), U, order="F"), np.zeros((n, n, n), order="F"),
                np.zeros((n, n, n), order="F"))
    lim = s.vof_step_limits()
    print(f"    uniform |u| = {U}: WY CFL dt {lim['cfl_dt']:.4g}, capillary dt "
          f"{lim['capillary_dt']:.4g}, binding {lim['binding']:.4g} "
          f"({'capillary' if lim['capillary_binds'] else 'CFL'})")


# --------------------------------------------------------------------------- main
ALL = {
    "static": gate_static,
    "wave": gate_wave,
    "lamb": gate_lamb,
    "hysing1": lambda: gate_hysing(1, 32 if QUICK else 64),
    "hysing2": lambda: gate_hysing(2, 32 if QUICK else 64),
    "falling": gate_falling,
    "limits": gate_limits,
}

if __name__ == "__main__":
    run = GATES or list(ALL)
    for g in run:
        if g not in ALL:
            print(f"unknown gate {g!r}; known: {', '.join(ALL)}")
            sys.exit(2)
    for g in run:
        ALL[g]()
    print("\ndone.")
