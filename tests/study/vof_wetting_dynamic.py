#!/usr/bin/env python3
"""WO-V6 (rung V6) — the DYNAMIC contact angle and hysteresis: the gate battery, in Python.

Every gate prints its numbers; nothing is summarised into an adjective. The MODEL (state it with
every result, VOF_PLAN section 6):

    theta_Delta^3 = theta_e^3 + 9 Ca_cl ln(Delta/lambda),   Ca_cl = mu_l U_cl / sigma

Afkhami, Zaleski & Bussmann, JCP 228:5370 (2009). `lambda` is an EXPLICIT slip length in cells and
is never omitted from a reported number: a VoF contact line's numerical slip is proportional to the
cell size, so without it the imposed angle is silently grid-dependent.

  jurin   G3-static. CAPILLARY RISE, STATIC. Two 8-cell plates with ROUNDED ends (a capsule SDF, so
          there is no sharp 90-degree corner where |grad sdf| != 1) at QUARTER-integer placement,
          bounding a gap `w`; the periodic outer channel is a second capillary of width `w_out`.
          Both menisci rise, so the reference is the LEVEL DIFFERENCE. Jurin's
          `2 sigma cos(theta)/(drho g) (1/w - 1/w_out)` is the Bond -> 0 asymptote and is NOT the
          right reference at the Bond numbers a finite box affords; the exact reference is the
          static meniscus ODE `sigma z''/(1+z'^2)^{3/2} = drho g (z - z_ref)` with `z'(w/2) =
          cot(theta)`, solved here by shooting. Both are printed.

  spread  G2. SPREADING DROP vs Cox-Voinov. A drop released as a hemisphere on a wall with
          `theta_e = 30`: the contact radius a(t) and the apparent angle theta_app(t) are measured
          from the shape, `Ca_cl = mu (da/dt)/sigma` from the contact line's own motion, and
          `theta_app^3 - theta_e^3` is fitted against `Ca_cl`. The composition property of
          Cox-Voinov says the slope is `9 ln(a/lambda)`: the fill supplies `9 ln(Delta/lambda)` and
          the resolved hydrodynamics supply `9 ln(a/Delta)`. Tanner's `a ~ t^(1/10)` at late times
          is reported alongside.

  rise    G3-dynamics. The same plate scene from a FLAT interface, with the dynamic angle, over a
          slip sweep: the rise curve h(t), the final height against the static equilibrium, and the
          damping class (asymptotic vs oscillatory) — the two regimes Gruending et al. (AMM 86:142,
          2020) separate.

  incline G4. HYSTERESIS: a drop on a tilted wall (the wall is kept axis-aligned and GRAVITY is
          tilted, which is the same problem and keeps the SDF exact). Below the critical Bond
          number `Bo_c = cos(theta_r) - cos(theta_a)` the contact line must stay PINNED; above it
          the drop slides with the front near theta_a and the rear near theta_r.

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_wetting_dynamic.py [jurin spread rise incline]
    ... --quick        shorter runs
"""
import math
import sys

import numpy as np

import peclet.flow as pf

PRESS_MAXIT = 300
DEG = math.pi / 180.0


# ------------------------------------------------------------------- the exact static meniscus
_MEN_CACHE = {}


def meniscus(w, theta_deg, dg, sigma=1.0, n=600):
    """The exact 2-D static meniscus in a slot of width `w` with contact angle `theta` on both
    walls, in gravity `dg = drho*g`:

        sigma z''/(1 + z'^2)^{3/2} = dg (z - z_ref),    z'(0) = 0,   z'(w/2) = cot(theta)

    measured from the flat far-field datum `z_ref = 0`. Shooting on z(0) (monotone: a higher
    meniscus has a steeper wall slope). Returns (x, z) on [0, w/2] — the profile is symmetric.
    Reduces to Jurin `z = 2 sigma cos(theta)/(dg w)` as Bond -> 0.
    """
    key = (round(w, 9), round(theta_deg, 9), round(dg, 12), round(sigma, 12), n)
    if key in _MEN_CACHE:
        return _MEN_CACHE[key]
    target = 1.0 / math.tan(theta_deg * DEG)
    h = 0.5 * w / n
    k = dg / sigma

    def shoot(z0):
        z, zp = z0, 0.0
        zs = np.empty(n + 1)
        zs[0] = z0
        for i in range(n):
            def f(a, b):
                return b, k * a * (1.0 + min(b * b, 1e12)) ** 1.5
            a1, b1 = f(z, zp)
            a2, b2 = f(z + 0.5 * h * a1, zp + 0.5 * h * b1)
            a3, b3 = f(z + 0.5 * h * a2, zp + 0.5 * h * b2)
            a4, b4 = f(z + h * a3, zp + h * b3)
            z += h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
            zp += h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
            zs[i + 1] = z
            if not (abs(z) < 1e8 and abs(zp) < 1e8):
                zs[i + 1:] = z
                return 1e9, zs
        return zp, zs

    lo, hi = 0.0, max(4.0 * abs(jurin(w, theta_deg, dg, sigma)), 1.0)
    while shoot(hi)[0] < target:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if shoot(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    _, zs = shoot(0.5 * (lo + hi))
    out = (np.arange(n + 1) * h, zs)
    _MEN_CACHE[key] = out
    return out


def meniscus_mean(w, theta_deg, dg, sigma=1.0, xlo=None, xhi=None):
    """The MEAN interface height over x in [xlo, xhi] (default the whole slot, x measured from the
    LEFT wall) — the quantity the colour-integral level read-out measures."""
    xs, zs = meniscus(w, theta_deg, dg, sigma)
    x = np.concatenate([-xs[::-1], xs[1:]]) + 0.5 * w   # 0 .. w
    z = np.concatenate([zs[::-1], zs[1:]])
    if xlo is None:
        xlo, xhi = 0.0, w
    m = (x >= xlo) & (x <= xhi)
    if m.sum() < 2:
        return float(np.interp(0.5 * (xlo + xhi), x, z))
    return float(np.trapezoid(z[m], x[m]) / (x[m][-1] - x[m][0]))


def jurin(w, theta_deg, dg, sigma=1.0):
    return 2.0 * sigma * math.cos(theta_deg * DEG) / (dg * w)


# --------------------------------------------------------------------------------------- helpers
def max_u(s, zlo=None):
    m = 0.0
    for f in (s.get_u, s.get_v, s.get_w):
        v = np.abs(f())
        m = max(m, float(v.max()) if zlo is None else float(v[:, :, zlo:].max()))
    return m


def relax(s, steps, cfl=0.15, dt_cap=None, probe=0, cb=None, ramp=0.02):
    """`ramp`: the first step is taken at `ramp * dt_cap`, because the interface-local Courant
    number is read BEFORE the step and is 0 at rest — a full first step on a scene with a strong
    initial capillary imbalance overshoots the Weymouth-Yue cap and the advector (correctly) throws."""
    dt_cap = dt_cap if dt_cap is not None else 0.5 * s.capillary_dt()
    dt = ramp * dt_cap
    s.set_dt(dt)
    maxit, capped, t = 0, 0, 0.0
    trace = []
    for i in range(steps):
        c = s.vof_max_courant()
        if c > cfl:
            dt = min(dt_cap, dt * cfl / c)
            s.set_dt(dt)
        elif dt < dt_cap and c < 0.5 * cfl:
            dt = min(dt_cap, 1.1 * dt)
            s.set_dt(dt)
        s.step()
        cl = s.vof_last_courant()
        if cl > 0.2:  # the sweep that just ran came close to the WY cap: back off before the next
            dt = max(dt * 0.5, 1e-6 * dt_cap)
            s.set_dt(dt)
        t += dt
        it = s.last_pressure_iterations()
        maxit = max(maxit, it)
        capped += 1 if it >= PRESS_MAXIT else 0
        if probe and (i + 1) % probe == 0:
            trace.append((i + 1, t, cb(s) if cb else max_u(s)))
    return dt, maxit, capped, trace, t


def capsule_sdf(nx, ny, nz, xm, r, z0, z1):
    """A plate of thickness 2r centred on x = xm, spanning z in [z0, z1] with SEMICIRCULAR ends:
    the distance to the segment (xm, z0+r) -> (xm, z1-r) minus r. Flat vertical faces at xm +/- r,
    no sharp corner anywhere (|grad sdf| = 1 everywhere outside the axis)."""
    ax = (np.arange(nx) + 0.5)[:, None]
    az = (np.arange(nz) + 0.5)[None, :]
    qz = np.maximum((z0 + r) - az, az - (z1 - r))
    d = np.hypot(np.abs(ax - xm), np.maximum(qz, 0.0)) - r
    return np.broadcast_to(d[:, None, :], (nx, ny, nz))


# ------------------------------------------------------------------------------- G3 static Jurin
def jurin_scene(nx=96, ny=4, nz=112, w=16.0, plate=8.0, zp=(12.0, 100.0)):
    """The two rounded plates. Returns (sdf, xm_gap, xm_outer, w_out, subsdf); the plate FACES sit
    at quarter-integer x (WO-S finding 5: at a half-integer the wall cell's tangential MAC faces sit
    on the SDF zero level and close, and the contact line cannot move). `subsdf(x, z)` is the same
    SDF as a callable, for sub-cell sampling of the initial colour."""
    r = 0.5 * plate
    xm1 = 32.25 + r
    xm2 = xm1 + plate + w
    w_out = nx - (xm2 + r - (xm1 - r))

    def subsdf(px, pz):
        qz = np.maximum((zp[0] + r) - pz, pz - (zp[1] - r))
        q = np.maximum(qz, 0.0)
        return np.minimum(np.hypot(np.abs(px - xm1), q), np.hypot(np.abs(px - xm2), q)) - r

    d = np.minimum(capsule_sdf(nx, ny, nz, xm1, r, zp[0], zp[1]),
                   capsule_sdf(nx, ny, nz, xm2, r, zp[0], zp[1]))
    return np.asfortranarray(d.astype(float)), 0.5 * (xm1 + xm2), (xm2 + r + 0.5 * w_out) % nx, \
        w_out, subsdf


def jurin_colour(nx, ny, nz, subsdf, zi_of_x, sub=4):
    """C = liquid fraction of the FLUID volume of each cell (VOF_PLAN rule 2), for an interface
    z = zi(x). The FLUID restriction is not cosmetic: a sub-cell of a cut wall column that is
    sampled as if it were fluid drops that column's C to `eps`, and a C of 0.75 in the middle of
    the liquid is a FAKE INTERFACE that the V3 cascade turns into |kappa| ~ 3e6 (measured)."""
    off = (np.arange(sub) + 0.5) / sub
    px = (np.arange(nx)[:, None, None, None] + off[None, :, None, None])   # (nx, sub, 1, 1)
    pz = (np.arange(nz)[None, None, :, None] + off[None, None, None, :])   # (1, 1, nz, sub)
    fluid = subsdf(px, pz) > 0.0                                            # (nx, sub, nz, sub)
    liq = fluid & (pz < zi_of_x(px))
    nf = fluid.sum(axis=(1, 3))
    c2 = np.where(nf > 0, liq.sum(axis=(1, 3)) / np.maximum(nf, 1), 0.0)
    return np.asfortranarray(np.broadcast_to(c2[:, None, :], (nx, ny, nz)).astype(float).copy())


def set_zero_mean_buoyancy(s, dg, ratio, wos=False):
    """`force_z = g (rho_bar - rho)` with `g = dg/(rho_l - rho_g)`: the buoyancy that produces the
    prescribed `drho g` AND has zero volume mean.

    THIS IS THE FIX THAT MAKES THE JURIN SCENE WORK, and the defect it removes is what WO-S's G4
    measured. The obvious "zero in the gas, -dg in the liquid" form (`[dg/(ratio-1), -dg/(1-1/ratio)]`,
    which is what WO-S used) has a NON-ZERO volume mean, and in a fully periodic box a non-zero-mean
    body force accelerates the whole fluid without bound (WO-Q finding 9). Worse than a drift: the
    accelerating frame contributes its own `-rho a` body force, i.e. it CANCELS PART OF GRAVITY, so
    the menisci relax towards a smaller level difference the longer the run goes — exactly the -83 %
    WO-S recorded, with the accompanying non-decaying `max|u|`.
    """
    if wos:  # the ABLATION: WO-S's G4 force model, zero in the gas and -dg in the liquid
        s.set_property_model("force_z", "linear", "rho",
                             [dg / (ratio - 1.0), -dg / (1.0 - 1.0 / ratio)])
        return None
    rho_g, rho_l = 1.0 / ratio, 1.0
    g = dg / (rho_l - rho_g)
    eps = s.vof_geometry(0)
    cc = s.get_vof()
    cbar = float((cc * eps).sum()) / float(eps.sum())
    rho_bar = rho_g + (rho_l - rho_g) * cbar
    s.set_property_model("force_z", "linear", "rho", [g * rho_bar, -g])
    return rho_bar


def _channel_columns(sdf, nz):
    """The fluid columns of the gap and of the periodic outer channel, as index arrays. EVERY fluid
    column is included, cut ones too: the level read-out is eps-weighted, so the full-width mean is
    what the exact identity below refers to."""
    nx = sdf.shape[0]
    solid = (sdf[:, 0, nz // 2] < 0.0)
    px = np.where(solid)[0]
    # the two plates, in x; the gap is between them
    gaps = np.split(px, np.where(np.diff(px) > 1)[0] + 1)
    left, right = gaps[0], gaps[-1]
    xin = np.arange(left.max(), right.min() + 1)
    xout = np.concatenate([np.arange(right.max(), nx), np.arange(0, left.min() + 1)])
    return xin, xout


def level_of(cc, eps, xs):
    col = (cc[xs, :, :] * eps[xs, :, :]).sum(axis=(0, 1))
    tot = eps[xs, :, :].sum(axis=(0, 1))
    return float(np.where(tot > 0, col / np.maximum(tot, 1e-30), 0.0).sum())


def g3_static(theta=30.0, steps=500, nx=96, ny=4, nz=112, w=16.0, plate=8.0, dg=3.0e-3, sigma=1.0,
              mu=0.2, ratio=10.0, datum=28.0, zp=(12.0, 100.0), flat_start=False, wos=False):
    sdf, xm_in, xm_out, w_out, subsdf = jurin_scene(nx, ny, nz, w, plate, zp=zp)
    jin, jout = jurin(w, theta, dg, sigma), jurin(w_out, theta, dg, sigma)
    zin, zout = meniscus_mean(w, theta, dg, sigma), meniscus_mean(w_out, theta, dg, sigma)
    bo_in, bo_out = dg * w * w / sigma, dg * w_out * w_out / sigma
    ref = jin - jout
    print(f"\n=== G3-static  JURIN on the FIXED scene — grid {nx}x{ny}x{nz}, plates {plate:.0f} "
          f"cells thick with ROUNDED ends (capsule SDF), QUARTER-integer faces, gap w = {w:.0f}, "
          f"periodic outer channel w_out = {w_out:.0f}, theta = {theta}, sigma = {sigma}, "
          f"drho*g = {dg:g}, ratio {ratio:g}, {steps} steps, "
          f"{'FLAT start (attraction)' if flat_start else 'equilibrium start (fixed point)'}"
          f"{', WO-S NON-zero-mean buoyancy ABLATION' if wos else ''}")
    print(f"    Bond inner {bo_in:.3f}  outer {bo_out:.3f}")
    print(f"    Jurin: inner {jin:.4f}  outer {jout:.4f}  DIFFERENCE {ref:.4f} cells")
    print(f"    the exact meniscus ODE's MEAN level: inner {zin:.4f}  outer {zout:.4f}  "
          f"DIFFERENCE {zin - zout:.4f} — identical to Jurin BY AN EXACT IDENTITY at ANY Bond "
          f"(integrate sigma d/dx[z'/sqrt(1+z'^2)] = drho g z over the slot: the flux term gives "
          f"2 sigma cos(theta) and the source gives drho g w zbar), so the eps-weighted "
          f"level integral is the RIGHT read-out and needs no low-Bond assumption")

    xs_in, zs_in = meniscus(w, theta, dg, sigma)
    xs_o, zs_o = meniscus(w_out, theta, dg, sigma)

    def zi_of_x(x):
        if flat_start:
            return np.full_like(x, datum)
        dxi = np.abs(x - xm_in)
        dxo = np.abs(((x - xm_out + 0.5 * nx) % nx) - 0.5 * nx)
        zi = np.interp(np.clip(dxi, 0, 0.5 * w), xs_in, zs_in) + datum
        zo = np.interp(np.clip(dxo, 0, 0.5 * w_out), xs_o, zs_o) + datum
        return np.where(dxi <= 0.5 * w, zi, zo)

    c0 = jurin_colour(nx, ny, nz, subsdf, zi_of_x)
    s = pf.Solver(nx, ny, nz)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_solid(sdf, cutcell_pressure=True)
    s.enable_vof()
    s.set_property_model("rho", "linear", "C", [1.0 / ratio, 1.0 - 1.0 / ratio])
    s.set_property_model("mu", "linear", "C", [mu / ratio, mu - mu / ratio])
    s.set_vof(c0)
    set_zero_mean_buoyancy(s, dg, ratio, wos)
    s.set_surface_tension(sigma)
    s.set_contact_angle(theta)
    v0 = s.vof_diagnostics()["volume"]
    eps = s.vof_geometry(0)
    xin, xout = _channel_columns(sdf, nz)
    l0 = level_of(s.get_vof(), eps, xin) - level_of(s.get_vof(), eps, xout)
    print(f"    initial level DIFFERENCE {l0:.4f} (the initial condition IS the exact equilibrium "
          f"for this datum when the run starts from it)")
    dt, maxit, capped, trace, tend = relax(
        s, steps, probe=max(1, steps // 10),
        cb=lambda q: level_of(q.get_vof(), eps, xin) - level_of(q.get_vof(), eps, xout))
    cc = s.get_vof()
    lin, lout = level_of(cc, eps, xin), level_of(cc, eps, xout)
    d = s.vof_diagnostics()
    err = ((lin - lout) - ref) / ref
    print(f"  measured inner level {lin:.4f}  outer {lout:.4f}  DIFFERENCE {lin-lout:.4f} cells "
          f"vs Jurin {ref:.4f}  -> {100*err:+.2f} % (gate 5 %)  "
          f"{'PASS' if abs(err) <= 0.05 else 'FAIL'}")
    print(f"  dV/V {abs(d['volume']-v0)/v0:.3e}  max|u| {max_u(s):.3e}  iters {maxit} "
          f"(capped {capped})  dt {dt:.4f}  solid sum {d['solid_sum']:.1e}  t_end {tend:.1f}")
    print("  level-difference trace: " + " ".join(f"{t:.1f}:{v:.3f}" for _, t, v in trace))
    return lin - lout, ref


# ------------------------------------------------------------------------------------- G2 spread
def cap_sphere(volume, theta_deg):
    ct = math.cos(theta_deg * DEG)
    Rc = (3.0 * volume / (math.pi * (1.0 - ct) ** 2 * (2.0 + ct))) ** (1.0 / 3.0)
    return Rc, -Rc * ct


def cap_colour(nx, ny, nz, cx, cy, zw, R, theta_deg=90.0):
    Rc, zc = cap_sphere(2.0 / 3.0 * math.pi * R ** 3, theta_deg)
    sub = (np.arange(4) + 0.5) / 4.0
    px = np.arange(nx)[:, None, None, None, None, None] + sub[None, None, None, :, None, None]
    py = np.arange(ny)[None, :, None, None, None, None] + sub[None, None, None, None, :, None]
    pz = np.arange(nz)[None, None, :, None, None, None] + sub[None, None, None, None, None, :]
    inside = ((px - cx) ** 2 + (py - cy) ** 2 + (pz - zw - zc) ** 2) < Rc * Rc
    fluid = np.broadcast_to(pz >= zw, inside.shape)
    tot = fluid.sum(axis=(3, 4, 5))
    ins = (fluid & inside).sum(axis=(3, 4, 5))
    return np.asfortranarray(np.where(tot > 0, ins / np.maximum(tot, 1), 0.0).astype(float))


def g2_spread(theta_e=30.0, slip=0.1, steps=1500, probe=50, nx=64, nz=40, R=12.0, zw=4.25,
              sigma=1.0, oh=0.1, dynamic=True, wall_slip=False):
    mu = oh * math.sqrt(1.0 * sigma * 2.0 * R)
    print(f"\n=== G2  SPREADING DROP vs Cox-Voinov — D/dx {2*R:.0f}, grid {nx}x{nx}x{nz}, wall "
          f"z = {zw}, theta_e = {theta_e}, lambda = {slip} cells, sigma {sigma}, mu {mu:.4f} "
          f"(Oh {oh}), {steps} steps, dynamic angle {'ON' if dynamic else 'OFF (static control)'}"
          f", momentum Navier slip {'ON (WO-V6b)' if wall_slip else 'OFF'}")
    z = (np.arange(nz) + 0.5)[None, None, :]
    sdf = np.asfortranarray(
        np.broadcast_to(np.minimum(z - zw, (nz - zw) - z), (nx, nx, nz)).astype(float))
    c0 = cap_colour(nx, nx, nz, nx * 0.5, nx * 0.5, zw, R, 90.0)   # released as a HEMISPHERE
    s = pf.Solver(nx, nx, nz)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_solid(sdf, cutcell_pressure=True)
    s.enable_vof()
    s.set_vof(c0)
    s.set_surface_tension(sigma)
    if dynamic:
        s.set_contact_angle_dynamic(theta_e, slip, mu, sigma)
    else:
        s.set_contact_angle(theta_e)
    if wall_slip:                    # WO-V6b: the SAME lambda in the momentum wall closure
        s.set_wall_slip_length(slip)
    eps = s.vof_geometry(0)
    ix = nx // 2
    hist = []

    def shape(sv):
        cc = sv
        h = float((cc[ix, ix, :] * eps[ix, ix, :]).sum())
        V = s.vof_diagnostics()["volume"]
        a = math.sqrt(max((6.0 * V / (math.pi * h) - h * h) / 3.0, 1e-12))
        return h, a, 2.0 * math.degrees(math.atan2(h, a))

    dt_cap = 0.5 * s.capillary_dt()
    dt = dt_cap
    s.set_dt(dt)
    t, maxit, capped = 0.0, 0, 0
    for i in range(steps):
        c = s.vof_max_courant()
        if c > 0.15:
            dt = min(dt_cap, dt * 0.15 / c)
            s.set_dt(dt)
        elif dt < dt_cap and c < 0.075:
            dt = min(dt_cap, 1.2 * dt)
            s.set_dt(dt)
        s.step()
        t += dt
        maxit = max(maxit, s.last_pressure_iterations())
        capped += 1 if s.last_pressure_iterations() >= PRESS_MAXIT else 0
        if (i + 1) % probe == 0:
            h, a, th = shape(s.get_vof())
            cd = s.contact_angle_diagnostics()
            hist.append((t, h, a, th, cd.get("mean_imposed_theta", 0.0),
                         cd.get("max_Ca_cl", 0.0), cd.get("max_contact_speed", 0.0)))
    print(f"  {'t':>9} {'h':>7} {'a':>7} {'theta_app':>9} {'theta_imp':>9} {'Ca_cl(dadt)':>12} "
          f"{'max Ca_cl':>10}")
    rows = []
    for k in range(1, len(hist)):
        t1, h1, a1, th1, ti1, ca1, u1 = hist[k]
        t0, _, a0, _, _, _, _ = hist[k - 1]
        ca_meas = mu * (a1 - a0) / (t1 - t0) / sigma
        print(f"  {t1:9.3f} {h1:7.3f} {a1:7.3f} {th1:9.3f} {ti1:9.3f} {ca_meas:12.4e} {ca1:10.3e}")
        rows.append((ca_meas, th1, a1))
    # the fit over the window Ca in [1e-3, 1e-1]
    sel = [(c, th, a) for c, th, a in rows if 1e-3 <= c <= 1e-1]
    slope = want = am = float("nan")
    if len(sel) >= 3:
        x = np.array([c for c, _, _ in sel])
        y = np.array([(th * DEG) ** 3 - (theta_e * DEG) ** 3 for _, th, _ in sel])
        am = float(np.mean([a for _, _, a in sel]))
        slope = float(np.polyfit(x, y, 1)[0])
        want = 9.0 * math.log(am / slip)
        print(f"  fit over {len(sel)} points with Ca in [1e-3, 1e-1] (mean a = {am:.2f} cells): "
              f"slope {slope:.3f} vs 9 ln(a/lambda) = {want:.3f}  "
              f"({100*(slope/want-1):+.1f} %, gate 25 %)  "
              f"{'PASS' if abs(slope/want - 1) <= 0.25 else 'FAIL'}")
        print(f"  effective inner cut-off implied by the measured slope: "
              f"lambda_eff = a exp(-slope/9) = {am*math.exp(-slope/9.0):.4f} cells "
              f"(prescribed lambda = {slip} cells)")
    else:
        print(f"  fit: only {len(sel)} points fell in the Ca window — inconclusive")
    # Tanner
    late = [(t, a) for t, _, a, _, _, _, _ in hist[len(hist) // 2:]]
    if len(late) >= 4:
        lt = np.log(np.array([t for t, _ in late]))
        la = np.log(np.array([a for _, a in late]))
        n_t = float(np.polyfit(lt, la, 1)[0])
        print(f"  Tanner: fitted a ~ t^{n_t:.4f} over the late half (the law is 1/10 = 0.1)")
    print(f"  pressure iters max {maxit} (capped {capped}); final theta_app "
          f"{hist[-1][3]:.3f} vs theta_e {theta_e}")
    return slope, want, am


# ---------------------------------------------------------------------------------- G3 dynamics
def g3_rise(theta_e=30.0, slips=(0.05, 0.3), steps=1500, nx=96, ny=4, nz=112, w=16.0,
            plate=8.0, dg=3.0e-3, sigma=1.0, mu=0.2, ratio=10.0, datum=28.0, zp=(12.0, 100.0),
            static_control=True, wall_slip=False, probe=None, lw=False):
    """G3-dynamics: the same plate scene from a FLAT interface, so the contact line has to travel.
    The static equilibrium level difference is Jurin's, EXACTLY (see g3_static), so the gate is the
    final height against it; the damping class (asymptotic vs oscillatory) is the second half of
    the Gruending et al. (AMM 86:142, 2020) comparison."""
    sdf, xm_in, xm_out, w_out, subsdf = jurin_scene(nx, ny, nz, w, plate, zp=zp)
    ref = jurin(w, theta_e, dg, sigma) - jurin(w_out, theta_e, dg, sigma)
    xin, xout = _channel_columns(sdf, nz)
    print(f"\n=== G3-dynamics  CAPILLARY RISE with the DYNAMIC angle — grid {nx}x{ny}x{nz}, "
          f"gap {w:.0f}, outer {w_out:.0f}, theta_e {theta_e}, sigma {sigma}, mu_l {mu}, "
          f"drho*g {dg:g}, ratio {ratio:g}, {steps} steps, FLAT start")
    print(f"    static equilibrium level DIFFERENCE (Jurin, exact for this read-out) {ref:.4f} "
          f"cells;  Oh = mu/sqrt(rho sigma w) = {mu/math.sqrt(1.0*sigma*w):.4f}")
    cases = [("static (control)", None)] if static_control else []
    cases += [(f"lambda = {sl:g} cells", sl) for sl in slips]
    for label, sl in cases:
        c0 = jurin_colour(nx, ny, nz, subsdf, lambda x: np.full_like(x, datum))
        s = pf.Solver(nx, ny, nz)
        s.set_rho(1.0)
        s.set_mu(mu)
        s.set_solid(sdf, cutcell_pressure=True)
        s.enable_vof()
        s.set_property_model("rho", "linear", "C", [1.0 / ratio, 1.0 - 1.0 / ratio])
        s.set_property_model("mu", "linear", "C", [mu / ratio, mu - mu / ratio])
        s.set_vof(c0)
        set_zero_mean_buoyancy(s, dg, ratio)
        s.set_surface_tension(sigma)
        if sl is None:
            s.set_contact_angle(theta_e)
        else:
            s.set_contact_angle_dynamic(theta_e, sl, mu, sigma)
        if wall_slip and sl is not None:      # WO-V6b: the SAME lambda in the momentum closure
            s.set_wall_slip_length(sl)
        eps = s.vof_geometry(0)
        v0 = s.vof_diagnostics()["volume"]
        dt, maxit, capped, trace, tend = relax(
            s, steps, probe=probe if probe else max(1, steps // 15),
            cb=lambda q: level_of(q.get_vof(), eps, xin) - level_of(q.get_vof(), eps, xout))
        h = [v for _, _, v in trace]
        final, peak = h[-1], max(h)
        over = (peak - final) / max(abs(final), 1e-12)
        cd = s.contact_angle_diagnostics() if sl is not None else None
        d = s.vof_diagnostics()
        print(f"  {label:20s}: final {final:.3f} vs equilibrium {ref:.3f} "
              f"({100*(final/ref-1):+.1f} %, gate 10 %)  "
              f"{'PASS' if abs(final/ref-1) <= 0.10 else 'FAIL'} | peak {peak:.3f} -> overshoot "
              f"{100*over:.1f} % => {'OSCILLATORY' if over > 0.02 else 'ASYMPTOTIC'}")
        if cd:
            print(f"     mean imposed theta {cd['mean_imposed_theta']:.2f} (theta_e {theta_e}), "
                  f"mean apparent {cd['mean_apparent_theta']:.2f}, max|Ca_cl| {cd['max_Ca_cl']:.3e}, "
                  f"max|U_cl| {cd['max_contact_speed']:.3e}, dynamic cells {cd['dynamic_cells']}")
        print(f"     dV/V {abs(d['volume']-v0)/v0:.2e}  iters {maxit} (capped {capped})  "
              f"t_end {tend:.1f}  max|u| {max_u(s):.2e}")
        print("     h(t): " + " ".join(f"{t:.0f}:{v:.2f}" for _, t, v in trace))
        if lw:
            # GATE 4 (WO-V6b): the Lucas-Washburn early-time law h^2 = (sigma w cos(theta)/(3 mu)) t
            # for a slot of width w.  Gravity is neglected in that law, so the fit window is the
            # part of the trace with h <= 0.4 * the Jurin equilibrium (where the hydrostatic term
            # is <= 40 % of the capillary drive) and h >= 2 cells (below that the meniscus is not
            # yet formed).  The slope of h^2 against t is the measured coefficient.
            k_lw = sigma * w * math.cos(math.radians(theta_e)) / (3.0 * mu)
            pts = [(t, v) for _, t, v in trace if 2.0 <= v <= 0.4 * ref]
            if len(pts) >= 3:
                tt = np.array([t for t, _ in pts])
                hh = np.array([v for _, v in pts]) ** 2
                k = float(np.polyfit(tt, hh, 1)[0])
                print(f"     LUCAS-WASHBURN: d(h^2)/dt measured {k:.4g} vs "
                      f"sigma w cos(theta)/(3 mu) = {k_lw:.4g}  "
                      f"({100*(k/k_lw-1):+.1f} %, gate 20 %)  "
                      f"{'PASS' if abs(k/k_lw - 1) <= 0.20 else 'FAIL'}   "
                      f"[{len(pts)} points, h in {pts[0][1]:.2f}..{pts[-1][1]:.2f} cells]")
            else:
                print(f"     LUCAS-WASHBURN: only {len(pts)} trace points in the window "
                      f"h in [2, {0.4*ref:.1f}] cells — the rise never got there")


# ------------------------------------------------------------------------------------- G4 incline
def g4_incline(theta_a=70.0, theta_r=50.0, bo_fracs=(0.5, 0.7, 1.5, 2.5), steps=1200, nx=48, nz=32,
               R=8.0, zw=4.25, sigma=1.0, oh=0.2, slip=0.1):
    """The wall is kept axis-aligned and GRAVITY is tilted — the same problem, an exact SDF.
    Bo = drho g V^(2/3) sin(alpha) / sigma against Bo_c = cos(theta_r) - cos(theta_a)
    (ElSherbini & Jacobi 2006's retention relation in its simplest form)."""
    mu = oh * math.sqrt(1.0 * sigma * 2.0 * R)
    V = 2.0 / 3.0 * math.pi * R ** 3
    bo_c = math.cos(theta_r * DEG) - math.cos(theta_a * DEG)
    print(f"\n=== G4  HYSTERESIS: a drop on an INCLINE — theta_a {theta_a}, theta_r {theta_r}, "
          f"lambda {slip} cells, D/dx {2*R:.0f}, grid {nx}x{nx}x{nz}, wall z = {zw}, mu {mu:.4f} "
          f"(Oh {oh}), {steps} steps")
    print(f"    Bo_c = cos(theta_r) - cos(theta_a) = {bo_c:.4f};  V^(2/3) = {V**(2/3):.2f}")
    z = (np.arange(nz) + 0.5)[None, None, :]
    sdf = np.asfortranarray(
        np.broadcast_to(np.minimum(z - zw, (nz - zw) - z), (nx, nx, nz)).astype(float))
    c0 = cap_colour(nx, nx, nz, nx * 0.5, nx * 0.5, zw, R, 0.5 * (theta_a + theta_r))
    for fr in bo_fracs:
        g_t = fr * bo_c * sigma / V ** (2.0 / 3.0)   # the TANGENTIAL body force per unit volume
        s = pf.Solver(nx, nx, nz)
        s.set_rho(1.0)
        s.set_mu(mu)
        s.set_solid(sdf, cutcell_pressure=True)
        s.enable_vof()
        s.set_vof(c0)
        s.set_surface_tension(sigma)
        s.set_contact_angle(0.5 * (theta_a + theta_r))
        s.set_contact_angle_dynamic(0.5 * (theta_a + theta_r), slip, mu, sigma)
        s.set_contact_angle_hysteresis(theta_a, theta_r)
        # ZERO-MEAN tangential force: `g_t (C - Cbar)`. A non-zero-mean body force in a fully
        # periodic box accelerates the whole fluid without bound (WO-Q finding 9), and the
        # accelerating frame then cancels part of the very force this gate is measuring.
        cc = s.get_vof()
        eps = s.vof_geometry(0)
        cbar = float((cc * eps).sum()) / float(eps.sum())
        s.set_property_model("force_x", "linear", "C", [-g_t * cbar, g_t])
        xg = (np.arange(nx) + 0.5)[:, None, None]
        w0 = float((cc * eps).sum())
        x0 = float((cc * eps * xg).sum()) / w0
        dt, maxit, capped, trace, tend = relax(
            s, steps, probe=max(1, steps // 8),
            cb=lambda q: float((q.get_vof() * eps * xg).sum()) / max(float((q.get_vof() * eps).sum()), 1e-30))
        cc1 = s.get_vof()
        x1 = float((cc1 * eps * xg).sum()) / max(float((cc1 * eps).sum()), 1e-30)
        cd = s.contact_angle_diagnostics()
        moved = x1 - x0
        pin_frac = cd["pinned_cells"] / max(cd["dynamic_cells"], 1)
        verdict = "PINNED" if abs(moved) < 0.5 else "SLIDING"
        print(f"  Bo/Bo_c {fr:4.2f} (f_t {g_t:.4e}): centroid moved {moved:+7.3f} cells over "
              f"t = {tend:.1f}  -> {verdict}   pinned {cd['pinned_cells']}/{cd['dynamic_cells']} "
              f"({100*pin_frac:.0f} %)  advancing {cd['advancing_cells']} receding "
              f"{cd['receding_cells']}  mean imposed {cd['mean_imposed_theta']:.2f}  "
              f"max|Ca_cl| {cd['max_Ca_cl']:.2e}  iters {maxit} (capped {capped})")
        print("     centroid trace: " + " ".join(f"{t:.1f}:{v:.2f}" for _, t, v in trace))


# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    which = args or ["jurin"]
    if "jurin" in which:
        g3_static(theta=30.0, steps=200 if quick else 600)
        if not quick:
            g3_static(theta=60.0, steps=600)
            g3_static(theta=30.0, steps=1200, flat_start=True)
    if "jurin_wos" in which:
        g3_static(theta=30.0, steps=600, wos=True)
    if "spread" in which:
        g2_spread(steps=1200 if quick else 2000, probe=50)
    if "slipsweep" in which:
        # THE gate on the MODEL rather than on the scheme's own inner cut-off: the Cox-Voinov
        # slope must move by exactly 9 dln(1/lambda) when lambda changes, whatever the scheme's
        # residual sub-grid slip adds on top of it.
        print("\n=== G2b  the SLIP SENSITIVITY of the fitted Cox-Voinov slope")
        res = []
        for sl in (0.02, 0.1, 0.5):
            sp, wt, am = g2_spread(slip=sl, steps=800, probe=50)
            res.append((sl, sp, wt, am))
        print("\n  lambda   fitted slope   9 ln(a/lambda)   difference")
        for sl, sp, wt, am in res:
            print(f"  {sl:6.2f}   {sp:12.3f}   {wt:14.3f}   {sp-wt:+10.3f}")
        for i in range(1, len(res)):
            d_meas = res[i][1] - res[i - 1][1]
            d_pred = 9.0 * math.log(res[i - 1][0] / res[i][0])
            print(f"  lambda {res[i-1][0]} -> {res[i][0]}: d(slope) measured {d_meas:+.3f} vs "
                  f"9 dln(1/lambda) = {d_pred:+.3f}  ({100*(d_meas/d_pred-1):+.1f} %, gate 25 %)  "
                  f"{'PASS' if abs(d_meas/d_pred - 1) <= 0.25 else 'FAIL'}")
    if "slipsweep_v6b" in which:
        # WO-V6b gate 3: the SAME sweep with the momentum-side Navier slip ON, so the resolved
        # hydrodynamics between the cell size and the contact radius can supply the part of
        # 9 ln(a/lambda) that the band fill alone could not (WO-V6 finding 7).
        print("\n=== G2b/V6b  the SLIP SENSITIVITY with the MOMENTUM Navier closure ON")
        res = []
        for sl in (0.02, 0.1, 0.5):
            sp, wt, am = g2_spread(slip=sl, steps=800, probe=50, wall_slip=True)
            res.append((sl, sp, wt, am))
        print("\n  lambda   fitted slope   9 ln(a/lambda)   difference")
        for sl, sp, wt, am in res:
            print(f"  {sl:6.2f}   {sp:12.3f}   {wt:14.3f}   {sp-wt:+10.3f}")
        for i in range(1, len(res)):
            d_meas = res[i][1] - res[i - 1][1]
            d_pred = 9.0 * math.log(res[i - 1][0] / res[i][0])
            print(f"  lambda {res[i-1][0]} -> {res[i][0]}: d(slope) measured {d_meas:+.3f} vs "
                  f"9 dln(1/lambda) = {d_pred:+.3f}  ({100*(d_meas/d_pred-1):+.1f} %, gate 25 %)  "
                  f"{'PASS' if abs(d_meas/d_pred - 1) <= 0.25 else 'FAIL'}")
    if "lw" in which:
        # WO-V6b gate 4: Lucas-Washburn on the repaired plate scene, momentum slip ON vs OFF.
        g3_rise(steps=400 if quick else 1200, slips=(0.05, 0.3), probe=20, lw=True,
                wall_slip=False, static_control=True)
        g3_rise(steps=400 if quick else 1200, slips=(0.05, 0.3), probe=20, lw=True,
                wall_slip=True, static_control=False)
    if "rise" in which:
        g3_rise(steps=800 if quick else 1500)
    if "incline" in which:
        g4_incline(steps=800 if quick else 1500)
