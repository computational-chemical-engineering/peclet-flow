#!/usr/bin/env python3
"""WO-S (rung V5b) — the static contact angle on SDF solids: the gate battery, in Python.

Every gate prints its numbers; nothing is summarised into an adjective.

  G1  DROP ON A FLAT SDF WALL. A liquid cap on a flat wall at a HALF-INTEGER z (so the wall cells
      are genuinely cut) with surface tension and a prescribed theta, run to rest. The equilibrium
      of a wetting condition is a spherical cap, so the apparent angle is read off the cap
      relations: h from the colour column on the axis (weighted by the cell FLUID fraction, since C
      is a fraction of that), the conserved liquid volume V from `vof_diagnostics()['volume']`,
      then a = sqrt((6V/(pi h) - h^2)/3) and theta = 2 atan(h/a). Reading `a` off the first fluid
      plane instead biases it (that plane's colour is the mean over z in [z_wall, z_wall+eps], not
      the section AT the wall), and the bias grows as the cap flattens — both are printed.
      Reported alongside: the near-wall spurious Ca in the OPEN fluid and the raw max|u| including
      the wall band (WO-Q finding 8's open question), the band census, the volume drift and the
      pressure iteration count against its cap.

  G2  DROP ON AN SDF SPHERE. A liquid drop resting on a solid sphere of radius Rs. The equilibrium
      free surface is a spherical cap of radius Rc whose sphere meets the solid sphere at theta;
      the law of cosines gives the centre distance d^2 = Rs^2 + Rc^2 - 2 Rs Rc cos(theta) and the
      liquid volume is (4/3) pi Rc^3 minus the two-sphere lens, so (V, theta) -> Rc by a 1-D root
      solve. The run is measured the other way round: (V, apex height H = d + Rc) -> (Rc, d) ->
      theta.

  G4  CAPILLARY RISE / JURIN. Two SDF plates a gap `w` apart in a periodic box whose OUTER channel
      has width `w_out`, with a liquid bath below them and gravity. Both channels rise, so the
      measured level difference is Jurin's `2 sigma cos(theta) / (drho g) * (1/w - 1/w_out)`.

Usage:
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_wetting.py [g1 g2 g4 ...]
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_wetting.py g1 --quick
"""
import math
import sys

import numpy as np

import peclet.flow as pf

PRESS_MAXIT = 300
DEG = math.pi / 180.0


# --------------------------------------------------------------------------------------- helpers
def cap_sphere(volume, theta_deg):
    """The sphere (radius Rc, centre offset below the wall) whose spherical cap of contact angle
    `theta` above the wall has the given volume: h = Rc(1 - cos t), a = Rc sin t, and
    V = pi Rc^3 (1-cos t)^2 (2 + cos t)/3."""
    ct = math.cos(theta_deg * DEG)
    Rc = (3.0 * volume / (math.pi * (1.0 - ct) ** 2 * (2.0 + ct))) ** (1.0 / 3.0)
    return Rc, -Rc * ct  # (radius, centre z relative to the wall)


def cap_colour(nx, ny, nz, cx, cy, zw, R, theta_deg=90.0):
    """A liquid spherical cap of the same VOLUME as a hemisphere of radius R, resting on the wall
    z = zw with contact angle `theta`, by 4^3 subsampling of the FLUID part of each cell (C is the
    liquid fraction of the fluid volume, `VOF_PLAN` rule 2)."""
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


def max_u(s, zlo=None):
    m = 0.0
    for f in (s.get_u, s.get_v, s.get_w):
        v = np.abs(f())
        m = max(m, float(v.max()) if zlo is None else float(v[:, :, zlo:].max()))
    return m


def relax(s, steps, cfl=0.15, dt_cap=None, probe=0):
    """Run `steps` steps with a dt kept under the WY interface CFL and the capillary limit."""
    dt_cap = dt_cap if dt_cap is not None else 0.5 * s.capillary_dt()
    dt = dt_cap
    s.set_dt(dt)
    maxit, capped = 0, 0
    trace = []
    for i in range(steps):
        c = s.vof_max_courant()
        if c > cfl:
            dt = min(dt_cap, dt * cfl / c)
            s.set_dt(dt)
        elif dt < dt_cap and c < 0.5 * cfl:
            dt = min(dt_cap, 1.2 * dt)
            s.set_dt(dt)
        s.step()
        it = s.last_pressure_iterations()
        maxit = max(maxit, it)
        capped += 1 if it >= PRESS_MAXIT else 0
        if probe and (i + 1) % probe == 0:
            trace.append((i + 1, max_u(s)))
    return dt, maxit, capped, trace


# ------------------------------------------------------------------------------------------- G1
def g1(thetas=(30.0, 60.0, 90.0, 120.0, 150.0), ratio=1.0, steps=600, R=12.0, nx=64, nz=40,
       zw=4.25, sigma=1.0, oh=0.1, pivot=0, init_offset=0.0, init_abs=None):
    """`init_offset`: the initial cap is at `theta_set + init_offset` (clamped to [10,170]) so the
    run has to MOVE the contact line to the prescribed angle; `init_abs` overrides it with a fixed
    initial angle (90 = the hemisphere, the largest excursion).

    `zw` is a QUARTER-integer, not the half-integer the work order asks for, and that is a measured
    correction rather than a convenience: at exactly k + 1/2 the tangential MAC faces of the
    wall-adjacent cell sit ON the SDF zero level, `buildOpenness` (sdf > 0 is fluid) closes them,
    and the cell is tangentially isolated — the contact line then cannot move at all and the
    unrelieved Young force appears as a large velocity on those closed-face DOFs. At k + 1/4 the
    wall cell is still genuinely CUT (eps = 3/4, face openness 3/4) and the contact line is mobile.
    See the WO-S findings and `g1_wall_placement`."""
    mu = oh * math.sqrt(1.0 * sigma * 2.0 * R)
    print(f"\n=== G1  drop on a FLAT SDF wall — D/dx {2*R:.0f}, grid {nx}x{nx}x{nz}, wall z = {zw}, "
          f"sigma {sigma}, mu {mu:.4f} (Oh {oh}), rho ratio {ratio:g}, {steps} steps, pivot {pivot}, "
          f"init {'theta_set%+.0f' % init_offset if init_abs is None else '%.0f deg' % init_abs}")
    # A SLIT, not a single wall: the box is periodic, so a lone ramp SDF would make the wrap plane
    # a spurious fluid/solid seam three cells from the band. With a second wall at nz - zw the wrap
    # sits deep inside the solid and the SDF is continuous across it.
    z = (np.arange(nz) + 0.5)[None, None, :]
    sdf = np.asfortranarray(
        np.broadcast_to(np.minimum(z - zw, (nz - zw) - z), (nx, nx, nz)).astype(float))
    rows = []
    for th in thetas:
        th0 = init_abs if init_abs is not None else min(170.0, max(10.0, th + init_offset))
        c0 = cap_colour(nx, nx, nz, nx * 0.5, nx * 0.5, zw, R, th0)
        s = pf.Solver(nx, nx, nz)
        s.set_rho(1.0)
        s.set_mu(mu)
        s.set_solid(sdf, cutcell_pressure=True)
        s.enable_vof()
        if ratio != 1.0:
            s.set_property_model("rho", "linear", "C", [1.0 / ratio, 1.0 - 1.0 / ratio])
            s.set_property_model("mu", "linear", "C", [mu / ratio, mu - mu / ratio])
        s.set_vof(c0)
        s.set_surface_tension(sigma)
        if pivot:
            s.set_contact_angle_pivot(pivot)
        s.set_contact_angle(th)
        v0 = s.vof_diagnostics()["volume"]
        dt, maxit, capped, _ = relax(s, steps)
        d = s.vof_diagnostics()
        cd = s.contact_angle_diagnostics()
        cc = s.get_vof()
        eps = s.vof_geometry(0)
        ix = iy = nx // 2
        h = float((cc[ix, iy, :] * eps[ix, iy, :]).sum())
        V = d["volume"]
        a = math.sqrt(max((6.0 * V / (math.pi * h) - h * h) / 3.0, 1e-12))
        theta = 2.0 * math.degrees(math.atan2(h, a))
        z0 = int(math.ceil(zw))
        a_dir = math.sqrt(float(cc[:, :, z0].sum()) / math.pi)
        th_dir = 2.0 * math.degrees(math.atan2(h, a_dir))
        um_open = max_u(s, z0 + 1)
        um_all = max_u(s)
        print(f"  theta_set {th:5.1f} (init {th0:5.1f}) -> theta {theta:7.3f} (err {theta-th:+6.3f})  "
              f"h {h:7.3f}  a {a:7.3f} | contour a {a_dir:7.3f} -> theta {th_dir:7.3f} | "
              f"dV/V {abs(V-v0)/v0:.2e}  Ca(open) {mu*um_open/sigma:.3e}  max|u| all {um_all:.3e} | "
              f"band th/nbr/pure/par/neu/unf {cd['contact_cells']}/{cd['neighbour_cells']}/"
              f"{cd['pure_cells']}/{cd['parallel_cells']}/{cd['neutral_cells']}/{cd['unfilled_cells']}"
              f"  apparent {cd['mean_apparent_angle']:6.2f}"
              f" | iters {maxit} (capped {capped})  dt {dt:.4f}  solid sum {d['solid_sum']:.1e}")
        rows.append((th, theta, abs(V - v0) / v0, mu * um_open / sigma, um_all, maxit, capped))
    worst = max(abs(t - s_) for s_, t, *_ in rows)
    print(f"  G1 worst |theta - theta_set| = {worst:.3f} deg   (gate 3.0)   "
          f"{'PASS' if worst <= 3.0 else 'FAIL'}")
    return rows



# ---------------------------------------------------- G1D: the same drop on a DOMAIN-BC wall
def g1_domain(thetas=(60.0, 90.0, 120.0), ratio=1.0, steps=500, R=12.0, nx=64, nz=40,
              sigma=1.0, oh=0.1):
    """ISSUES sweep item 3. The G1 scene with the wall as a DOMAIN FACE (bc type 1) instead of an
    SDF slab: the wall plane is z = 0 exactly, the wall-adjacent cell is WHOLE (no cut cell at
    all), and the theta band is the colour block's three ghost layers below z = 0.

    Reference: `g1` on the same drop against an SDF wall at a quarter-integer z. The two are
    different discretisations of the same physics -- one cuts the wall cell, the other does not --
    so the gate is agreement, not identity."""
    mu = oh * math.sqrt(1.0 * sigma * 2.0 * R)
    print(f"\n=== G1D drop on a DOMAIN-BC wall (bc type 1 at z = 0) — D/dx {2*R:.0f}, grid "
          f"{nx}x{nx}x{nz}, sigma {sigma}, mu {mu:.4f} (Oh {oh}), rho ratio {ratio:g}, "
          f"{steps} steps")
    rows = []
    for th in thetas:
        c0 = cap_colour(nx, nx, nz, nx * 0.5, nx * 0.5, 0.0, R, th)
        s = pf.Solver(nx, nx, nz)
        s.set_rho(1.0)
        s.set_mu(mu)
        s.set_domain_bc(4, 1)    # -z: the wetting wall
        s.set_domain_bc(5, 1)    # +z: a lid, far from the drop
        s.set_pressure_geometry(np.full((nx, nx, nz), 1e30, order="F"))
        s.enable_vof()
        if ratio != 1.0:
            s.set_property_model("rho", "linear", "C", [1.0 / ratio, 1.0 - 1.0 / ratio])
            s.set_property_model("mu", "linear", "C", [mu / ratio, mu - mu / ratio])
        s.set_vof(c0)
        s.set_surface_tension(sigma)
        s.set_contact_angle(th)
        v0 = s.vof_diagnostics()["volume"]
        dt, maxit, capped, _ = relax(s, steps)
        d = s.vof_diagnostics()
        cd = s.contact_angle_diagnostics()
        cc = s.get_vof()
        eps = s.vof_geometry(0)
        ix = iy = nx // 2
        h = float((cc[ix, iy, :] * eps[ix, iy, :]).sum())
        V = d["volume"]
        a = math.sqrt(max((6.0 * V / (math.pi * h) - h * h) / 3.0, 1e-12))
        theta = 2.0 * math.degrees(math.atan2(h, a))
        a_dir = math.sqrt(float(cc[:, :, 0].sum()) / math.pi)
        th_dir = 2.0 * math.degrees(math.atan2(h, a_dir))
        print(f"  theta_set {th:5.1f} -> theta {theta:7.3f} (err {theta-th:+6.3f})  "
              f"h {h:7.3f}  a {a:7.3f} | contour a {a_dir:7.3f} -> theta {th_dir:7.3f} | "
              f"dV/V {abs(V-v0)/v0:.2e}  max|u| {max_u(s):.3e} | "
              f"band th/nbr/pure/par/neu/unf {cd['contact_cells']}/{cd['neighbour_cells']}/"
              f"{cd['pure_cells']}/{cd['parallel_cells']}/{cd['neutral_cells']}/"
              f"{cd['unfilled_cells']}  apparent {cd['mean_apparent_angle']:6.2f}"
              f" | iters {maxit} (capped {capped})  dt {dt:.4f}")
        rows.append((th, theta))
    return rows


def g1_domain_vs_sdf(thetas=(60.0, 90.0, 120.0), steps=500, gate=1.0):
    """The ISSUES-sweep item-3 gate: the domain-wall equilibrium against the SDF-wall one."""
    ref = {t: v for t, v in ((r[0], r[1]) for r in g1(thetas=thetas, steps=steps))}
    dom = {t: v for t, v in g1_domain(thetas=thetas, steps=steps)}
    print(f"\n  --- item 3 gate: DOMAIN wall vs SDF wall, {steps} steps")
    worst = 0.0
    for t in thetas:
        d = dom[t] - ref[t]
        worst = max(worst, abs(d))
        print(f"    theta_set {t:5.1f}:  SDF {ref[t]:7.3f}   DOMAIN {dom[t]:7.3f}   "
              f"difference {d:+6.3f} deg")
    print(f"  worst |domain - SDF| = {worst:.3f} deg (gate {gate})  "
          f"{'PASS' if worst <= gate else 'FAIL'}")
    return worst


# --------------------------------------------------------------------- the wall-placement sweep
def g1_wall_placement(theta=60.0, steps=1200, fracs=(0.0, 0.25, 0.5, 0.75)):
    """WHERE the SDF wall sits inside the cell decides whether the contact line can move at all.
    One theta, four wall placements, everything else identical; the run starts from a hemisphere so
    the contact line HAS to travel."""
    print(f"\n=== G1w wall placement — theta {theta}, initial cap 90 deg, {steps} steps")
    for f in fracs:
        rows = g1(thetas=(theta,), steps=steps, zw=4.0 + f, init_abs=90.0)
        print(f"   ^ wall at z = {4.0 + f}")


# ------------------------------------------------------------------------------------------- G2
def lens_volume(Rs, Rc, d):
    """Volume of the intersection of spheres (0, Rs) and (d, Rc)."""
    if d >= Rs + Rc:
        return 0.0
    if d <= abs(Rs - Rc):
        return 4.0 / 3.0 * math.pi * min(Rs, Rc) ** 3
    return (math.pi * (Rs + Rc - d) ** 2
            * (d * d + 2 * d * Rc - 3 * Rc * Rc + 2 * d * Rs + 6 * Rc * Rs - 3 * Rs * Rs)
            / (12.0 * d))


def drop_volume(Rs, Rc, d):
    return 4.0 / 3.0 * math.pi * Rc ** 3 - lens_volume(Rs, Rc, d)


def cap_from_volume(Rs, V, theta_deg):
    """(V, theta) -> the cap sphere radius Rc and centre distance d (the reference shape)."""
    ct = math.cos(theta_deg * DEG)
    lo, hi = 1e-3, 50.0 * Rs
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        d = math.sqrt(max(Rs * Rs + mid * mid - 2 * Rs * mid * ct, 0.0))
        if drop_volume(Rs, mid, d) < V:
            lo = mid
        else:
            hi = mid
    Rc = 0.5 * (lo + hi)
    return Rc, math.sqrt(max(Rs * Rs + Rc * Rc - 2 * Rs * Rc * ct, 0.0))


def theta_from_volume_apex(Rs, V, H):
    """(V, apex height H = d + Rc) -> (Rc, d, theta)."""
    lo, hi = 1e-3, H
    for _ in range(200):
        Rc = 0.5 * (lo + hi)
        d = H - Rc
        if drop_volume(Rs, Rc, max(d, 0.0)) < V:
            lo = Rc
        else:
            hi = Rc
    Rc = 0.5 * (lo + hi)
    d = H - Rc
    ct = (Rs * Rs - d * d + Rc * Rc) / (2.0 * Rs * Rc)
    return Rc, d, math.degrees(math.acos(max(-1.0, min(1.0, ct))))


def g2(thetas=(60.0, 90.0, 120.0), steps=600, Rs=12.0, Rd=8.0, n=64, sigma=1.0, oh=0.1):
    mu = oh * math.sqrt(1.0 * sigma * 2.0 * Rd)
    cx = cy = cz = n * 0.5
    ax = (np.arange(n) + 0.5)[:, None, None]
    ay = (np.arange(n) + 0.5)[None, :, None]
    az = (np.arange(n) + 0.5)[None, None, :]
    sdf = np.asfortranarray(np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2 + (az - cz) ** 2) - Rs)
    V = 4.0 / 3.0 * math.pi * Rd ** 3
    print(f"\n=== G2  drop on an SDF SPHERE — Rs {Rs}, drop volume of a sphere R {Rd} (V {V:.2f}), "
          f"grid {n}^3, sigma {sigma}, mu {mu:.4f}, {steps} steps")
    rows = []
    for th in thetas:
        Rc0, d0 = cap_from_volume(Rs, V, th)
        # initial condition: the reference cap itself (the gate is that it STAYS, and that a
        # different theta relaxes to its own cap — the 90-degree row starts off-equilibrium too
        # because the fill is a plane, not a sphere)
        sub = (np.arange(4) + 0.5) / 4.0
        px = np.arange(n)[:, None, None, None, None, None] + sub[None, None, None, :, None, None]
        py = np.arange(n)[None, :, None, None, None, None] + sub[None, None, None, None, :, None]
        pz = np.arange(n)[None, None, :, None, None, None] + sub[None, None, None, None, None, :]
        rs2 = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
        rc2 = (px - cx) ** 2 + (py - cy) ** 2 + (pz - (cz + d0)) ** 2
        fluid = rs2 > Rs * Rs
        liq = fluid & (rc2 < Rc0 * Rc0)
        tot = fluid.sum(axis=(3, 4, 5))
        c0 = np.asfortranarray(np.where(tot > 0, liq.sum(axis=(3, 4, 5)) / np.maximum(tot, 1), 0.0)
                               .astype(float))
        s = pf.Solver(n, n, n)
        s.set_rho(1.0)
        s.set_mu(mu)
        s.set_solid(sdf, cutcell_pressure=True)
        s.enable_vof()
        s.set_vof(c0)
        s.set_surface_tension(sigma)
        s.set_contact_angle(th)
        v0 = s.vof_diagnostics()["volume"]
        dt, maxit, capped, _ = relax(s, steps)
        d = s.vof_diagnostics()
        cd = s.contact_angle_diagnostics()
        cc = s.get_vof()
        eps = s.vof_geometry(0)
        ix = iy = int(cx)
        col = cc[ix, iy, :] * eps[ix, iy, :]
        H = float(col[int(cz):].sum()) + Rs  # apex height above the solid centre
        Vm = d["volume"]
        Rc, dd, theta = theta_from_volume_apex(Rs, Vm, H)
        print(f"  theta_set {th:5.1f} -> theta {theta:7.3f} (err {theta-th:+6.3f})  "
              f"Rc {Rc:6.3f} vs ref {Rc0:6.3f} ({100*(Rc-Rc0)/Rc0:+5.2f} %)  H {H:6.3f} "
              f"(ref {d0+Rc0:6.3f}) | dV/V {abs(Vm-v0)/v0:.2e} "
              f"Ca {mu*max_u(s)/sigma:.3e} | band th/nbr/pure/par/neu {cd['contact_cells']}/"
              f"{cd['pure_cells']}/{cd['parallel_cells']}/{cd['neutral_cells']} "
              f"apparent {cd['mean_apparent_angle']:6.2f} | iters {maxit} (capped {capped}) "
              f"solid sum {d['solid_sum']:.1e}")
        rows.append((th, theta, Rc, Rc0))
    worst = max(abs(t - s_) for s_, t, *_ in rows)
    worstR = max(abs(rc - r0) / r0 for *_, rc, r0 in rows)
    print(f"  G2 worst |theta - theta_set| = {worst:.3f} deg (gate 3.0), worst cap-radius error "
          f"{100*worstR:.2f} % (gate 3 %)  {'PASS' if worst <= 3.0 and worstR <= 0.03 else 'FAIL'}")
    return rows


# ------------------------------------------------------------------------------------------- G4
def box_sdf(ax, ay, az, xlo, xhi, zlo, zhi):
    qx = np.maximum(xlo - ax, ax - xhi)
    qz = np.maximum(zlo - az, az - zhi)
    out = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qz, 0.0) ** 2)
    return out + np.minimum(np.maximum(qx, qz), 0.0)


def g4(theta=30.0, steps=4000, nx=80, ny=4, nz=96, gap=24, plate=4, sigma=1.0, ratio=10.0,
       drho_g=2.4e-3, zbath=40.0, zplate=(16.0, 88.0), mu=0.2):
    """Jurin. Plates of thickness `plate` bounding a gap `gap` wide; the periodic OUTER channel is
    `nx - gap - 2*plate` wide. Both menisci rise, so the reference is the DIFFERENCE."""
    x1 = 0.5 * (nx - gap) - plate
    x2 = 0.5 * (nx + gap)
    w_out = nx - gap - 2 * plate
    ax = (np.arange(nx) + 0.5)[:, None, None]
    az = (np.arange(nz) + 0.5)[None, None, :]
    d1 = box_sdf(ax, None, az, x1, x1 + plate, zplate[0], zplate[1])
    d2 = box_sdf(ax, None, az, x2, x2 + plate, zplate[0], zplate[1])
    sdf = np.asfortranarray(np.broadcast_to(np.minimum(d1, d2), (nx, ny, nz)).astype(float))
    c0 = np.asfortranarray(np.broadcast_to(
        np.clip(zbath - (np.arange(nz) + 0.5), 0.0, 1.0)[None, None, :], (nx, ny, nz)).astype(float))
    h_pred = 2.0 * sigma * math.cos(theta * DEG) / drho_g * (1.0 / gap - 1.0 / w_out)
    bo = drho_g * gap * gap / sigma
    print(f"\n=== G4  Jurin — gap {gap}, outer channel {w_out}, plates x [{x1},{x1+plate}] and "
          f"[{x2},{x2+plate}], z [{zplate[0]},{zplate[1]}], grid {nx}x{ny}x{nz}, theta {theta}, "
          f"sigma {sigma}, drho*g {drho_g:g}, Bond {bo:.3f}, ratio {ratio:g}\n"
          f"    predicted level DIFFERENCE 2 sigma cos(theta)/(drho g) (1/w - 1/w_out) = "
          f"{h_pred:.3f} cells")
    s = pf.Solver(nx, ny, nz)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_solid(sdf, cutcell_pressure=True)
    s.enable_vof()
    s.set_property_model("rho", "linear", "C", [1.0 / ratio, 1.0 - 1.0 / ratio])
    s.set_property_model("mu", "linear", "C", [mu / ratio, mu - mu / ratio])
    # ZERO-MEAN buoyancy: in a fully periodic box a non-zero-mean force accelerates the whole fluid
    # without bound (WO-Q finding 9). Jurin depends only on drho, so subtracting a constant from the
    # force is exact for this problem.
    s.set_property_model("force_z", "linear", "rho", [drho_g / (ratio - 1.0), -drho_g / (1.0 - 1.0 / ratio)])
    s.set_vof(c0)
    s.set_surface_tension(sigma)
    s.set_contact_angle(theta)
    v0 = s.vof_diagnostics()["volume"]
    dt, maxit, capped, trace = relax(s, steps, probe=max(1, steps // 8))
    cc = s.get_vof()
    eps = s.vof_geometry(0)
    zc = np.arange(nz) + 0.5

    def level(xs):
        col = (cc[xs, :, :] * eps[xs, :, :]).sum(axis=(0, 1))
        tot = eps[xs, :, :].sum(axis=(0, 1))
        # the level is the integral of the liquid area fraction: sum over z of (liquid/fluid area)
        f = np.where(tot > 0, col / np.maximum(tot, 1e-30), 0.0)
        return float(f.sum())

    xin = np.arange(int(x1 + plate) + 2, int(x2) - 1)
    xout = np.concatenate([np.arange(0, int(x1) - 1), np.arange(int(x2 + plate) + 2, nx)])
    lin, lout = level(xin), level(xout)
    d = s.vof_diagnostics()
    print(f"  inner level {lin:.3f}  outer level {lout:.3f}  difference {lin-lout:.3f} cells "
          f"vs Jurin {h_pred:.3f}  ({100*((lin-lout)-h_pred)/h_pred:+.2f} %)")
    print(f"  dV/V {abs(d['volume']-v0)/v0:.3e}  max|u| {max_u(s):.3e}  iters {maxit} "
          f"(capped {capped})  dt {dt:.4f}  solid sum {d['solid_sum']:.1e}")
    print("  max|u| trace: " + " ".join(f"{n}:{u:.2e}" for n, u in trace))
    err = abs((lin - lout) - h_pred) / abs(h_pred)
    print(f"  G4 relative error {100*err:.2f} % (gate 5 %)  {'PASS' if err <= 0.05 else 'FAIL'}")


# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    which = args or ["g1", "g2"]
    if "g1" in which:
        # (a) the FIXED POINT: start at the prescribed angle; the equilibrium must stay there.
        g1(steps=200 if quick else 500)
        if not quick:
            g1(ratio=100.0, steps=500)
    if "g1b" in which:
        # (b) ATTRACTION: start from a hemisphere and let the contact line travel to the prescribed
        #     angle. Slow — the driving vanishes as the angle approaches its target — so only the
        #     two mid angles are run to convergence.
        g1(thetas=(60.0, 120.0), steps=1000 if quick else 3000, init_abs=90.0)
    if "g1d" in which:
        g1_domain_vs_sdf(steps=200 if quick else 500)
    if "g1w" in which:
        g1_wall_placement(steps=400 if quick else 1200)
    if "g2" in which:
        g2(steps=200 if quick else 600)
    if "g4" in which:
        g4(steps=800 if quick else 4000)
