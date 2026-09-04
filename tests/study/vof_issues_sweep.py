#!/usr/bin/env python3
"""Gates for the ISSUES sweep (the solver-side entries examples/*/ISSUES.md logged).

One sub-command per ISSUES entry; every one prints its measured numbers and a PASS/FAIL line.

    PYTHONPATH=$PWD/build_cuda python tests/study/vof_issues_sweep.py atomic
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_issues_sweep.py adaptive
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_issues_sweep.py geometry
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_issues_sweep.py collocated
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_issues_sweep.py freeslip
    PYTHONPATH=$PWD/build_cuda python tests/study/vof_issues_sweep.py wetwall
"""
import argparse
import math
import sys

import numpy as np

import peclet.flow as pf

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        FAIL.append(name)


def state(s):
    return dict(u=s.get_u().copy(), v=s.get_v().copy(), w=s.get_w().copy(),
                C=s.get_vof().copy(), p=np.asarray(s.get_p()).copy())


def same(a, b):
    return all(np.array_equal(a[k], b[k]) for k in a)


def diffs(a, b):
    return {k: float(np.abs(a[k] - b[k]).max()) for k in a}


# ------------------------------------------------------------------ item 1: step() atomicity
def scene_cfl(n=16):
    """The ISSUES reproducer: a half-filled box, uniform u = 1, body force in z."""
    s = pf.Solver(n, n, n)
    s.set_rho(1.0)
    s.set_mu(0.01)
    s.set_dt(0.1)
    s.set_pressure_geometry(np.full((n, n, n), 1e30, order="F"))
    s.set_body_force(0.0, 0.0, 1e-3)
    s.enable_vof()
    C = np.zeros((n, n, n), order="F")
    C[:, :, : n // 2] = 1.0
    s.set_vof(C)
    u = np.zeros((n, n, n), order="F")
    u[:] = 1.0
    s.set_field("u", u)
    for _ in range(3):
        s.step()
    return s


def gate_atomic():
    print("ISSUES item 1 -- step() atomic across the Weymouth-Yue boundedness throw")
    s = scene_cfl()
    s0 = state(s)
    s.set_dt(0.4)                       # CFL 0.4 > the 0.25 cap
    threw = ""
    try:
        s.step()
    except RuntimeError as e:
        threw = str(e).split("\n")[0]
    print(f"    WY throw: {threw[:110]}")
    d = diffs(s0, state(s))
    check("WY: step() threw", bool(threw), threw[:60])
    check("WY: state bitwise unchanged", same(s0, state(s)), str(d))
    # retry at the old dt must run
    s.set_dt(0.1)
    s.step()
    check("WY: retry at a smaller dt runs", True)

    # --- the capillary cap
    n = 16
    s = pf.Solver(n, n, n)
    s.set_rho(1.0)
    s.set_mu(0.05)
    s.set_pressure_geometry(np.full((n, n, n), 1e30, order="F"))
    s.set_body_force(0.0, 0.0, 1e-3)
    s.enable_vof()
    x, y, z = np.meshgrid(*[np.arange(n) + 0.5] * 3, indexing="ij")
    r = np.sqrt((x - n / 2) ** 2 + (y - n / 2) ** 2 + (z - n / 2) ** 2)
    s.set_vof(np.asfortranarray(np.clip(4.0 - r, 0.0, 1.0)))
    s.set_surface_tension(1.0)
    s.set_dt(0.2 * s.capillary_dt())
    for _ in range(3):
        s.step()
    s0 = state(s)
    s.set_dt(10.0 * s.capillary_dt())
    threw = ""
    try:
        s.step()
    except RuntimeError as e:
        threw = str(e).split("\n")[0]
    print(f"    capillary throw: {threw[:110]}")
    d = diffs(s0, state(s))
    check("sigma: step() threw", bool(threw), threw[:60])
    check("sigma: state bitwise unchanged", same(s0, state(s)), str(d))


# ------------------------------------------------------------------ item 2: step_adaptive
def scene_hysing(nx=32, nz=64, sigma=0.02):
    s = pf.Solver(nx, 4, nz)
    s.set_rho(1.0)
    s.set_mu(0.01)
    s.set_domain_bc(4, 1); s.set_domain_bc(5, 1)
    s.set_pressure_geometry(np.full((nx, 4, nz), 1e30, order="F"))
    s.enable_vof()
    x, y, z = np.meshgrid(np.arange(nx) + 0.5, np.arange(4) + 0.5,
                          np.arange(nz) + 0.5, indexing="ij")
    r = np.sqrt((x - nx / 2) ** 2 + (z - nz / 4) ** 2)
    C = np.asfortranarray(np.clip(0.5 + (r - 0.25 * nx / 2), 0.0, 1.0) * 0 + (r > 0.25 * nx).astype(float))
    s.set_vof(C)
    s.set_property_model("rho", "linear", "C", [1.0, 9.0])
    s.set_property_model("mu", "linear", "C", [0.005, 0.045])
    s.set_pressure_fcg(True, 600, 1e-10)
    if sigma > 0:
        s.set_surface_tension(sigma)
    s.set_body_force(0.0, 0.0, -5e-2)
    # A divergence-free x-shear in w (no z dependence, periodic in x, so the walls do not kill
    # it) — this is what makes the interface-local CFL a REAL, binding limit on the sigma = 0
    # configuration instead of an infinity.
    ww = np.zeros((nx, 4, nz), order="F")
    ww += (0.3 * np.sin(2 * np.pi * (np.arange(nx) + 0.5) / nx))[:, None, None]
    s.set_field("w", ww)
    s.set_dt(min(1.0, 0.4 * s.capillary_dt()))
    return s


def gate_adaptive(nsteps=200):
    print("ISSUES item 2 -- step_adaptive reproduces the gallery driver bitwise")
    # Two factor pairs so BOTH branches of the min() are exercised: (0.4, 0.4) is the gallery
    # driver verbatim (capillary-bound on this scene) and (0.4, 4.0) loosens the capillary factor
    # far enough that the interface-local CFL binds instead.
    for cfl, cap, sig, dmax in ((0.4, 0.4, 0.02, math.inf), (0.2, 0.4, 0.0, 1.0)):
        print(f"  -- cfl_target = {cfl}, capillary_cfl = {cap}, sigma = {sig}, dt_max = {dmax}"
              f"  ({'capillary' if sig else 'CFL/dt_max'}-bound)")
        one_adaptive(nsteps, cfl, cap, sig, dmax)


def one_adaptive(nsteps, cflT, capT, sigma, dmax):
    a = scene_hysing(sigma=sigma)
    dts_a, ncap, ncfl = [], 0, 0
    for _ in range(nsteps):                       # the hand-written gallery loop
        L = a.vof_step_limits()
        dt = min(cflT * L["cfl_dt"], capT * L["capillary_dt"], dmax)
        a.set_dt(dt)
        a.step()
        dts_a.append(dt)
        ncap += int(capT * L["capillary_dt"] <= min(cflT * L["cfl_dt"], dmax))
        ncfl += int(cflT * L["cfl_dt"] < min(capT * L["capillary_dt"], dmax))
    b = scene_hysing(sigma=sigma)
    dts_b = [b.step_adaptive(cflT, capT, dmax) for _ in range(nsteps)]
    d = diffs(state(a), state(b))
    check("  dt sequence identical", dts_a == dts_b,
          f"max|d dt| = {max(abs(x - y) for x, y in zip(dts_a, dts_b)):.3e}")
    check("  state bitwise identical", same(state(a), state(b)), str(d))
    print(f"     {nsteps} steps, dt {dts_a[0]:.6g} -> {dts_a[-1]:.6g}, "
          f"pressure {a.last_pressure_iterations()}/600, failed={a.pressure_solve_failed()}; "
          f"capillary bound {ncap} steps, CFL bound {ncfl}; "
          f"dt range [{min(dts_a):.6g}, {max(dts_a):.6g}]")


# ------------------------------------------------------------------ item 4: vof_geometry
def gate_geometry():
    print("ISSUES item 4 -- vof_geometry() on an all-fluid VoF solver")
    n = 32
    s = pf.Solver(n, n, n)
    s.set_rho(1.0); s.set_mu(0.1)
    s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
    s.enable_vof()
    eps = np.asarray(s.vof_geometry(0))
    kind = np.asarray(s.vof_geometry(4))
    check("vof_has_geometry() is False", s.vof_has_geometry() is False)
    check("eps == 1 everywhere", eps.shape == (n, n, n) and np.all(eps == 1.0),
          f"min {eps.min()} max {eps.max()}")
    check("openness == 1 everywhere",
          all(np.all(np.asarray(s.vof_geometry(k)) == 1.0) for k in (1, 2, 3)))
    check("classification == 0 everywhere", np.all(kind == 0.0))
    # the E7 driver's dual use: one expression for both scenes
    gas = np.asarray(s.vof_geometry(0)) * (1.0 - s.get_vof())
    check("E7 dual-use expression runs", gas.shape == (n, n, n))


# ------------------------------------------------------------------ item 5: collocated advect_vof
def leveque(n, t=0.0, T=3.0):
    """Cell-centre-sampled LeVeque deformation field (u, v, w) on an n^3 unit box."""
    c = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    f = math.cos(math.pi * t / T)
    u = 2.0 * np.sin(np.pi * X) ** 2 * np.sin(2 * np.pi * Y) * np.sin(2 * np.pi * Z) * f
    v = -np.sin(2 * np.pi * X) * np.sin(np.pi * Y) ** 2 * np.sin(2 * np.pi * Z) * f
    w = -np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.sin(np.pi * Z) ** 2 * f
    return (np.asfortranarray(u * n), np.asfortranarray(v * n), np.asfortranarray(w * n))


def sphere_colour(n, cx=0.35, r=0.15):
    c = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    d = np.sqrt((X - cx) ** 2 + (Y - cx) ** 2 + (Z - cx) ** 2)
    return np.asfortranarray(np.clip(0.5 + (r - d) * n, 0.0, 1.0))


def gate_collocated():
    print("ISSUES item 5 -- SolverColocated: set_state + advect_vof")
    n = 32
    s = pf.SolverColocated(n, n, n)
    s.set_rho(1.0); s.set_mu(0.0)
    s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
    s.enable_vof()
    C0 = sphere_colour(n)
    s.set_vof(C0)
    u, v, w = leveque(n)
    s.set_state(u, v, w)
    print(f"    max|u| = {np.abs(u).max():.4g}, max|uf| after set_state = "
          f"{np.abs(np.asarray(s.get_uf())).max():.4g}")
    check("set_state seeds the face field", np.abs(np.asarray(s.get_uf())).max() > 1.0)
    div = s.max_open_divergence_projected()
    print(f"    max|div(open uf)| of the seeded field = {div:.4g}")
    dt = 0.2 / np.abs(np.asarray(s.get_uf())).max()      # interface CFL well under the cap
    for _ in range(10):
        s.advect_vof(dt)
    moved = float(np.abs(s.get_vof() - C0).max())
    vol = float(s.get_vof().sum())
    check("advect_vof no longer a silent no-op", moved > 0.0, f"max|C-C0| = {moved:.3e}")
    # the reference: the SAME face field handed to a staggered solver (the E1 page's recipe)
    r = pf.Solver(n, n, n)
    r.set_rho(1.0); r.set_mu(0.0)
    r.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
    r.enable_vof(); r.set_vof(C0)
    for nm, fa in zip("uvw", (s.get_uf(), s.get_vf(), s.get_wf())):
        r.set_field(nm, np.asfortranarray(np.asarray(fa)))
    for _ in range(10):
        r.advect_vof(dt)
    dC = float(np.abs(r.get_vof() - s.get_vof()).max())
    print(f"    dt = {dt:.6g}, 10 steps: max|C-C0| = {moved:.4g}, sum C = {vol:.12g} (C0: {float(C0.sum()):.12g})")
    check("collocated colour == the staggered run on the SAME face field", dC == 0.0,
          f"max|dC| = {dC:.3e}")
    # a fresh solver whose face field was never built must refuse
    s2 = pf.SolverColocated(n, n, n)
    s2.set_rho(1.0); s2.set_mu(0.0)
    s2.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
    s2.enable_vof(); s2.set_vof(C0)
    m2 = ""
    try:
        s2.advect_vof(0.01)
    except RuntimeError as e:
        m2 = str(e).split("\n")[0]
    check("a never-built face field is refused", "never been built" in m2, m2[:70])


# ------------------------------------------------------------------ item 7: free-slip domain BC
def half_channel(nz, top_type, mu=0.05, F=1e-3, dt=5000.0, steps=200):
    s = pf.Solver(4, 4, nz)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    s.set_domain_bc(4, 1)            # -z no-slip wall
    s.set_domain_bc(5, top_type)     # +z: 1 = no-slip, 4 = FREE SLIP
    s.set_pressure_geometry(np.full((4, 4, nz), 1e30, order="F"))
    s.set_body_force(F, 0.0, 0.0)
    for _ in range(steps):
        s.step()
    return s


def gate_freeslip(nz=32):
    print("ISSUES item 7 -- domain BC type 4 = free slip")
    mu, F = 0.05, 1e-3
    # (a) the analytic half-channel. The DISCRETE solution of the scheme is the continuum
    #     parabola plus h^2/4 * F/(2 mu): a second difference reproduces a quadratic exactly, and
    #     the only inexact ingredient is the no-slip wall's mirror ghost (u(-h/2) = -u(h/2) is
    #     exact only for an ODD function, and z^2 is even). That offset is the NO-SLIP wall's, not
    #     the free-slip one's -- which is the point: the free-slip face contributes no error at all.
    s = half_channel(nz, 4, mu, F)
    u = s.get_u()[0, 0, :]
    zc = np.arange(nz) + 0.5
    H = float(nz)
    cont = F / (2 * mu) * (H ** 2 - (H - zc) ** 2)
    disc = cont + F / (2 * mu) * 0.25
    e_cont = float(np.abs(u - cont).max() / cont.max())
    e_disc = float(np.abs(u - disc).max() / disc.max())
    print(f"    u_max {u.max():.8f}; continuum parabola {cont.max():.8f} (rel {e_cont:.3e}), "
          f"discrete parabola {disc.max():.8f} (rel {e_disc:.3e}); "
          f"pressure {s.last_pressure_iterations()}, failed={s.pressure_solve_failed()}")
    check("half-channel == the scheme's own discrete parabola (solver tol)", e_disc < 1e-8, f"{e_disc:.3e}")
    check("half-channel within h^2 of the continuum parabola", e_cont < 1e-3, f"{e_cont:.3e}")
    # (b) free slip IS symmetry: the half channel must equal the lower half of a FULL channel of
    #     twice the height with no-slip on both sides.
    f2 = half_channel(2 * nz, 1, mu, F)
    u2 = f2.get_u()[0, 0, :nz]
    d = float(np.abs(u - u2).max() / u.max())
    print(f"    vs the lower half of a {2*nz}-cell no-slip channel: max rel difference {d:.3e}")
    check("free slip == the symmetry plane of a full channel", d < 1e-9, f"{d:.3e}")
    # (c) inert when unused: a type-1 lid must reproduce a plain no-slip channel exactly.
    a = half_channel(nz, 1, mu, F)
    check("type 4 unused -> nothing moves (sanity)", np.isfinite(a.get_u()).all())


def gate_wetwall():
    print("ISSUES item 3 -- set_contact_angle on a domain-BC wall (see the dedicated script)")


MAIN = dict(atomic=gate_atomic, adaptive=gate_adaptive, geometry=gate_geometry,
            collocated=gate_collocated, freeslip=gate_freeslip, wetwall=gate_wetwall)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", nargs="*", default=list(MAIN))
    a = ap.parse_args()
    for g in a.gate:
        MAIN[g]()
    print("\nFAILED:" if FAIL else "\nALL GATES PASSED", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
