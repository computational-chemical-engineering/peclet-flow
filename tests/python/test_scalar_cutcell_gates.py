#!/usr/bin/env python
"""Cut-cell scalar transport: the accuracy gates of WO-3 (doc/scalar_ibm_design.md §11).

Single phase, diffusion only, the level-0 (2 + 2 red-black Gauss-Seidel) preconditioner of WO-3.
Every case is in PHYSICAL units (an extent), so the unit conversions of §1.2 are on the path.

  api  the opt-in, the refusals and the name checks (cheap).
  g1   Dirichlet sphere, steady: Nu -> 2 (order >= 1.7), field L1 / Linf orders (>= 1.8 / 1.5),
       the steady budget identity, R0 = 100 %; plus the anisotropic h' = (1, 1, 2) ladder.
  g2   concentric shells, steady: Neumann flux q on the inner sphere, an immersed Dirichlet outer
       sphere (a two-instance scene): area-mean wall value c_Gamma (order >= 1.7), field L1.
  g3a  Robin external sphere, steady, Da = kR/D in {0.1, 1, 10, 100}: Sh = 2 Da/(1 + Da).
  g3b  Robin interior sphere, transient, Bi in {0.1, 1, 10, 100, inf}: the backward-Euler
       late-time ratio of the total mass gives the spatial eigenvalue mu = D lam^2/R^2,
       lam cot lam = 1 - Bi; the budget identity every step.
  g7   pipe (an SDF along z, nz = 4 periodic): Dirichlet decay j01^2 and Neumann dipole decay
       j'11^2 (BE ratio), and the Graetz number Nu_T = 3.656793 by inverse iteration with
       solve_scalar_steady (G7a, G7b).

Ruling D-WO3-1: the STEADY gates (g1, g2, g3a, g7) run on the TWO COARSEST rungs with maxit 3000
(the full ladder reruns with ScalarMG in WO-4); g3b runs its full ladder. "Order" is between
successive rungs of the RMS error over 3 random grid offsets. Iteration counts are printed.

Run: OMP_NUM_THREADS=4 PYTHONPATH=<build> python tests/python/test_scalar_cutcell_gates.py <gate>
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import math
import sys
import time

import numpy as np

try:
    import peclet.flow as pf
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

FACES = ("-x", "+x", "-y", "+y", "-z", "+z")
OFFSETS = np.random.default_rng(20261003).uniform(-0.5, 0.5, size=(3, 3))
MAXIT_STEADY = 3000
failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def rms(v):
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v * v)))


def order(ec, ef):
    return math.log2(ec / ef) if ec > 0 and ef > 0 else float("inf")


def grid(s):
    x, y, z = s.cell_centers()
    return np.meshgrid(x, y, z, indexing="ij")


def face_points(s, face):
    """Face-centre coordinates (X, Y, Z) of a domain face, shaped (N_t1, N_t2)."""
    x, y, z = s.cell_centers()
    lo = [x[0] - 0.5 * (x[1] - x[0]), y[0] - 0.5 * (y[1] - y[0]), z[0] - 0.5 * (z[1] - z[0])]
    hi = [x[-1] + 0.5 * (x[1] - x[0]), y[-1] + 0.5 * (y[1] - y[0]), z[-1] + 0.5 * (z[1] - z[0])]
    a, side = FACES.index(face) // 2, FACES.index(face) % 2
    c = [x, y, z]
    t = [k for k in range(3) if k != a]
    T1, T2 = np.meshgrid(c[t[0]], c[t[1]], indexing="ij")
    P = [None, None, None]
    P[a] = np.full(T1.shape, hi[a] if side else lo[a])
    P[t[0]], P[t[1]] = T1, T2
    return P


def walled(cells, extent, origin=(0.0, 0.0, 0.0), periodic_z=False):
    s = pf.Solver(cells, extent=extent, origin=origin)
    for f in FACES:
        if not (periodic_z and f in ("-z", "+z")):
            s.set_domain_bc(f, "wall")
    s.set_rho(1.0)
    s.set_mu(1.0)
    return s


def add_cc(s, D, steady_maxit=True, box="neumann", periodic_z=False):
    s.add_scalar("c", diffusivity=D, cutcell=True)
    for f in FACES:
        if periodic_z and f in ("-z", "+z"):
            continue
        s.set_scalar_bc("c", f, box)
    if steady_maxit:
        s.diagnostics.set_scalar_max_iterations("c", MAXIT_STEADY)


def field_errors(s, exact):
    geo = s.diagnostics.scalar_geometry("c")
    unk = geo["unknown"] > 0.5
    w = geo["kappa"][unk]
    e = (s.get_field("c") - exact)[unk]
    return float(np.sum(w * np.abs(e)) / np.sum(w)), float(np.max(np.abs(e)))


def solve_info(s):
    c = s.diagnostics.scalar_census("c")
    return c["krylov_iterations"], c["krylov_converged"], c["probe_rungs"]["fluid"]


# ---------------------------------------------------------------------------------------- G1 ----
def g1_case(Rh, off, aniso=False):
    R, D = 1.0, 0.7
    L = 4.0 * R
    h = R / Rh
    cells = (int(round(L / h)), int(round(L / h)), int(round(L / (2 * h if aniso else h))))
    s = walled(cells, (L, L, L))
    hz = L / cells[2]
    c0 = np.array([0.5 * L + off[0] * h, 0.5 * L + off[1] * h, 0.5 * L + off[2] * hz])
    X, Y, Z = grid(s)
    r = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2)
    s.set_solid(np.asfortranarray(r - R))
    add_cc(s, D, box="neumann")
    for f in FACES:
        P = face_points(s, f)
        rf = np.sqrt((P[0] - c0[0]) ** 2 + (P[1] - c0[1]) ** 2 + (P[2] - c0[2]) ** 2)
        s.set_scalar_bc("c", f, "dirichlet", R / rf)
    s.set_scalar_wall("c", "dirichlet", 1.0)
    s.solve_scalar_steady("c")
    flux = float(s.scalar_wall_flux("c")[0])
    nu = flux / (2.0 * math.pi * R * D)
    l1, linf = field_errors(s, R / r)
    b = s.diagnostics.scalar_budget("c")
    it, conv, rungs = solve_info(s)
    return dict(err=nu / 2.0 - 1.0, l1=l1, linf=linf, it=it, conv=conv, rungs=rungs, b=b, flux=flux)


def gate_g1():
    print("G1 Dirichlet sphere (steady), R/h in {8, 16}, box 4R")
    res = {}
    for Rh in (8, 16):
        rows = [g1_case(Rh, off) for off in OFFSETS]
        res[Rh] = rows
        idr = max(abs(r["b"]["identity_error"]) / abs(r["b"]["wall_in"]) for r in rows)
        dfr = max(abs(r["b"]["defect"]) / abs(r["b"]["wall_in"]) for r in rows)
        print(f"  R/h={Rh:3d}  Nu/2-1 = {[f'{r['err']:+.3e}' for r in rows]}  L1 {rms([r['l1'] for r in rows]):.3e}"
              f"  Linf {rms([r['linf'] for r in rows]):.3e}  iters {[r['it'] for r in rows]}"
              f"  identity/|wall| <= {idr:.1e}  defect/|wall| <= {dfr:.1e}")
        for r in rows:
            check(r["conv"], f"R/h={Rh}: the solve converged ({r['it']} iterations)")
            check(r["rungs"][1] + r["rungs"][2] + r["rungs"][3] == 0, f"R/h={Rh}: R0 = 100 % ({r['rungs']})")
            b = r["b"]
            ident = abs(b["wall_in"] + b["boundary_in"] - b["defect"])
            check(ident <= 1e-12 * abs(b["wall_in"]),
                  f"R/h={Rh}: |wall - box - defect| = {ident:.2e} <= 1e-12 |wall| ({abs(b['wall_in']):.3e})")
            check(abs(b["defect"]) <= 1e-6 * abs(b["wall_in"]),
                  f"R/h={Rh}: |defect| = {abs(b['defect']):.2e} <= 1e-6 |wall|")
            check(abs(b["identity_error"]) <= 1e-12 * abs(b["wall_in"]),
                  f"R/h={Rh}: identity_error {b['identity_error']:.2e}")
    e = {k: rms([r["err"] for r in v]) for k, v in res.items()}
    l1 = {k: rms([r["l1"] for r in v]) for k, v in res.items()}
    li = {k: rms([r["linf"] for r in v]) for k, v in res.items()}
    check(order(e[8], e[16]) >= 1.7, f"Nu order {order(e[8], e[16]):.2f} >= 1.7 (rms {e[8]:.3e} -> {e[16]:.3e})")
    check(order(l1[8], l1[16]) >= 1.8, f"field L1 order {order(l1[8], l1[16]):.2f} >= 1.8")
    check(order(li[8], li[16]) >= 1.5, f"field Linf order {order(li[8], li[16]):.2f} >= 1.5")
    print("G1 anisotropic h' = (1, 1, 2), R/h_min in {16, 32}")
    ea = {}
    for Rh in (16, 32):
        rows = [g1_case(Rh, off, aniso=True) for off in OFFSETS]
        ea[Rh] = rms([r["err"] for r in rows])
        print(f"  R/h={Rh:3d}  Nu/2-1 = {[f'{r['err']:+.3e}' for r in rows]}  iters {[r['it'] for r in rows]}"
              f"  rungs {[r['rungs'] for r in rows]}")
        for r in rows:
            check(r["conv"], f"aniso R/h={Rh}: the solve converged ({r['it']} iterations)")
            # ruling D-WO2-2: on h' = (1, 1, 2) R0 >= 99.9 % of the facets and no R2
            check(r["rungs"][0] >= 0.999 * sum(r["rungs"]) and r["rungs"][3] == 0,
                  f"aniso R/h={Rh}: R0 >= 99.9 %, no R2 ({r['rungs']})")
    check(order(ea[16], ea[32]) >= 1.7, f"aniso Nu order {order(ea[16], ea[32]):.2f} >= 1.7 (rms {ea[16]:.3e} -> {ea[32]:.3e})")


# ---------------------------------------------------------------------------------------- G2 ----
def two_body_scene(c0, Ri, Ro):
    """Instance 0: a sphere Ri. Instance 1: a huge box minus a sphere Ro (solid outside Ro)."""
    ni = np.array([1, -1, -1,  3, -1, -1,  1, -1, -1,  34, 1, 2], dtype=np.int32)
    nr = np.zeros(4 * 16)
    for k in range(4):
        nr[16 * k + 14] = 1.0  # rotation w
        nr[16 * k + 15] = 1.0  # scale
    nr[0] = Ri
    nr[16 + 0] = nr[16 + 1] = nr[16 + 2] = 1e3
    nr[32 + 0] = Ro
    ii = np.array([0, -1, 3, -1], dtype=np.int32)
    ir = np.zeros(2 * 18)
    for k in range(2):
        ir[18 * k:18 * k + 3] = c0
        ir[18 * k + 6] = 1.0
        ir[18 * k + 7] = 1.0
    return ni, nr, ii, ir


def g2_case(Rih, off):
    Ri, D, q, co = 1.0, 0.8, 0.3, 0.0
    Ro = 2.5 * Ri
    h = Ri / Rih
    n = int(math.ceil(2.0 * Ro / h)) + 6
    L = n * h
    s = walled((n, n, n), (L, L, L))
    c0 = np.array([0.5 * L, 0.5 * L, 0.5 * L]) + np.asarray(off) * h
    s.set_scene(*two_body_scene(c0, Ri, Ro), periodic=False)
    s.set_solid_from_scene(cutcell_pressure=False)
    add_cc(s, D, box="neumann")
    s.set_scalar_wall("c", "neumann", q, instance=0)
    s.set_scalar_wall("c", "dirichlet", co, instance=1)
    s.solve_scalar_steady("c")
    fac = s.diagnostics.scalar_facets("c")
    inner = fac["instance"] == 0
    cg = float(np.sum(fac["area"][inner] * fac["wall_value"][inner]) / np.sum(fac["area"][inner]))
    exact_g = co + (q * Ri / D) * (1.0 - Ri / Ro)
    X, Y, Z = grid(s)
    r = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2)
    l1, linf = field_errors(s, co + (q * Ri * Ri / D) * (1.0 / r - 1.0 / Ro))
    flux = s.scalar_wall_flux("c")
    it, conv, rungs = solve_info(s)
    # the Neumann flux is imposed exactly on the DISCRETE area (q times the facet areas)
    return dict(err=cg / exact_g - 1.0, l1=l1, linf=linf, it=it, conv=conv, rungs=rungs,
                fin=float(flux[0]) / (q * float(np.sum(fac["area"][inner]))) - 1.0)


def gate_g2():
    print("G2 concentric shells (steady), Neumann q inner, Dirichlet outer Ro = 2.5 Ri, Ri/h in {6, 12}")
    e, l1 = {}, {}
    for Rih in (6, 12):
        rows = [g2_case(Rih, off) for off in OFFSETS]
        e[Rih] = rms([r["err"] for r in rows])
        l1[Rih] = rms([r["l1"] for r in rows])
        print(f"  Ri/h={Rih:3d}  c_Gamma rel err {[f'{r['err']:+.3e}' for r in rows]}  L1 {l1[Rih]:.3e}"
              f"  inner flux/(q area)-1 {[f'{r['fin']:+.1e}' for r in rows]}  iters {[r['it'] for r in rows]}")
        for r in rows:
            check(r["conv"], f"Ri/h={Rih}: converged ({r['it']} iterations)")
            check(r["rungs"][1] + r["rungs"][2] + r["rungs"][3] == 0, f"Ri/h={Rih}: R0 = 100 % ({r['rungs']})")
            check(abs(r["fin"]) <= 1e-12, f"Ri/h={Rih}: inner flux = q x discrete area ({r['fin']:+.1e})")
    check(order(e[6], e[12]) >= 1.7, f"c_Gamma order {order(e[6], e[12]):.2f} >= 1.7 (rms {e[6]:.3e} -> {e[12]:.3e})")
    check(order(l1[6], l1[12]) >= 1.8, f"field L1 order {order(l1[6], l1[12]):.2f} >= 1.8")


# --------------------------------------------------------------------------------------- G3a ----
def g3a_case(Rh, off, Da):
    R, D = 1.0, 0.6
    L = 4.0 * R
    h = R / Rh
    n = int(round(L / h))
    s = walled((n, n, n), (L, L, L))
    s.set_dt(0.37)  # pins tRef != 1, so the Robin k goes through speedToInt
    c0 = np.array([0.5 * L, 0.5 * L, 0.5 * L]) + np.asarray(off) * h
    X, Y, Z = grid(s)
    r = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2)
    s.set_solid(np.asfortranarray(r - R))
    add_cc(s, D, box="neumann")
    A = Da / (1.0 + Da)
    for f in FACES:
        P = face_points(s, f)
        rf = np.sqrt((P[0] - c0[0]) ** 2 + (P[1] - c0[1]) ** 2 + (P[2] - c0[2]) ** 2)
        s.set_scalar_bc("c", f, "dirichlet", 1.0 - A * R / rf)
    s.set_scalar_wall("c", "robin", 0.0, coefficient=Da * D / R)
    s.solve_scalar_steady("c")
    sh = -float(s.scalar_wall_flux("c")[0]) / (2.0 * math.pi * R * D)
    it, conv, rungs = solve_info(s)
    return dict(err=sh / (2.0 * A) - 1.0, it=it, conv=conv, rungs=rungs)


def gate_g3a():
    print("G3a Robin external sphere (steady), R/h in {8, 16}")
    for Da in (0.1, 1.0, 10.0, 100.0):
        e = {}
        for Rh in (8, 16):
            rows = [g3a_case(Rh, off, Da) for off in OFFSETS]
            e[Rh] = rms([r["err"] for r in rows])
            print(f"  Da={Da:6.1f} R/h={Rh:3d}  Sh rel err {[f'{r['err']:+.3e}' for r in rows]}  iters {[r['it'] for r in rows]}")
            for r in rows:
                check(r["conv"], f"Da={Da} R/h={Rh}: converged ({r['it']} iterations)")
                check(r["rungs"][1] + r["rungs"][2] + r["rungs"][3] == 0, f"Da={Da} R/h={Rh}: R0 = 100 %")
        check(order(e[8], e[16]) >= 1.7, f"Da={Da}: Sh order {order(e[8], e[16]):.2f} >= 1.7 (rms {e[8]:.3e} -> {e[16]:.3e})")


# --------------------------------------------------------------------------------------- G3b ----
def robin_root(Bi):
    if math.isinf(Bi):
        return math.pi
    f = lambda l: l * math.cos(l) - (1.0 - Bi) * math.sin(l)  # lam cot lam = 1 - Bi
    a, b = 1e-9, math.pi - 1e-12
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


def g3b_case(Rh, off, Bi, nsteps=30):
    R, D = 1.0, 0.9
    h = R / Rh
    n = 2 * Rh + 6
    L = n * h
    s = walled((n, n, n), (L, L, L))
    lam = robin_root(Bi)
    mu = D * lam * lam / (R * R)
    dt = 1.0 / mu
    s.set_dt(dt)
    c0 = np.array([0.5 * L, 0.5 * L, 0.5 * L]) + np.asarray(off) * h
    X, Y, Z = grid(s)
    r = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2)
    s.set_solid(np.asfortranarray(R - r))  # the fluid is INSIDE the sphere
    add_cc(s, D, steady_maxit=False, box="neumann")
    s.diagnostics.set_scalar_max_iterations("c", MAXIT_STEADY)
    if math.isinf(Bi):
        s.set_scalar_wall("c", "dirichlet", 0.0)
    else:
        s.set_scalar_wall("c", "robin", 0.0, coefficient=Bi * D / R)
    x = lam * r / R
    s.set_field("c", np.asfortranarray(np.where(x > 1e-12, np.sin(x) / np.where(x > 1e-12, x, 1.0), 1.0)))
    masses, idmax, its = [], 0.0, []
    for k in range(nsteps):
        s.advance_scalars()
        b = s.diagnostics.scalar_budget("c")
        masses.append(b["mass"])
        scale = max(abs(b["d_mass"]), abs(b["mass"]) * 1e-300)
        idmax = max(idmax, abs(b["identity_error"]) / scale)
        it, conv, rungs = solve_info(s)
        its.append(it)
        if not conv:
            return dict(err=float("nan"), idrel=idmax, its=its, conv=False, rungs=rungs)
    mu_h = (masses[-2] / masses[-1] - 1.0) / dt
    return dict(err=mu_h / mu - 1.0, idrel=idmax, its=its, conv=True, rungs=rungs)


def gate_g3b():
    print("G3b Robin interior sphere (transient, BE late-time ratio, dt mu = 1, 30 steps), R/h in {8, 16, 32}")
    for Bi in (0.1, 1.0, 10.0, 100.0, math.inf):
        e = {}
        for Rh in (8, 16, 32):
            t0 = time.time()
            rows = [g3b_case(Rh, off, Bi) for off in OFFSETS]
            e[Rh] = rms([r["err"] for r in rows])
            print(f"  Bi={Bi:6.1f} R/h={Rh:3d}  mu rel err {[f'{r['err']:+.3e}' for r in rows]}"
                  f"  identity/|dM| <= {max(r['idrel'] for r in rows):.1e}"
                  f"  iters/step {min(min(r['its']) for r in rows)}..{max(max(r['its']) for r in rows)}"
                  f"  ({time.time() - t0:.0f} s)")
            for r in rows:
                check(r["conv"], f"Bi={Bi} R/h={Rh}: every step converged")
                check(r["idrel"] <= 1e-13, f"Bi={Bi} R/h={Rh}: budget identity {r['idrel']:.1e} <= 1e-13 |d_mass|")
                check(r["rungs"][1] + r["rungs"][2] + r["rungs"][3] == 0, f"Bi={Bi} R/h={Rh}: R0 = 100 %")
        o1, o2 = order(e[8], e[16]), order(e[16], e[32])
        check(o1 >= 1.7 and o2 >= 1.7, f"Bi={Bi}: orders {o1:.2f}, {o2:.2f} >= 1.7")
        check(e[32] <= 2e-3, f"Bi={Bi}: |err| {e[32]:.2e} <= 2e-3 at R/h = 32 (prov.)")


# ---------------------------------------------------------------------------------------- G7 ----
J01_SQ = 5.783185962946784
J11P_SQ = 3.389957
GRAETZ = 3.656793


def pipe(Rh, off, D):
    R = 1.0
    h = R / Rh
    n = 2 * Rh + 6
    L = n * h
    s = walled((n, n, 4), (L, L, 4 * h), periodic_z=True)
    c0 = np.array([0.5 * L, 0.5 * L]) + np.asarray(off[:2]) * h
    X, Y, Z = grid(s)
    rr = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2)
    s.set_solid(np.asfortranarray(R - rr))
    add_cc(s, D, box="neumann", periodic_z=True)
    return s, X - c0[0], Y - c0[1], rr


def j0(x):
    # series, |x| <= 2.5 here
    t, s, k = 1.0, 1.0, 0
    while abs(t) > 1e-17:
        k += 1
        t *= -(x * x / 4.0) / (k * k)
        s += t
    return s


def j1(x):
    t = x / 2.0
    s, k = t, 0
    while abs(t) > 1e-17:
        k += 1
        t *= -(x * x / 4.0) / (k * (k + 1))
        s += t
    return s


def g7a_case(Rh, off, neumann, nsteps=30):
    D = 0.5
    s, dx, dy, rr = pipe(Rh, off, D)
    eig = J11P_SQ if neumann else J01_SQ
    mu = D * eig
    dt = 1.0 / mu
    s.set_dt(dt)
    s.set_scalar_wall("c", "neumann" if neumann else "dirichlet", 0.0)
    geo = s.diagnostics.scalar_geometry("c")
    kap = geo["kappa"] * (geo["unknown"] > 0.5)
    if neumann:
        jv = np.vectorize(j1)(math.sqrt(J11P_SQ) * rr)
        c = jv * np.where(rr > 0, dx / np.where(rr > 0, rr, 1.0), 0.0)
        c -= np.sum(kap * c) / np.sum(kap)
        mom = dx
    else:
        c = np.vectorize(j0)(math.sqrt(J01_SQ) * rr)
        mom = np.ones_like(dx)
    s.set_field("c", np.asfortranarray(c))
    vals, its = [], []
    for k in range(nsteps):
        s.advance_scalars()
        vals.append(float(np.sum(kap * mom * s.get_field("c"))))
        it, conv, rungs = solve_info(s)
        its.append(it)
    mu_h = (vals[-2] / vals[-1] - 1.0) / dt
    return dict(err=mu_h / mu - 1.0, its=its, conv=conv, rungs=rungs)


def g7b_case(Rh, off, iters=25):
    D = 0.5
    s, dx, dy, rr = pipe(Rh, off, D)
    s.set_scalar_wall("c", "dirichlet", 0.0)
    uz = np.maximum(2.0 * (1.0 - rr * rr), 0.0)  # Poiseuille, mean U = 1, R = 1
    geo = s.diagnostics.scalar_geometry("c")
    kap = geo["kappa"] * (geo["unknown"] > 0.5)
    v = np.maximum(1.0 - rr * rr, 0.0)
    beta, its = 0.0, []
    for k in range(iters):
        s.set_scalar_source("c", np.asfortranarray(uz * v))
        s.set_field("c", np.asfortranarray(v))
        s.solve_scalar_steady("c")
        w = s.get_field("c")
        it, conv, rungs = solve_info(s)
        its.append(it)
        bnew = float(np.sum(kap * uz * v * v) / np.sum(kap * uz * v * w))
        nrm = math.sqrt(float(np.sum(kap * w * w)))
        v = w / nrm
        if k > 3 and abs(bnew - beta) <= 1e-12 * abs(bnew):
            beta = bnew
            break
        beta = bnew
    nu = beta * 1.0 * 1.0 / D  # Nu_T = beta U R^2 / D
    return dict(err=nu / GRAETZ - 1.0, its=its, conv=conv, rungs=rungs)


def gate_g7():
    print("G7a pipe decay (BE ratio, dt mu = 1, 30 steps), R/h in {16, 32}")
    for neumann in (False, True):
        e = {}
        for Rh in (16, 32):
            rows = [g7a_case(Rh, off, neumann) for off in OFFSETS]
            e[Rh] = rms([r["err"] for r in rows])
            print(f"  {'neumann  j11p^2' if neumann else 'dirichlet j01^2'} R/h={Rh:3d}  rel err "
                  f"{[f'{r['err']:+.3e}' for r in rows]}  iters/step {min(min(r['its']) for r in rows)}..{max(max(r['its']) for r in rows)}")
            for r in rows:
                check(r["conv"], f"R/h={Rh}: converged")
                check(r["rungs"][1] + r["rungs"][2] + r["rungs"][3] == 0, f"R/h={Rh}: R0 = 100 %")
        tag = "neumann" if neumann else "dirichlet"
        check(order(e[16], e[32]) >= 1.8, f"{tag}: order {order(e[16], e[32]):.2f} >= 1.8 (rms {e[16]:.3e} -> {e[32]:.3e})")
        check(e[32] <= 5e-4, f"{tag}: |err| {e[32]:.2e} <= 5e-4 at R/h = 32")
    print("G7b Graetz Nu_T (inverse iteration with solve_scalar_steady), R/h in {16, 32}")
    e = {}
    for Rh in (16, 32):
        rows = [g7b_case(Rh, off) for off in OFFSETS]
        e[Rh] = rms([r["err"] for r in rows])
        print(f"  R/h={Rh:3d}  Nu_T rel err {[f'{r['err']:+.3e}' for r in rows]}  iters/solve "
              f"{min(min(r['its']) for r in rows)}..{max(max(r['its']) for r in rows)}")
        for r in rows:
            check(r["conv"], f"R/h={Rh}: converged")
    check(order(e[16], e[32]) >= 1.7, f"Graetz order {order(e[16], e[32]):.2f} >= 1.7 (rms {e[16]:.3e} -> {e[32]:.3e})")
    check(e[32] <= 1e-3, f"Graetz |err| {e[32]:.2e} <= 1e-3 at R/h = 32 (prov.)")


# --------------------------------------------------------------------------------------- API ----
def raises(exc, fn, msg):
    try:
        fn()
    except exc:
        check(True, msg)
        return
    except Exception as e:  # noqa: BLE001 - report the wrong type
        check(False, f"{msg} (raised {type(e).__name__}: {e})")
        return
    check(False, msg + " (did not raise)")


def gate_api():
    print("API: the opt-in, the refusals (§8.1) and the name checks (ruling D-WO3-2)")
    n = 12
    s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(0.1)
    X, Y, Z = grid(s)
    s.set_solid(np.asfortranarray(np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) - 0.2))
    raises(ValueError, lambda: s.add_scalar("bad", 1.0, cutcell=True, iters=5),
           "add_scalar(cutcell=True, iters=5) is a ValueError")
    s.add_scalar("legacy", 1.0)
    s.add_scalar("c", 1.0, cutcell=True)
    for nm, fn in (("scalar_geometry", lambda: s.diagnostics.scalar_geometry("legacy")),
                   ("scalar_census", lambda: s.diagnostics.scalar_census("legacy")),
                   ("scalar_budget", lambda: s.diagnostics.scalar_budget("legacy")),
                   ("set_scalar_wall", lambda: s.set_scalar_wall("legacy", "dirichlet", 1.0)),
                   ("set_scalar_source", lambda: s.set_scalar_source("legacy", 1.0)),
                   ("solve_scalar_steady", lambda: s.solve_scalar_steady("legacy"))):
        raises(ValueError, fn, f"{nm} on a legacy scalar is a ValueError")
    raises(ValueError, lambda: s.set_scalar_wall("c", "dirichlet", 1.0, instance=0),
           "set_scalar_wall(instance=0) without a scene is a ValueError")
    raises(ValueError, lambda: s.set_scalar_wall("c", "robin", 0.0, coefficient=-1.0),
           "a negative robin coefficient is a ValueError")
    raises(ValueError, lambda: s.set_scalar_tolerance("c", 0.0), "rtol = 0 is a ValueError")
    raises(ValueError, lambda: s.set_scalar_bc("c", "-x", "dirichlet", np.zeros((n, n + 1))),
           "a profile of the wrong shape is a ValueError")
    s.set_scalar_wall("c", "dirichlet", 1.0)
    s.advance_scalars()  # the periodic box: one step runs
    c = s.diagnostics.scalar_census("c")
    check(c["krylov_converged"] and c["mg_levels"] == 1, f"one step converged ({c['krylov_iterations']} iterations)")
    s.set_scalar_bc("c", "-x", "dirichlet", 0.5)  # the flow -x face is periodic
    raises(RuntimeError, s.advance_scalars, "a scalar Dirichlet face against a periodic flow face is refused")
    s.set_scalar_bc("c", "-x", "periodic")
    s.set_porous_continuity(True)
    raises(RuntimeError, s.advance_scalars, "porous continuity is refused")


GATES = {"api": gate_api, "g1": gate_g1, "g2": gate_g2, "g3a": gate_g3a, "g3b": gate_g3b, "g7": gate_g7}

if __name__ == "__main__":
    names = sys.argv[1:] or list(GATES)
    for nm in names:
        t0 = time.time()
        GATES[nm]()
        print(f"[{nm}: {time.time() - t0:.0f} s]")
    if failures:
        print(f"{len(failures)} FAILED")
        sys.exit(1)
    print("PASS")
