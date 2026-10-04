#!/usr/bin/env python
"""Cut-cell scalar transport: the accuracy and iteration gates of WO-3 to WO-6 (doc/scalar_ibm_design.md §11).

Single phase; diffusion, and advection by the projection's face flux (WO-5); BiCGStab preconditioned by one ScalarMG V-cycle (WO-4; level 0 alone
under the transient level rule). Every case is in PHYSICAL units (an extent), so the unit
conversions of §1.2 are on the path.

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
  g9   advection in an annulus (WO-5): solid-body rotation carrying a Gaussian blob one revolution,
       fou / koren at bulk Courant 0.5 / 0.9: per-step budget identity, conservation against the
       accounted defects, positivity (koren gated at bulk Courant <= 1/2, ruling D-WO5-1; its 0.9
       row is information), finiteness, slivers (min kappa < 1e-2).
  g9c  open domain faces (WO-5b, ruling D-WO5-3): a channel with a solid sphere, inflow / walls /
       outflow, the scalar carried by flow's own projection: the per-step budget, boundary
       advective and diffusive fluxes included, closes to <= 1e-13 of the mass change; constant
       preservation; boundedness; a steady row; the collocated grid is refused (open question).
  g9b  constant preservation under flow's own projection (Stokes through an SC array, c = 1):
       staggered and the collocated 'gauge-exact', 'plain', 'embed' pass; the ghost projection
       is refused (§13 Q13).
  gadv  §11 G-adv (WO-5c, design Amendment A2: the steady advective surrogate), Pe_h = the census
       max_cell_peclet 0.1 / 1 / 10 by rescaling the face field: (a) WO-5's C3 problem (periodic
       box 4R, a Dirichlet sphere with a source, a projected Stokes field) at R/h 16 and 32;
       (b) the closure on G9b's SC array (Neumann spheres, singular, the mean-gradient mode of
       WO-6 with G = e_x) and (b')
       the same with Dirichlet spheres, 32^3 and 64^3; (c) G9c's channel (open faces), Pe_h 1 and
       10 (set through D). (i) converged, (ii) iteration bounds (Q20: min(prov., 2x measured)), (iii) growth per
       doubling <= 1.7x, (v) the steady budget identity <= 1e-11 of the gross budget (D-WO5c-1). (iv) C4 is in the
       `scalar_mg` ctest, (vi) a one-time logged check.
  g8   Taylor-Aris (WO-6): G7's pipe, frozen face-averaged Poiseuille fluxes (D-WO6-1), insulating walls, the
       mean-gradient mode G = e_z, steady (advecting, singular): -<u'theta> -> Pe^2 D/48 at
       Pe = 10 (order >= 1.8, <= 1e-3 at R/h 32), scalar_mean_flux = its field sums to 1e-10;
       G-iter (<= 30, growth <= 5) on multigrid-friendly boxes (ruling D-WO4-1).
  g5b  the insulating periodic SC array (c = 0.3) in the mean-gradient mode (WO-6): k* =
       -J_x/(D G_x) from scalar_mean_flux, self-convergence order >= 1.7, k* below the
       Hashin-Shtrikman bound; G-iter on multigrid-friendly boxes; the no-solid identity k* = 1
       to 1e-12 (at rest, and under a uniform flow: the moving frame).
  giter  §11 G-iter, the rows not carried by g1/g2/g3a: transient at dt D/h^2 = 1 (<= 10 per warm
       step, ruling D-WO4-3 restating the provisional 8; <= 12 on the cold first step, ruling
       D-WO9-3, which moved the level-rule switch so that this row runs level 0 alone) and the singular
       steady problem on G5b's geometry (periodic simple-cubic array, c = 0.3, insulating + flux +
       source; <= 30, growth <= 5 per rung) on multigrid-friendly n.

  g13  contacts (WO-8): two spheres R in a box (6R, 4R, 4R), R/h 8 / 16 / 32, steady: Dirichlet
       spheres at a gap R/32 and in contact (overlap R/64), far field 0, and the contacting pair
       conjugate at Lam_s/Lam_f = 100 under a far-field gradient: converged, <= 50 iterations,
       census two-sided > 0 (where the gap is under a cell), R2 <= 1 % of the probes, total-flux
       self-convergence order >= 1.0; the 0.4 h thin plate: num_thin_solid > 0 and the §9
       warning printed.

Every gate runs its FULL resolution ladder (WO-4; ruling D-WO3-1 had deferred the steady ones).
"Order" is between successive rungs of the RMS error over 3 random grid offsets; every successive
pair is gated. G-iter on g1/g2/g3a (Da = 1): <= 20 BiCGStab iterations at every rung, growth <= 3
per doubling (the max over the offsets of each rung). Ruling Q-C (WO-4): G-iter is gated on
multigrid-friendly boxes (n rounded up to a multiple of 2^(rungs + 1) = 16); a box with few factors
of two stays gated for correctness only (G2's native box). Iteration counts are printed.

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


def add_cc(s, D, box="neumann", periodic_z=False):
    s.add_scalar("c", diffusivity=D, cutcell=True)
    for f in FACES:
        if periodic_z and f in ("-z", "+z"):
            continue
        s.set_scalar_bc("c", f, box)


def giter_check(tag, its):
    """§11 G-iter on a ladder: {rung: [iterations per offset]} -> <= 20 each, growth <= 3."""
    rungs = sorted(its)
    mx = [max(its[k]) for k in rungs]
    check(max(mx) <= 20, f"{tag}: G-iter <= 20 iterations at every rung ({mx})")
    gr = [b - a for a, b in zip(mx, mx[1:])]
    check(all(g <= 3 for g in gr), f"{tag}: G-iter growth <= 3 per doubling ({gr})")


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
    print("G1 Dirichlet sphere (steady), R/h in {8, 16, 32}, box 4R")
    res = {}
    for Rh in (8, 16, 32):
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
    for a, b in ((8, 16), (16, 32)):
        check(order(e[a], e[b]) >= 1.7, f"Nu order {a}->{b} {order(e[a], e[b]):.2f} >= 1.7 (rms {e[a]:.3e} -> {e[b]:.3e})")
        check(order(l1[a], l1[b]) >= 1.8, f"field L1 order {a}->{b} {order(l1[a], l1[b]):.2f} >= 1.8 ({l1[a]:.3e} -> {l1[b]:.3e})")
        check(order(li[a], li[b]) >= 1.5, f"field Linf order {a}->{b} {order(li[a], li[b]):.2f} >= 1.5 ({li[a]:.3e} -> {li[b]:.3e})")
    check(e[32] <= 2e-3, f"|Nu/2 - 1| {e[32]:.2e} <= 2e-3 at R/h = 32 (prov.)")
    giter_check("G1", {k: [r["it"] for r in v] for k, v in res.items()})
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


def mg_box(n, rungs=3):
    """Ruling Q-C (WO-4): G-iter is gated on a multigrid-friendly box — n rounded UP to a multiple
    of 2^(levels + 1) per axis, read as levels = the number of ladder rungs (3 -> 16)."""
    m = 2 ** (rungs + 1)
    return ((n + m - 1) // m) * m


def g2_case(Rih, off, mgbox=False):
    Ri, D, q, co = 1.0, 0.8, 0.3, 0.0
    Ro = 2.5 * Ri
    h = Ri / Rih
    n = int(math.ceil(2.0 * Ro / h)) + 6
    if mgbox:  # the added cells lie outside Ro: solid (the record's identity rows)
        n = mg_box(n)
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
    print("G2 concentric shells (steady), Neumann q inner, Dirichlet outer Ro = 2.5 Ri, Ri/h in {6, 12, 24}")
    e, l1, its = {}, {}, {}
    for Rih in (6, 12, 24):
        rows = [g2_case(Rih, off) for off in OFFSETS]
        e[Rih] = rms([r["err"] for r in rows])
        l1[Rih] = rms([r["l1"] for r in rows])
        its[Rih] = [r["it"] for r in rows]
        print(f"  Ri/h={Rih:3d}  c_Gamma rel err {[f'{r['err']:+.3e}' for r in rows]}  L1 {l1[Rih]:.3e}"
              f"  inner flux/(q area)-1 {[f'{r['fin']:+.1e}' for r in rows]}  iters {[r['it'] for r in rows]}")
        for r in rows:
            check(r["conv"], f"Ri/h={Rih}: converged ({r['it']} iterations)")
            check(r["rungs"][1] + r["rungs"][2] + r["rungs"][3] == 0, f"Ri/h={Rih}: R0 = 100 % ({r['rungs']})")
            check(abs(r["fin"]) <= 1e-12, f"Ri/h={Rih}: inner flux = q x discrete area ({r['fin']:+.1e})")
    for a, b in ((6, 12), (12, 24)):
        check(order(e[a], e[b]) >= 1.7, f"c_Gamma order {a}->{b} {order(e[a], e[b]):.2f} >= 1.7 (rms {e[a]:.3e} -> {e[b]:.3e})")
        check(order(l1[a], l1[b]) >= 1.8, f"field L1 order {a}->{b} {order(l1[a], l1[b]):.2f} >= 1.8 ({l1[a]:.3e} -> {l1[b]:.3e})")
    check(abs(e[24]) <= 2e-3, f"|c_Gamma err| {e[24]:.2e} <= 2e-3 at Ri/h = 24 (prov.)")
    print("  G-iter on multigrid-friendly boxes (ruling Q-C): n rounded up to a multiple of 16")
    itm = {}
    for Rih in (6, 12, 24):
        rows = [g2_case(Rih, off, mgbox=True) for off in OFFSETS]
        itm[Rih] = [r["it"] for r in rows]
        print(f"  Ri/h={Rih:3d}  n={mg_box(int(math.ceil(5.0 * Rih)) + 6)}  c_Gamma rel err "
              f"{[f'{r['err']:+.3e}' for r in rows]}  iters {itm[Rih]}")
        for r in rows:
            check(r["conv"], f"Ri/h={Rih} (mg box): converged ({r['it']} iterations)")
    print(f"  (native boxes, correctness only: iters {[max(its[k]) for k in sorted(its)]})")
    giter_check("G2 (mg box)", itm)


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
    print("G3a Robin external sphere (steady), R/h in {8, 16, 32}")
    for Da in (0.1, 1.0, 10.0, 100.0):
        e, its = {}, {}
        for Rh in (8, 16, 32):
            rows = [g3a_case(Rh, off, Da) for off in OFFSETS]
            e[Rh] = rms([r["err"] for r in rows])
            its[Rh] = [r["it"] for r in rows]
            print(f"  Da={Da:6.1f} R/h={Rh:3d}  Sh rel err {[f'{r['err']:+.3e}' for r in rows]}  iters {[r['it'] for r in rows]}")
            for r in rows:
                check(r["conv"], f"Da={Da} R/h={Rh}: converged ({r['it']} iterations)")
                check(r["rungs"][1] + r["rungs"][2] + r["rungs"][3] == 0, f"Da={Da} R/h={Rh}: R0 = 100 %")
        for a, b in ((8, 16), (16, 32)):
            check(order(e[a], e[b]) >= 1.7, f"Da={Da}: Sh order {a}->{b} {order(e[a], e[b]):.2f} >= 1.7 (rms {e[a]:.3e} -> {e[b]:.3e})")
        check(e[32] <= 3e-3, f"Da={Da}: |Sh err| {e[32]:.2e} <= 3e-3 at R/h = 32 (prov.)")
        if Da == 1.0:
            giter_check("G3a Da=1", its)


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
    add_cc(s, D, box="neumann")
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
    print("G7a pipe decay (BE ratio, dt mu = 1, 30 steps), R/h in {16, 32, 64}")
    for neumann in (False, True):
        e = {}
        for Rh in (16, 32, 64):
            rows = [g7a_case(Rh, off, neumann) for off in OFFSETS]
            e[Rh] = rms([r["err"] for r in rows])
            print(f"  {'neumann  j11p^2' if neumann else 'dirichlet j01^2'} R/h={Rh:3d}  rel err "
                  f"{[f'{r['err']:+.3e}' for r in rows]}  iters/step {min(min(r['its']) for r in rows)}..{max(max(r['its']) for r in rows)}")
            for r in rows:
                check(r["conv"], f"R/h={Rh}: converged")
                check(r["rungs"][1] + r["rungs"][2] + r["rungs"][3] == 0, f"R/h={Rh}: R0 = 100 %")
        tag = "neumann" if neumann else "dirichlet"
        for a, b in ((16, 32), (32, 64)):
            check(order(e[a], e[b]) >= 1.8, f"{tag}: order {a}->{b} {order(e[a], e[b]):.2f} >= 1.8 (rms {e[a]:.3e} -> {e[b]:.3e})")
        check(e[32] <= 5e-4, f"{tag}: |err| {e[32]:.2e} <= 5e-4 at R/h = 32")
    print("G7b Graetz Nu_T (inverse iteration with solve_scalar_steady), R/h in {16, 32, 64}")
    e = {}
    for Rh in (16, 32, 64):
        rows = [g7b_case(Rh, off) for off in OFFSETS]
        e[Rh] = rms([r["err"] for r in rows])
        print(f"  R/h={Rh:3d}  Nu_T rel err {[f'{r['err']:+.3e}' for r in rows]}  iters/solve "
              f"{min(min(r['its']) for r in rows)}..{max(max(r['its']) for r in rows)}")
        for r in rows:
            check(r["conv"], f"R/h={Rh}: converged")
    for a, b in ((16, 32), (32, 64)):
        check(order(e[a], e[b]) >= 1.7, f"Graetz order {a}->{b} {order(e[a], e[b]):.2f} >= 1.7 (rms {e[a]:.3e} -> {e[b]:.3e})")
    check(e[32] <= 1e-3, f"Graetz |err| {e[32]:.2e} <= 1e-3 at R/h = 32 (prov.)")


# ------------------------------------------------------------------------------------- G-iter ----
def giter_transient_case(Rh, off, nsteps=6):
    """G1's sphere, transient at dt D/h^2 = 1 from c = 0 (Dirichlet wall c = 1, box c = 0)."""
    R, D = 1.0, 0.7
    L = 4.0 * R
    h = R / Rh
    n = int(round(L / h))
    s = walled((n, n, n), (L, L, L))
    s.set_dt(h * h / D)
    c0 = np.array([0.5 * L, 0.5 * L, 0.5 * L]) + np.asarray(off) * h
    X, Y, Z = grid(s)
    r = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2)
    s.set_solid(np.asfortranarray(r - R))
    add_cc(s, D, box="dirichlet")
    s.set_scalar_wall("c", "dirichlet", 1.0)
    its, lv = [], []
    for k in range(nsteps):
        s.advance_scalars()
        c = s.diagnostics.scalar_census("c")
        its.append(c["krylov_iterations"])
        lv.append(c["mg_levels"])
    return its, lv


def g5b_geometry_case(n, off):
    """G5b's geometry (periodic simple-cubic array, c = 0.3) on an n^3 grid, insulating walls with a
    flux and a source, steady: the singular case of §5.1 (same surrogate as G5b's closure)."""
    L = 1.0
    h = L / n
    s = pf.Solver((n, n, n), extent=(L, L, L))
    s.set_rho(1.0)
    s.set_mu(1.0)
    R = 0.5 * L * (0.3 * 6.0 / math.pi) ** (1.0 / 3.0)
    c0 = np.array([0.5 * L, 0.5 * L, 0.5 * L]) + np.asarray(off) * h
    X, Y, Z = grid(s)
    r = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2)
    s.set_solid(np.asfortranarray(r - R))
    s.add_scalar("c", diffusivity=0.6, cutcell=True)
    s.set_scalar_wall("c", "neumann", 0.2)
    s.set_scalar_source("c", -0.5)
    s.solve_scalar_steady("c")
    c = s.diagnostics.scalar_census("c")
    return c["krylov_iterations"], c["krylov_converged"], c["mg_levels"], 2.0 * R / h


def gate_giter():
    print("G-iter transient: G1's sphere at dt D/h^2 = 1, 6 steps from c = 0, R/h in {8, 16, 32}")
    for Rh in (8, 16, 32):
        rows = [giter_transient_case(Rh, off) for off in OFFSETS]
        print(f"  R/h={Rh:3d}  iterations/step {[its for its, _ in rows]}  mg levels {rows[0][1][0]}")
        # rulings D-WO4-3 (the provisional <= 8 restated as <= 10) and D-WO9-3 (cold first step <= 12)
        cold = max(its[0] for its, _ in rows)
        warm = max(max(its[1:]) for its, _ in rows)
        check(cold <= 12, f"R/h={Rh}: <= 12 iterations on the cold first step ({cold})")
        check(warm <= 10, f"R/h={Rh}: <= 10 iterations per warm step ({warm})")
    print("G-iter singular steady: G5b's geometry (periodic SC array, c = 0.3), n in {32, 48, 80}"
          " (n = 20/40/80 rounded up to multiples of 16, ruling Q-C)")
    mx = []
    for n in (mg_box(20), mg_box(40), mg_box(80)):
        rows = [g5b_geometry_case(n, off) for off in OFFSETS]
        mx.append(max(r[0] for r in rows))
        print(f"  n={n:3d} (ND {rows[0][3]:.1f})  iterations {[r[0] for r in rows]}  mg levels {rows[0][2]}")
        for r in rows:
            check(r[1], f"n={n}: converged ({r[0]} iterations)")
    check(max(mx) <= 30, f"singular: <= 30 iterations at every rung ({mx})")
    gr = [b - a for a, b in zip(mx, mx[1:])]
    check(all(g <= 5 for g in gr), f"singular: growth <= 5 per doubling ({gr})")


# ---------------------------------------------------------------------------------------- G9 ----
R_I, R_O = 0.4, 1.0


def psi_node(X, Y):
    """Stream function of solid-body rotation (Omega = 1, u = d psi/dy, v = -d psi/dx), clamped to
    its wall values inside the solids."""
    r = np.clip(np.hypot(X, Y), R_I, R_O)
    return -0.5 * r * r


def annulus_case(Rh, off, scheme="koren"):
    """Coaxial cylinders R_i = 0.4 < r < R_o = 1, an SDF along z, nz = 4, every face periodic."""
    h = R_O / Rh
    n = 2 * Rh + 4
    L = n * h
    s = pf.Solver((n, n, 4), extent=(L, L, 4 * h), origin=(-0.5 * L + off[0] * h,
                                                           -0.5 * L + off[1] * h, 0.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    X, Y, Z = grid(s)
    r = np.hypot(X, Y)
    s.set_solid(np.asfortranarray(np.minimum(r - R_I, R_O - r)), cutcell_pressure=True)
    s.add_scalar("c", diffusivity=0.0, scheme=scheme, cutcell=True)
    return s, h


def annulus_fluxes(s, h):
    """Physical F/V through the low x and y faces of every cell: the differences of a NODE
    potential (round 6's exact stream-function fluxes), so every cell's net flux telescopes to zero.
    The potential is the clamped psi, set to its wall value on both nodes of every face the
    projection closes (openness 0): the field is then discretely divergence-free in the openness
    the predicate reads (the solver's gated cut-cell openness), as a projected field is."""
    x, y, _ = s.cell_centers()
    xl, yl = np.asarray(x) - 0.5 * h, np.asarray(y) - 0.5 * h
    Xn, Yn = np.meshgrid(xl, yl, indexing="ij")  # node (x_lo, y_lo) of cell (i, j)
    P = psi_node(Xn, Yn)
    rn = np.hypot(Xn, Yn)
    wall = np.where(rn < 0.5 * (R_I + R_O), -0.5 * R_I * R_I, -0.5 * R_O * R_O)
    shut_x = s.get_ox_proj()[:, :, 0] <= 0.0  # x face (i, j): nodes (i, j), (i, j + 1)
    shut_y = s.get_oy_proj()[:, :, 0] <= 0.0  # y face (i, j): nodes (i, j), (i + 1, j)
    mark = shut_x | np.roll(shut_x, 1, axis=1) | shut_y | np.roll(shut_y, 1, axis=0)
    P = np.where(mark, wall, P)
    fx = (np.roll(P, -1, axis=1) - P) / (h * h)     # int u dy over the low x face, per volume
    fy = -(np.roll(P, -1, axis=0) - P) / (h * h)
    return fx[:, :, None] * np.ones(4), fy[:, :, None] * np.ones(4)


def set_annulus_velocity(s, fx, fy):
    """Face velocities (internal index velocity) with open * vel = F/V * tRef, the openness the
    projection conserves; returns max |F| on faces whose openness is 0 (lost) over max |F|."""
    tref = s.unit_scales["t_ref"]
    ox, oy = s.get_ox_proj(), s.get_oy_proj()
    vx = np.where(ox > 0.0, fx * tref / np.where(ox > 0.0, ox, 1.0), 0.0)
    vy = np.where(oy > 0.0, fy * tref / np.where(oy > 0.0, oy, 1.0), 0.0)
    s.set_field("u", np.asfortranarray(vx))
    s.set_field("v", np.asfortranarray(vy))
    s.set_field("w", np.asfortranarray(np.zeros_like(vx)))
    lost = max(float(np.max(np.abs(np.where(ox > 0.0, 0.0, fx)))),
               float(np.max(np.abs(np.where(oy > 0.0, 0.0, fy)))))
    return lost / max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))))


def g9_case(Rh, off, scheme, cr, rtol=None, nsteps=None):
    s, h = annulus_case(Rh, off, scheme)
    geo = s.diagnostics.scalar_geometry("c")
    unk = geo["unknown"] > 0.5
    kap = geo["kappa"]
    ax, ay, az = geo["aperture_x"], geo["aperture_y"], geo["aperture_z"]
    full = unk & (kap >= 1.0)
    for a, ap in ((0, ax), (1, ay), (2, az)):
        full &= (ap >= 1.0) & (np.roll(ap, -1, axis=a) >= 1.0)
    fx, fy = annulus_fluxes(s, h)
    out = (np.maximum(np.roll(fx, -1, axis=0), 0) + np.maximum(-fx, 0) +
           np.maximum(np.roll(fy, -1, axis=1), 0) + np.maximum(-fy, 0))
    T = 2.0 * math.pi
    nrev = int(math.ceil(T * float(np.max(out[full])) / cr))
    dt = T / nrev
    s.set_dt(dt)
    lost = set_annulus_velocity(s, fx, fy)
    X, Y, _ = grid(s)
    c0 = np.where(unk, np.exp(-((X - 0.7) ** 2 + Y * Y) / (2 * 0.08**2)), 0.0)
    s.set_field("c", np.asfortranarray(c0))
    if rtol is not None:
        s.set_scalar_tolerance("c", rtol)
    m0 = float(np.sum(kap * c0) * h**3)
    cmax0 = float(c0.max())
    st = dict(min=0.0, max=cmax0, ident=0.0, defects=0.0, finite=True, small=0, impl=0.0,
              cb=0.0, its=0, lost=lost, minkap=float(kap[unk].min()), steps=0)
    nst = nrev if nsteps is None else nsteps
    for k in range(nst):
        s.advance_scalars()
        b = s.diagnostics.scalar_budget("c")
        st["ident"] = max(st["ident"], abs(b["identity_error"]) / m0)
        st["defects"] += b["defect"]
        c = s.get_field("c")[unk]
        st["finite"] = st["finite"] and bool(np.all(np.isfinite(c)))
        st["min"] = min(st["min"], float(c.min()))
        st["max"] = max(st["max"], float(c.max()))
        cen = s.diagnostics.scalar_census("c")
        st["small"] = max(st["small"], cen["num_small_cells"])
        st["impl"] = max(st["impl"], cen["num_implicit_faces"] / max(1, cen["num_flux_faces"]))
        st["cb"] = max(st["cb"], cen["bulk_courant"])
        st["its"] = max(st["its"], cen["krylov_iterations"])
        st["guarded"] = cen["num_guarded_flux_faces"]
        st["steps"] += 1
    c = s.get_field("c")
    m1 = float(np.sum(kap * c) * h**3)
    st["m0"], st["m1"], st["cmax0"] = m0, m1, cmax0
    st["cons"] = abs(m1 - m0 + st["defects"]) / m0  # d_mass = -defect per step (scalar_budget)
    st["drift"] = abs(m1 - m0) / m0
    err = np.where(unk, kap * np.abs(c - c0), 0.0)
    st["L1"] = float(err.sum() / np.sum(kap * c0))
    cut = unk & (kap < 1.0)
    band = cut.copy()
    for a in (0, 1):
        band |= np.roll(cut, 1, axis=a) | np.roll(cut, -1, axis=a)
    band &= unk
    st["cutshare"] = float(err[cut].sum() / err.sum())
    st["bandshare"] = float(err[band].sum() / err.sum())
    return st


def g9_offsets(Rh, want=3):
    """Grid offsets from a fixed sequence whose geometry carries a sliver (min kappa < 1e-2, gate
    (v)); the first `want` that do."""
    rng = np.random.default_rng(9000 + Rh)
    found = []
    for _ in range(40):
        off = rng.uniform(-0.5, 0.5, size=2)
        s, _h = annulus_case(Rh, off)
        geo = s.diagnostics.scalar_geometry("c")
        if float(geo["kappa"][geo["unknown"] > 0.5].min()) < 1e-2:
            found.append(off)
            if len(found) == want:
                break
    return found


def koren_bulk_control(cr, n=32):
    """No solid: a periodic box, uniform translation (1, 0.5) for one x-period, Koren at bulk Courant
    cr. Returns (min c, C_bulk) — the bulk scheme's own boundedness, for G9 (iii)'s reading."""
    s = pf.Solver((n, n, 4), extent=(1.0, 1.0, 4.0 / n))
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_pressure_geometry(np.asfortranarray(np.ones((n, n, 4))))
    s.add_scalar("c", diffusivity=0.0, scheme="koren", cutcell=True)
    h, U = 1.0 / n, (1.0, 0.5)
    dt = cr * h / (U[0] + U[1])
    s.set_dt(dt)
    tr = s.unit_scales["t_ref"]
    s.set_field("u", np.asfortranarray(np.full((n, n, 4), U[0] * tr / h)))
    s.set_field("v", np.asfortranarray(np.full((n, n, 4), U[1] * tr / h)))
    X, Y, _ = grid(s)
    s.set_field("c", np.asfortranarray(np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.08**2))))
    mn = 0.0
    for _ in range(int(round(1.0 / (U[0] * dt)))):
        s.advance_scalars()
        mn = min(mn, float(s.get_field("c").min()))
    return mn, s.diagnostics.scalar_census("c")["bulk_courant"]


def gate_g9():
    print("G9: advection in an annulus (R_i 0.4, R_o 1, nz 4), solid-body rotation, a Gaussian blob "
          "carried one revolution, D = 0 (round 6's set-up)")
    for cr in (0.5, 0.9):
        mn, cb = koren_bulk_control(cr)
        print(f"  control, NO solid, uniform translation, koren at C_bulk {cb:.3f}: min c = {mn:.2e} "
              f"(max c0 = 1)")
    for Rh in (16, 32, 64):
        off = g9_offsets(Rh, 1)
        check(len(off) == 1, f"R_o/h={Rh}: an offset with a sliver (min kappa < 1e-2) exists")
        off = off[0]
        for scheme in ("fou", "koren"):
            for cr in (0.5, 0.9):
                t0 = time.time()
                st = g9_case(Rh, off, scheme, cr)
                tag = f"R_o/h={Rh} {scheme:5s} C={cr}"
                print(f"  {tag}: {st['steps']} steps, C_bulk {st['cb']:.3f}, small cells <= {st['small']}, "
                      f"implicit faces <= {100 * st['impl']:.2f} %, BiCGStab <= {st['its']}, min kappa "
                      f"{st['minkap']:.1e}, lost flux {st['lost']:.1e}, guarded faces {st['guarded']}, "
                      f"L1 {st['L1']:.3e} (cut cells {100 * st['cutshare']:.1f} %, cut band "
                      f"{100 * st['bandshare']:.1f} %), min {st['min']:.2e}, max {st['max']:.6f} "
                      f"[{time.time() - t0:.0f} s]")
                check(st["ident"] <= 1e-13, f"{tag}: (i) budget identity per step {st['ident']:.1e} M0 <= 1e-13")
                check(st["cons"] <= 1e-12, f"{tag}: (ii) |M_end - M0 + sum defect| = {st['cons']:.1e} M0 <= 1e-12")
                if scheme == "fou":
                    check(st["min"] >= -1e-12 * st["cmax0"], f"{tag}: (iii) min c {st['min']:.1e} >= -1e-12 max c0")
                elif cr <= 0.5:  # ruling D-WO5-1: koren's (iii) is gated at bulk Courant <= 1/2
                    check(st["min"] >= -1e-3 * st["cmax0"] and st["max"] <= (1 + 1e-3) * st["cmax0"],
                          f"{tag}: (iii) min c {st['min']:.1e} >= -1e-3, max c {st['max']:.6f} <= 1.001 max c0")
                else:  # information only (D-WO5-1): forward Euler + the legacy Koren reconstruction is
                    # TVD only to C <= 1/2 — it fails in the bulk with no solid at all
                    # (koren_bulk_control above), so no small-cell treatment can meet it.
                    print(f"    INFO {tag}: (iii) not gated above bulk Courant 1/2 (D-WO5-1): min c "
                          f"{st['min']:.1e}, max c {st['max']:.6f} (max c0 {st['cmax0']:.6f})")
                check(st["finite"], f"{tag}: (iv) finite throughout")
                check(st["minkap"] < 1e-2, f"{tag}: (v) min kappa {st['minkap']:.1e} < 1e-2")
                check(abs(st["cb"] - cr) <= 0.02 * cr, f"{tag}: bulk Courant {st['cb']:.3f} ~ {cr}")
        st = g9_case(Rh, off, "fou", 0.9, rtol=1e-13)
        print(f"  R_o/h={Rh} fou C=0.9 rtol 1e-13: |dM|/M0 = {st['drift']:.1e}, "
              f"|dM + sum defect| = {st['cons']:.1e} M0")
        check(st["drift"] <= 1e-8, f"R_o/h={Rh}: (ii) sanity: raw drift {st['drift']:.1e} <= 1e-8 at rtol 1e-13")


# --------------------------------------------------------------------------------------- G9b ----
def g9b_case(kind, scheme, n=32, nsteps=50):
    cls = pf.Solver if kind == "staggered" else pf.SolverColocated
    s = cls((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(0.01)
    if scheme == "staggered-ghost":
        s.diagnostics.set_ghost_projection(True)
    elif scheme is not None:
        s.set_collocated_scheme(scheme)
    R = (0.3 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)  # SC array, solid fraction 0.3
    X, Y, Z = grid(s)
    s.set_solid(np.asfortranarray(np.sqrt((X - 0.513) ** 2 + (Y - 0.479) ** 2 + (Z - 0.507) ** 2) - R),
                cutcell_pressure=True)
    s.set_body_force((30.0, 9.0, 0.0))  # Stokes (advection off): bulk Courant ~0.47 at step 50
    s.add_scalar("c", diffusivity=0.0, cutcell=True)
    unk = s.diagnostics.scalar_geometry("c")["unknown"] > 0.5
    s.set_field("c", np.asfortranarray(np.where(unk, 1.0, 0.0)))
    dev = 0.0
    for _ in range(nsteps):
        s.step()
        dev = max(dev, float(np.max(np.abs(s.get_field("c")[unk] - 1.0))))
    return dev, s.diagnostics.scalar_census("c")


def gate_g9b():
    print("G9b: constant preservation under flow's own projection (Stokes through an SC sphere "
          "array, c = 1, 50 steps, D = 0)")
    for kind, scheme in (("staggered", None), ("collocated", "gauge-exact"), ("collocated", "plain"),
                         ("collocated", "embed")):
        dev, cen = g9b_case(kind, scheme)
        tag = f"{kind} {scheme or ''}".strip()
        print(f"  {tag}: max|c - 1| = {dev:.2e}; C_bulk {cen['bulk_courant']:.3f}, flux faces "
              f"{cen['num_flux_faces']}, implicit {cen['num_implicit_faces']}, guarded "
              f"{cen['num_guarded_flux_faces']}")
        check(dev <= 1e-6, f"{tag}: max|c - 1| = {dev:.1e} <= 1e-6")
        check(cen["bulk_courant"] > 0.3, f"{tag}: the flow advects (C_bulk {cen['bulk_courant']:.3f})")
    for kind, scheme in (("collocated", "ghost"), ("staggered", "staggered-ghost")):
        raises(RuntimeError, lambda k=kind, sc=scheme: g9b_case(k, sc, nsteps=1),
               f"{kind} {scheme}: refused for a cut-cell scalar (Q13: no divergence-free face flux)")


# --------------------------------------------------------------------------------------- G9c ----
def channel_case(kind="staggered", n=24, dt=0.008, D=0.01, scheme="koren", out_bc="neumann",
                 c0=0.2, cin=1.0, cap=True):
    """A 2 x 1 x 1 channel (2n x n x n cells): inflow U = 1 on -x, outflow on +x, no-slip walls on y
    and z, mu 0.05; a solid sphere (R 0.23) off the axis and, with `cap`, a second one (R 0.2)
    centred just outside the outlet, cutting the outflow face (slivers in the outlet column). The
    cut-cell scalar: Dirichlet `cin` on the inflow face, `out_bc` on the outflow face (Dirichlet
    0.5), Neumann on the walls and the solids; `c0` in the fluid."""
    cls = pf.Solver if kind == "staggered" else pf.SolverColocated
    s = cls((2 * n, n, n), extent=(2.0, 1.0, 1.0))
    if kind != "staggered":
        s.set_collocated_scheme("gauge-exact")  # not the refused 'ghost' (D-WO5-4)
    s.set_rho(1.0)
    s.set_mu(0.05)
    s.set_dt(dt)
    s.set_domain_bc("-x", "inflow", velocity=(1.0, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow")
    for f in FACES[2:]:
        s.set_domain_bc(f, "wall")
    X, Y, Z = grid(s)
    sdf = np.sqrt((X - 0.71) ** 2 + (Y - 0.48) ** 2 + (Z - 0.53) ** 2) - 0.23
    if cap:
        sdf = np.minimum(sdf, np.sqrt((X - 2.07) ** 2 + (Y - 0.31) ** 2 + (Z - 0.64) ** 2) - 0.2)
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True)
    s.add_scalar("c", diffusivity=D, scheme=scheme, cutcell=True)
    s.set_scalar_bc("c", "-x", "dirichlet", cin)
    if out_bc == "dirichlet":
        s.set_scalar_bc("c", "+x", "dirichlet", 0.5)
    else:
        s.set_scalar_bc("c", "+x", "neumann")
    for f in FACES[2:]:
        s.set_scalar_bc("c", f, "neumann")
    geo = s.diagnostics.scalar_geometry("c")
    unk = geo["unknown"] > 0.5
    s.set_field("c", np.asfortranarray(np.where(unk, c0, 0.0)))
    return s, unk, geo["kappa"], dt, (1.0 / n) ** 3


def g9c_run(nsteps, **kw):
    """Step the channel; per step the budget identity with the mass change summed exactly
    (math.fsum of kappa V (c - c^n) over the unknowns: the budget's own d_mass differences two
    O(M) reductions whose round-off, ~1e-14 M, is ~1e-12 of a step's change in a long channel),
    and the run's conservation, bounds and census."""
    s, unk, kap, dt, V = channel_case(**kw)
    cp = s.get_field("c")
    m0 = V * math.fsum((kap * cp)[unk].ravel())
    st = dict(ident=0.0, internal=0.0, flows=0.0, defects=0.0, min=float(cp[unk].min()),
              max=float(cp[unk].max()), finite=True, small=0, impl=0, cb=0.0, its=0, steps=0)
    for _ in range(nsteps):
        s.step()
        b = s.diagnostics.scalar_budget("c")
        c = s.get_field("c")
        dm = V * math.fsum((kap * (c - cp))[unk].ravel())
        flows = dt * (b["wall_in"] + b["boundary_in"] + b["source_in"])
        if dm != 0.0:  # (the constant run changes nothing)
            st["ident"] = max(st["ident"], abs(dm - flows + b["defect"]) / abs(dm))
        st["internal"] = max(st["internal"], abs(b["identity_error"]) / b["mass"])
        st["flows"] += flows
        st["defects"] += b["defect"]
        cu = c[unk]
        st["finite"] = st["finite"] and bool(np.all(np.isfinite(cu)))
        st["min"] = min(st["min"], float(cu.min()))
        st["max"] = max(st["max"], float(cu.max()))
        cen = s.diagnostics.scalar_census("c")
        st["small"] = max(st["small"], cen["num_small_cells"])
        st["impl"] = max(st["impl"], cen["num_implicit_faces"])
        st["cb"] = max(st["cb"], cen["bulk_courant"])
        st["its"] = max(st["its"], cen["krylov_iterations"])
        st["steps"] += 1
        cp = c
    m1 = V * math.fsum((kap * cp)[unk].ravel())
    st["cons"] = abs(m1 - m0 - st["flows"] + st["defects"]) / m0
    last = unk[-1]
    st["outlet"] = float(np.sum((kap * cp)[-1][last]) / np.sum(kap[-1][last]))
    st["s"] = s
    return st


def gate_g9c():
    print("G9c: open domain faces (WO-5b, D-WO5-3): a 2 x 1 x 1 channel, inflow U = 1 / no-slip walls "
          "/ outflow, a sphere + a cap cutting the outlet, the scalar on flow's own projection")
    # (tag, settings, steps (t ~ 2, one flow-through, for the bounded rows), bounded, small cells)
    rows = (("koren C<=1/2", dict(scheme="koren", dt=0.007), 286, True, False),
            ("fou C~0.9", dict(scheme="fou", dt=0.0135), 148, True, True),
            ("koren, Dirichlet outlet", dict(scheme="koren", dt=0.007, out_bc="dirichlet"), 40, False,
             False))
    for tag, kw, nsteps, bounded, small in rows:
        t0 = time.time()
        st = g9c_run(nsteps, **kw)
        print(f"  {tag}: {st['steps']} steps, C_bulk <= {st['cb']:.3f}, small cells <= {st['small']}, "
              f"implicit faces <= {st['impl']}, BiCGStab <= {st['its']}; identity <= {st['ident']:.1e} "
              f"|dM| (internal identity_error <= {st['internal']:.1e} M), run conservation "
              f"{st['cons']:.1e} M0; c - [0.2, 1] in [{st['min'] - 0.2:.1e}, {st['max'] - 1.0:.1e}]; outlet mean "
              f"{st['outlet']:.3f} [{time.time() - t0:.0f} s]")
        check(st["ident"] <= 1e-13, f"{tag}: per-step budget identity {st['ident']:.1e} |dM| <= 1e-13")
        check(st["internal"] <= 1e-12, f"{tag}: identity_error {st['internal']:.1e} M <= 1e-12 (sanity)")
        check(st["cons"] <= 1e-12, f"{tag}: |M_end - M0 - sum dt flows + sum defects| {st['cons']:.1e} M0 <= 1e-12")
        check(st["finite"], f"{tag}: finite throughout")
        if small:
            check(st["small"] > 0 and st["impl"] > 0, f"{tag}: small cells and implicit faces occur")
        if bounded:  # fou: to the solve's tolerance (rtol 1e-10, implicit diffusion); koren: G9's 1e-3
            tol = 1e-10 if kw["scheme"] == "fou" else 1e-3
            check(st["min"] >= 0.2 - tol and st["max"] <= 1.0 + tol,
                  f"{tag}: bounded by the data, c in [{st['min']:.6f}, {st['max']:.6f}] within {tol:.0e}")
            check(st["outlet"] > 0.3, f"{tag}: the front reached the outlet (mean {st['outlet']:.3f})")
        if kw["scheme"] == "koren":
            check(st["cb"] <= 0.5, f"{tag}: bulk Courant {st['cb']:.3f} <= 1/2 (D-WO5-1)")
    # constant preservation: c = cin = 1
    st = g9c_run(20, scheme="koren", dt=0.008, c0=1.0, cin=1.0)
    dev = max(abs(st["min"] - 1.0), abs(st["max"] - 1.0))
    print(f"  constant c = 1 (inflow 1): max|c - 1| = {dev:.2e}")
    check(dev <= 1e-6, f"constant preservation through the open faces: max|c - 1| = {dev:.1e} <= 1e-6")
    # steady (implicit FOU everywhere, §6.7) at a low cell Peclet number: the rate balance
    s = st["s"]  # the developed flow of the last run; a second scalar, solved in place
    s.add_scalar("d", diffusivity=0.5, cutcell=True)
    s.set_scalar_bc("d", "-x", "dirichlet", 1.0)
    for f in FACES[1:]:
        s.set_scalar_bc("d", f, "neumann")
    s.solve_scalar_steady("d")
    b = s.diagnostics.scalar_budget("d")
    cen = s.diagnostics.scalar_census("d")
    rate = 1.0  # the inflow's advective rate Q cin = U A cin
    print(f"  steady, D = 0.5 (Pe_h {1.0 / 24 / 0.5:.3f}): {cen['krylov_iterations']} iterations, "
          f"boundary_in {b['boundary_in']:.3e}, identity_error {b['identity_error']:.1e} (rate {rate})")
    check(cen["krylov_converged"], "steady with open faces converges")
    check(abs(b["identity_error"]) <= 1e-12 * rate, f"steady rate balance {abs(b['identity_error']):.1e} <= 1e-12 Q cin")
    # the collocated grid is refused (open after WO-5b: no constrained high-side face flux to capture)
    raises(RuntimeError, lambda: channel_case(kind="collocated")[0].step(),
           "collocated: an open domain face is refused for a cut-cell scalar (open after WO-5b)")
    # before any projection there is no open-face flux to advect with: refused, naming step()
    s, unk, _k, _dt, _V = channel_case(n=12)
    s.set_field("u", np.asfortranarray(np.full(unk.shape, 1.0)))
    raises(RuntimeError, s.advance_scalars, "open faces, a moving fluid and no projection yet: refused")
    backflow_steady_row()


def capture_stderr(fn):
    """Run fn() with the C-level stderr captured; returns (fn's result, the text)."""
    import os
    import tempfile
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        sys.stderr.flush()
        fd = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            r = fn()
        finally:
            os.dup2(fd, 2)
            os.close(fd)
        tmp.seek(0)
        return r, tmp.read().decode(errors="replace")


def backflow_steady_row(n=12, nsteps=10):
    """Review finding 2 (rulings D-WOR-2a/b/c): open faces but NO inflow face -- both x ends
    'outflow', the flow driven by a body force, so the whole -x face backflows; walls on y and z, a
    Neumann sphere. A steady scalar, insulating everywhere, no source, from a non-uniform start:
    the problem is not the uniform-mean singular case (no compatibility projection, no gauge), the
    solve converges -- to a constant, the null vector --, and the backflow rows are counted and
    warned about once. Returns the census."""
    s = pf.Solver((2 * n, n, n), extent=(2.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(0.05)
    s.set_dt(0.01)
    s.set_domain_bc("-x", "outflow")
    s.set_domain_bc("+x", "outflow")
    for f in FACES[2:]:
        s.set_domain_bc(f, "wall")
    X, Y, Z = grid(s)
    s.set_solid(np.asfortranarray(np.sqrt((X - 0.97) ** 2 + (Y - 0.48) ** 2 + (Z - 0.53) ** 2) - 0.23),
                cutcell_pressure=True)
    s.set_body_force((1.0, 0.0, 0.0))
    for _ in range(nsteps):
        s.step()
    s.add_scalar("b", diffusivity=0.05, cutcell=True)
    for f in FACES:
        s.set_scalar_bc("b", f, "neumann")
    unk = s.diagnostics.scalar_geometry("b")["unknown"] > 0.5
    s.set_field("b", np.asfortranarray(np.where(unk, 0.5 + 0.3 * np.sin(3.0 * X) * np.cos(2.0 * Y), 0.0)))
    _, err = capture_stderr(lambda: s.solve_scalar_steady("b"))
    cen = s.diagnostics.scalar_census("b")
    c = s.get_field("b")[unk]
    spread = float(c.max() - c.min())
    print(f"  backflow (both x ends outflow, body force): {cen['num_backflow_faces']} backflow rows, "
          f"{cen['krylov_iterations']} iterations, residual {cen['krylov_residual']:.1e}, "
          f"incompatibility {cen['steady_incompatibility']:.1e}, max - min c = {spread:.1e}; "
          f"stderr: {err.strip()!r}")
    check(cen["num_backflow_faces"] > 0, f"backflow: census num_backflow_faces > 0 ({cen['num_backflow_faces']})")
    check("BACKFLOW" in err and "num_backflow_faces" in err, "backflow: the steady solve warns, naming the census key")
    check(cen["krylov_converged"], f"backflow, no inflow face: the steady solve converges "
                                   f"({cen['krylov_iterations']} iterations, D-WOR-2b)")
    check(cen["steady_incompatibility"] == 0.0, "backflow: not treated as the uniform-mean singular case")
    check(spread <= 1e-6, f"backflow: the converged field is the null vector, a constant (spread {spread:.1e})")
    _, err2 = capture_stderr(lambda: s.solve_scalar_steady("b"))
    check("BACKFLOW" not in err2, "backflow: the warning is printed once per scalar")
    return cen


# ------------------------------------------------------------------------- WO-6: closures ----
def low_face_flux(s, name, a):
    """The scalar's advective face flux per unit area through the LOW a-face of each cell
    (physical, c-free: open * u), from fields: the projection's openness (get_ox/oy/oz, the
    staggered non-ghost openness of §6.1) times the face velocity (get_field, internal, converted),
    zero unless both cells are fluid unknowns (the guard of §6.1). Periodic roll; a non-periodic
    axis' low boundary face is zeroed by the caller's mask."""
    geo = s.diagnostics.scalar_geometry(name)
    unk = geo["unknown"] > 0.5
    o = (s.get_ox, s.get_oy, s.get_oz)[a]()
    v = s.get_field("uvw"[a]) / s.unit_scales["velocity_to_internal"][a]
    return np.where(unk & np.roll(unk, 1, axis=a), o * v, 0.0)


def cell_velocity(s, name, periodic=(True, True, True)):
    """U_i/V (§1.5) per cell, physical: sum_f F_out (x_f - x_i) / V = the mean of the cell's two
    a-face fluxes per area, per axis."""
    U = []
    for a in range(3):
        f = low_face_flux(s, name, a)
        if not periodic[a]:
            idx = [slice(None)] * 3
            idx[a] = 0
            f[tuple(idx)] = 0.0
        U.append(0.5 * (f + np.roll(f, -1, axis=a)))
    return U


def mean_flux_fields(s, name, D, G, periodic):
    """scalar_mean_flux recomputed from fields (§8.1): the diffusive face sum of -D a (c_j - c_i)/h
    (c = theta + G.x, the true displacement) over the coupled low faces, plus the advective
    sum (U_i/V - kappa_i Ubar) theta_i, both over the whole box's cell count. Returns
    (J (3,), the advective part (3,), Ubar (3,), phi = sum kappa / N)."""
    geo = s.diagnostics.scalar_geometry(name)
    unk = geo["unknown"] > 0.5
    kap = np.where(unk, geo["kappa"], 0.0)
    th = s.get_field(name)
    h = s.spacing
    N = th.size
    U = cell_velocity(s, name, periodic)
    ksum = math.fsum(kap[unk].ravel())
    ub = [math.fsum(U[a][unk].ravel()) / ksum for a in range(3)]
    J, Ja = np.zeros(3), np.zeros(3)
    for a, ap in enumerate((geo["aperture_x"], geo["aperture_y"], geo["aperture_z"])):
        cpl = unk & np.roll(unk, 1, axis=a)
        if not periodic[a]:
            idx = [slice(None)] * 3
            idx[a] = 0
            cpl[tuple(idx)] = False
        dc = th - np.roll(th, 1, axis=a) + G[a] * h[a]
        J[a] = math.fsum((-D * ap * dc / h[a])[cpl].ravel()) / N
        Ja[a] = math.fsum(((U[a] - kap * ub[a]) * th)[unk].ravel()) / N
    return J + Ja, Ja, ub, ksum / N


def poiseuille_face_average(s, c0, R, U, oz, m=24):
    """G8's velocity (ruling D-WO6-1): the Poiseuille flux through the open part of each z-face
    (m x m midpoint sub-samples of the face), divided by oz A, so that F = oz w A carries no
    point-sampling error."""
    x, y, _ = (np.asarray(v) for v in s.cell_centers())
    h = x[1] - x[0]
    g = (np.arange(m) + 0.5) / m - 0.5
    GX, GY = np.meshgrid(g * h, g * h, indexing="ij")
    flux = np.zeros((x.size, y.size))
    for i in range(x.size):
        px = x[i] + GX - c0[0]
        for j in range(y.size):
            py = y[j] + GY - c0[1]
            r2 = (px * px + py * py) / (R * R)
            flux[i, j] = np.mean(np.where(r2 < 1.0, 2.0 * U * (1.0 - r2), 0.0))
    o = oz[:, :, :1][:, :, 0]
    w2 = np.where(o > 0, flux / np.where(o > 0, o, 1.0), 0.0)
    return np.repeat(w2[:, :, None], oz.shape[2], axis=2)


def g8_case(Rh, off, U=1.0, Pe=10.0, nbox=None, nz=4, exact_flux=True):
    """G8: G7's pipe (R = 1, nz = 4 periodic), frozen Poiseuille z-face velocities (set_field;
    ruling D-WO6-1: the FACE-AVERAGED flux over the open part of each face; divergence-free
    because z-invariant), insulating walls, mean gradient e_z, steady. Returns the Taylor-Aris coefficient -<u'theta> (interstitial, from fields) over
    its exact value Pe^2 D/48, Pe = U R/D. `nbox`/`nz` give G-iter's multigrid-friendly box (ruling
    D-WO4-1; the problem is z-invariant, so nz changes nothing but the cell count);
    `exact_flux=False` the INFO variant: face-centre point samples, clipped >= 0."""
    R = 1.0
    D = U * R / Pe
    h = R / Rh
    n = nbox or 2 * Rh + 6
    L = n * h
    s = walled((n, n, nz), (L, L, nz * h), periodic_z=True)
    c0 = np.array([0.5 * L, 0.5 * L]) + np.asarray(off[:2]) * h
    X, Y, Z = grid(s)
    rr = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2)
    s.set_solid(np.asfortranarray(R - rr), cutcell_pressure=True)
    if exact_flux:
        w = poiseuille_face_average(s, c0, R, U, s.get_oz())
    else:
        w = np.maximum(2.0 * U * (1.0 - rr * rr / (R * R)), 0.0)  # z-face centres share (x, y)
    s.set_field("w", np.asfortranarray(w * s.unit_scales["velocity_to_internal"][2]))
    add_cc(s, D, box="neumann", periodic_z=True)
    s.set_scalar_mean_gradient("c", (0.0, 0.0, 1.0))
    s.solve_scalar_steady("c")
    per = (False, False, True)
    J, Ja, ub, phi = mean_flux_fields(s, "c", D, (0.0, 0.0, 1.0), per)
    Jc = np.asarray(s.scalar_mean_flux("c"))
    val = -Ja[2] / phi  # -<u' theta> interstitial, G = 1
    c = s.diagnostics.scalar_census("c")
    return dict(err=48.0 * val / (Pe * Pe * D) - 1.0, its=c["krylov_iterations"],
                conv=c["krylov_converged"], pe_h=c["max_cell_peclet"], ubar=ub[2],
                jdiff=float(np.max(np.abs(Jc - J)) / np.max(np.abs(Jc))), J=Jc, phi=phi,
                incompat=c["steady_incompatibility"], s=s)


def gate_g8():
    print("G8 Taylor-Aris: pipe R/h in {16, 32, 64}, face-averaged Poiseuille fluxes (D-WO6-1), insulating walls, "
          "mean gradient e_z, steady; -<u'theta> -> Pe^2 D/48 at Pe = UR/D = 10")
    e = {}
    for Rh in (16, 32, 64):
        rows = [g8_case(Rh, off) for off in OFFSETS]
        e[Rh] = rms([r["err"] for r in rows])
        print(f"  R/h={Rh:3d}  rel err {[f'{r['err']:+.3e}' for r in rows]}  iterations "
              f"{[r['its'] for r in rows]}  census Pe_h {rows[0]['pe_h']:.3f}  Ubar/U - 1 "
              f"{rows[0]['ubar'] - 1.0:+.2e}  |J - J_fields| / |J| <= {max(r['jdiff'] for r in rows):.1e}"
              f"  incompatibility {max(r['incompat'] for r in rows):.1e}")
        for r in rows:
            check(r["conv"], f"R/h={Rh}: converged ({r['its']} iterations; native box: correctness "
                  f"only, ruling D-WO4-1)")
            check(r["jdiff"] <= 1e-10, f"R/h={Rh}: scalar_mean_flux = the field sums to "
                  f"{r['jdiff']:.1e} <= 1e-10")
    for a, b in ((16, 32), (32, 64)):
        check(order(e[a], e[b]) >= 1.8, f"Taylor-Aris order {a}->{b} {order(e[a], e[b]):.2f} >= 1.8 "
              f"(rms {e[a]:.3e} -> {e[b]:.3e})")
    check(e[32] <= 1e-3, f"Taylor-Aris |err| {e[32]:.2e} <= 1e-3 at R/h = 32")
    ex = {Rh: rms([g8_case(Rh, off, exact_flux=False)["err"] for off in OFFSETS]) for Rh in (16, 32)}
    print(f"  INFO face-centre point samples (an O(h^2) flux-quadrature error): rms {ex[16]:.3e} -> "
          f"{ex[32]:.3e}, order {order(ex[16], ex[32]):.2f}")
    print("G-iter singular steady on G8, multigrid-friendly boxes (ruling D-WO4-1): n = 48 / 80 / 144, "
          "nz = 16")
    its = []
    for Rh in (16, 32, 64):
        rows = [g8_case(Rh, off, nbox=mg_box(2 * Rh + 6), nz=mg_box(4)) for off in OFFSETS]
        its.append(max(r["its"] for r in rows))
        print(f"  R/h={Rh:3d}  iterations {[r['its'] for r in rows]}  rel err "
              f"{[f'{r['err']:+.3e}' for r in rows]}")
        for r in rows:
            check(r["conv"], f"R/h={Rh} mg box: converged")
    check(max(its) <= 30, f"G8 singular steady: <= 30 iterations at every rung ({its})")
    gr = [b - a for a, b in zip(its, its[1:])]
    check(all(g <= 5 for g in gr), f"G8 singular steady: growth <= 5 per doubling ({gr})")


def g5b_case(n, off, D=0.6):
    """G5b: the periodic simple-cubic array, solid fraction 0.3, on an n^3 grid of the unit
    period, insulating spheres, mean gradient e_x, steady (singular). k* = -J_x/(D G_x)."""
    L = 1.0
    h = L / n
    s = pf.Solver((n, n, n), extent=(L, L, L))
    s.set_rho(1.0)
    s.set_mu(1.0)
    R = 0.5 * L * (0.3 * 6.0 / math.pi) ** (1.0 / 3.0)
    c0 = np.array([0.5 * L, 0.5 * L, 0.5 * L]) + np.asarray(off) * h
    X, Y, Z = grid(s)
    r = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2)
    s.set_solid(np.asfortranarray(r - R))
    s.add_scalar("c", diffusivity=D, cutcell=True)
    s.set_scalar_mean_gradient("c", (1.0, 0.0, 0.0))
    s.solve_scalar_steady("c")
    J = np.asarray(s.scalar_mean_flux("c"))
    c = s.diagnostics.scalar_census("c")
    Jf, _, _, _ = mean_flux_fields(s, "c", D, (1.0, 0.0, 0.0), (True, True, True))
    return dict(k=-J[0] / D, its=c["krylov_iterations"], conv=c["krylov_converged"],
                lv=c["mg_levels"], nd=2.0 * R / h, jdiff=float(np.max(np.abs(J - Jf)) / abs(J[0])))


def no_solid_case(G, vel=None, D=0.6, n=16):
    """The no-solid identity: a box with no solid (SDF > 0 everywhere), mean gradient G, at rest
    or with a uniform velocity; the closure solution is theta = const and J = -D G exactly."""
    s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_solid(np.asfortranarray(np.ones((n, n, n))), cutcell_pressure=vel is not None)
    if vel is not None:
        for a, nm in enumerate("uvw"):
            s.set_field(nm, np.asfortranarray(np.full((n, n, n), vel[a] *
                                                      s.unit_scales["velocity_to_internal"][a])))
    s.add_scalar("c", diffusivity=D, cutcell=True)
    s.set_field("c", np.asfortranarray(np.random.default_rng(7).uniform(-1, 1, (n, n, n))))
    s.set_scalar_mean_gradient("c", G)
    s.solve_scalar_steady("c")
    J = np.asarray(s.scalar_mean_flux("c"))
    c = s.diagnostics.scalar_census("c")
    return J, c


def gate_g5b():
    print("G5b insulating periodic SC array, c = 0.3, mean gradient e_x: k* = -J_x/(D G_x) via "
          "scalar_mean_flux; ND = 16.6 / 33.2 / 66.4 (n = 20 / 40 / 80, doublings)")
    hs = (1.0 - 0.3) / (1.0 + 0.15)
    k = {}
    for n in (20, 40, 80):
        rows = [g5b_case(n, off) for off in OFFSETS]
        k[n] = [r["k"] for r in rows]
        print(f"  n={n:3d} (ND {rows[0]['nd']:.1f})  k* {[f'{v:.7f}' for v in k[n]]}  iterations "
              f"{[r['its'] for r in rows]}  mg levels {rows[0]['lv']}  |J - J_fields|/|J_x| <= "
              f"{max(r['jdiff'] for r in rows):.1e}")
        for r in rows:
            check(r["conv"], f"n={n}: converged ({r['its']} iterations)")
            if n >= 40:  # ruling D-WO6-2: the bound of the exact k*, gated from ND 32 up
                check(r["k"] <= hs, f"n={n}: k* {r['k']:.6f} <= Hashin-Shtrikman {hs:.6f}")
            check(r["jdiff"] <= 1e-10, f"n={n}: scalar_mean_flux = the field sums ({r['jdiff']:.1e})")
    d1 = rms([a - b for a, b in zip(k[20], k[40])])
    d2 = rms([a - b for a, b in zip(k[40], k[80])])
    p = order(d1, d2)
    check(p >= 1.7, f"self-convergence order {p:.2f} >= 1.7 (rms |k_20 - k_40| {d1:.3e}, "
          f"|k_40 - k_80| {d2:.3e})")
    kx = [c + (c - b) / (2.0 ** p - 1.0) for b, c in zip(k[40], k[80])]
    print(f"  Richardson-extrapolated k* {[f'{v:.6f}' for v in kx]}; ND 16.6 rung "
          f"{max(k[20]) - hs:+.2e} against the bound (INFO: the bound is of the exact k*)")
    check(max(kx) <= hs, f"extrapolated k* {max(kx):.6f} <= Hashin-Shtrikman {hs:.6f}")
    print("G-iter singular steady on the G5b closure, multigrid-friendly n = 32 / 48 / 80 (ruling Q-C)")
    mx = []
    for n in (mg_box(20), mg_box(40), mg_box(80)):
        rows = [g5b_case(n, off) for off in OFFSETS]
        mx.append(max(r["its"] for r in rows))
        print(f"  n={n:3d}  iterations {[r['its'] for r in rows]}  k* {[f'{r['k']:.6f}' for r in rows]}")
        for r in rows:
            check(r["conv"], f"n={n}: converged")
    check(max(mx) <= 30, f"G5b singular: <= 30 iterations at every rung ({mx})")
    gr = [b - a for a, b in zip(mx, mx[1:])]
    check(all(g <= 5 for g in gr), f"G5b singular: growth <= 5 per rung ({gr})")
    print("no-solid identity: k* = 1 to 1e-12")
    J, c = no_solid_case((1.0, 0.0, 0.0))
    ks = -J[0] / 0.6
    print(f"  at rest, G = e_x: k* - 1 = {ks - 1.0:+.1e}, J_y, J_z = {J[1]:+.1e}, {J[2]:+.1e}; "
          f"{c['krylov_iterations']} iterations")
    check(abs(ks - 1.0) <= 1e-12 and abs(J[1]) <= 1e-12 and abs(J[2]) <= 1e-12,
          "no solid, at rest: k* = 1 to 1e-12, transverse flux 0")
    G = np.array([0.3, -0.5, 0.8])
    J, c = no_solid_case(tuple(G), vel=(0.2, 0.1, -0.3))
    dev = float(np.max(np.abs(J + 0.6 * G)) / (0.6 * np.linalg.norm(G)))
    print(f"  uniform flow (0.2, 0.1, -0.3), G = (0.3, -0.5, 0.8): max|J + D G| / (D|G|) = {dev:.1e}; "
          f"census Pe_h {c['max_cell_peclet']:.3g}, {c['krylov_iterations']} iterations")
    check(dev <= 1e-12, "no solid, uniform flow (moving frame): J = -D G to 1e-12")


# -------------------------------------------------------------------------------------- G-adv ----
def rescale_peclet(s, name, target):
    """G-adv's parametrization (A2): one probe steady solve at the present face field, then u, v, w
    times target / (census max_cell_peclet), so the next solve's census reads `target`."""
    s.solve_scalar_steady(name)
    k = target / s.diagnostics.scalar_census(name)["max_cell_peclet"]
    for nm in ("u", "v", "w"):
        s.set_field(nm, np.asfortranarray(s.get_field(nm) * k))


def gadv_budget(s, name):
    """(v), ruling D-WO5c-1: the steady rate balance relative to the GROSS budget — the sum of the
    terms' absolute values, the source taken gross (V sum |r| over the unknowns, r the per-cell
    source density of the right-hand side: kappa s for a per-cell source, the mean-gradient terms
    in closure mode; |source_in| for a uniform one) — bound 1e-11. Returns (that ratio, the ratio
    against the largest NET term: INFO, O(1) for the mean-free closure source of (b) / (b'))."""
    b = s.diagnostics.scalar_budget(name)
    net = max(abs(b["wall_in"]), abs(b["source_in"]), abs(b["boundary_in"]))
    src = abs(b["source_in"])
    if name in GADV_SRC:
        geo = s.diagnostics.scalar_geometry(name)
        unk = geo["unknown"] > 0.5
        x, y, z = s.cell_centers()
        V = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
        src = V * math.fsum(np.abs(GADV_SRC[name])[unk].ravel())
    gross = abs(b["wall_in"]) + src + abs(b["boundary_in"])
    ie = abs(b["identity_error"])
    return ie / gross, (ie / net if net > 0.0 else None)  # closure mode: every net term is 0


def gadv_a_case(Rh, pe):
    """(a) WO-5's C3 problem: a periodic box 4R, a Dirichlet sphere (c = 1) R = 1 with a source,
    D = 0.7, the projected Stokes field of that geometry (20 steps under a body force, mu dt/h^2 =
    1), rescaled to census Pe_h = pe; steady."""
    R, D = 1.0, 0.7
    n = 4 * Rh
    L = 4.0 * R
    h = L / n
    s = pf.Solver((n, n, n), extent=(L, L, L))
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(h * h)
    X, Y, Z = grid(s)
    c0 = np.array([0.5 * L + 0.37 * h, 0.5 * L - 0.22 * h, 0.5 * L + 0.11 * h])
    s.set_solid(np.asfortranarray(np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2) - R),
                cutcell_pressure=True)
    s.set_body_force((1.0, 0.4, 0.2))
    for _ in range(20):
        s.step()
    s.add_scalar("c", diffusivity=D, cutcell=True)
    s.set_scalar_wall("c", "dirichlet", 1.0)
    s.set_scalar_source("c", -0.3)
    rescale_peclet(s, "c", pe)
    s.solve_scalar_steady("c")
    return s


def sc_array(n):
    """G9b's geometry and flow: the periodic simple-cubic sphere array, solid fraction 0.3, one
    sphere per unit period on n^3 cells; Stokes through it from step() (G9b's body force, 50
    steps)."""
    s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(0.01)
    R = (0.3 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    X, Y, Z = grid(s)
    s.set_solid(np.asfortranarray(np.sqrt((X - 0.513) ** 2 + (Y - 0.479) ** 2 + (Z - 0.507) ** 2) - R),
                cutcell_pressure=True)
    s.set_body_force((30.0, 9.0, 0.0))
    for _ in range(50):
        s.step()
    return s


def closure_source(s, name):
    """The B-field source for G = e_x (until WO-6's mean-gradient mode): s = u_x - <u_x> per
    unknown cell, u_x the cell average of its two x-face velocities (physical), <.> the kappa-
    weighted mean over the fluid unknowns; 0 elsewhere."""
    geo = s.diagnostics.scalar_geometry(name)
    unk = geo["unknown"] > 0.5
    kap = geo["kappa"]
    u = s.get_u()
    ux = 0.5 * (u + np.roll(u, -1, axis=0))
    mean = float(np.sum((kap * ux)[unk]) / np.sum(kap[unk]))
    return np.asfortranarray(np.where(unk, ux - mean, 0.0))


def closure_rhs_density(s, name, D, G):
    """The mean-gradient terms of the right-hand side per unit volume (§1.5), physical, per cell:
    the faces' D G_a (a_a^+ - a_a^-)/h_a over the coupled faces, minus G.(U_i/V - kappa_i Ubar);
    the gross scale of (v) in closure mode."""
    geo = s.diagnostics.scalar_geometry(name)
    unk = geo["unknown"] > 0.5
    kap = np.where(unk, geo["kappa"], 0.0)
    h = s.spacing
    U = cell_velocity(s, name)
    ksum = math.fsum(kap[unk].ravel())
    r = np.zeros(unk.shape)
    for a, ap in enumerate((geo["aperture_x"], geo["aperture_y"], geo["aperture_z"])):
        lo = np.where(unk & np.roll(unk, 1, axis=a), ap, 0.0)
        hi = np.roll(lo, -1, axis=a)
        ub = math.fsum(U[a][unk].ravel()) / ksum
        r += D * G[a] * (hi - lo) / h[a] - G[a] * (U[a] - kap * ub)
    return np.where(unk, r, 0.0)


def gadv_b_case(n, pe, wall):
    """(b) the closure problem on the SC array (insulating spheres: steady, singular) in the
    mean-gradient mode of WO-6 (theta, G = e_x), or (b') the reactive bed (Dirichlet spheres,
    c = 0) with the per-cell source u_x - <u_x> (WO-5c's form), D = 0.05; census Pe_h = pe."""
    s = sc_array(n)
    s.add_scalar("c", diffusivity=0.05, cutcell=True)
    if wall == "neumann":
        s.set_scalar_wall("c", "neumann", 0.0)
        s.set_scalar_mean_gradient("c", (1.0, 0.0, 0.0))
        rescale_peclet(s, "c", pe)
        GADV_SRC["c"] = closure_rhs_density(s, "c", 0.05, (1.0, 0.0, 0.0))
        s.solve_scalar_steady("c")
        return s
    s.set_scalar_wall("c", "dirichlet", 0.0)
    s.set_scalar_source("c", closure_source(s, "c"))
    rescale_peclet(s, "c", pe)
    src = closure_source(s, "c")  # the source of the rescaled field
    s.set_scalar_source("c", src)
    geo = s.diagnostics.scalar_geometry("c")
    GADV_SRC["c"] = geo["kappa"] * src
    s.solve_scalar_steady("c")
    return s


def gadv_c_case(pe):
    """(c) G9c's channel (inflow U = 1 / no-slip walls / outflow, a sphere + a cap cutting the
    outlet), the flow developed by 20 NS steps; a steady scalar, Dirichlet 1 at the inlet, Neumann
    elsewhere. The captured open-face flux cannot be rescaled through set_field, so the census
    Pe_h is set through D instead (Pe_h ~ 1/D: the same operator up to a factor)."""
    st = g9c_run(20, scheme="koren", dt=0.008)
    s = st["s"]
    D0 = 0.5
    s.add_scalar("p", diffusivity=D0, cutcell=True)
    s.set_scalar_bc("p", "-x", "dirichlet", 1.0)
    for f in FACES[1:]:
        s.set_scalar_bc("p", f, "neumann")
    s.solve_scalar_steady("p")
    p0 = s.diagnostics.scalar_census("p")["max_cell_peclet"]
    s.add_scalar("d", diffusivity=D0 * p0 / pe, cutcell=True)
    s.set_scalar_bc("d", "-x", "dirichlet", 1.0)
    for f in FACES[1:]:
        s.set_scalar_bc("d", f, "neumann")
    s.solve_scalar_steady("d")
    return s


GADV_PE = (0.1, 1.0, 10.0)
GADV_SRC = {}  # the per-cell source of the last (b) / (b') case, for the gross scale of (v)


def gadv_row(tag, s, name, pe, bound, rate=None):
    """(i), (ii), (v) of one row; returns the iteration count."""
    c = s.diagnostics.scalar_census(name)
    its = c["krylov_iterations"]
    b = s.diagnostics.scalar_budget(name)
    netr = None
    if rate:  # (c): the inflow rate Q c_in, a lower bound of the gross budget (in + out)
        ident = abs(b["identity_error"]) / rate
    else:
        ident, netr = gadv_budget(s, name)
    extra = f" (INFO: {netr:.1e} of the largest net term)" if netr is not None else ""
    print(f"  {tag} Pe_h {c['max_cell_peclet']:.4g}: {its} iterations (<= {bound}), residual "
          f"{c['krylov_residual']:.1e}, {c['mg_levels']} levels; budget identity {ident:.1e}{extra}")
    check(abs(c["max_cell_peclet"] / pe - 1.0) <= 1e-9, f"{tag}: census Pe_h {c['max_cell_peclet']:.6g} = {pe}")
    check(c["krylov_converged"] and c["krylov_residual"] <= 1e-10,
          f"{tag} Pe_h {pe}: (i) converged to rtol 1e-10 within 200 ({its} iterations)")
    check(its <= bound, f"{tag} Pe_h {pe}: (ii) {its} <= {bound} iterations (Q20, D-WOR-8)")
    check(ident <= 1e-11, f"{tag} Pe_h {pe}: (v) steady budget identity {ident:.1e} <= 1e-11 of the gross budget")
    return its


def gadv_growth(tag, lo, hi):
    for pe in GADV_PE:
        g = hi[pe] / lo[pe]
        check(g <= 1.7, f"{tag} Pe_h {pe}: (iii) growth per doubling {lo[pe]} -> {hi[pe]} = {g:.2f}x <= 1.7x")


def gate_gadv():
    print("G-adv: steady advection on the advective surrogate (design Amendment A2), "
          "Pe_h = census max_cell_peclet")
    t0 = time.time()
    # (ii), ruling Q20 / D-WOR-8: min(the provisional bound, 2x the count measured on the WO-R tree,
    # doc/scalar_ibm_log.md 2026-10-04 WO-R) -- tightened, never loosened. Provisional: (a) 20/25/40
    # and 25/35/55, (b) and (b') 15/20/30 at both sizes, (c) 40.
    bounds = {16: (16, 25, 40), 32: (20, 34, 55)}
    its = {}
    for Rh in (16, 32):
        its[Rh] = {}
        for pe, bd in zip(GADV_PE, bounds[Rh]):
            GADV_SRC.clear()
            s = gadv_a_case(Rh, pe)
            its[Rh][pe] = gadv_row(f"(a) R/h={Rh} ({4 * Rh}^3)", s, "c", pe, bd)
    gadv_growth("(a)", its[16], its[32])
    print(f"  [(a): {time.time() - t0:.0f} s]")
    bounds_b = {"neumann": {32: (10, 16, 24), 64: (12, 20, 30)},
                "dirichlet": {32: (15, 20, 30), 64: (15, 20, 30)}}
    for wall, nm in (("neumann", "(b) closure, Neumann spheres, mean gradient e_x"),
                     ("dirichlet", "(b') Dirichlet spheres")):
        t0 = time.time()
        its = {}
        for n in (32, 64):
            its[n] = {}
            for pe, bd in zip(GADV_PE, bounds_b[wall][n]):
                s = gadv_b_case(n, pe, wall)
                if wall == "neumann":
                    check(s.diagnostics.scalar_census("c")["steady_incompatibility"] >= 0.0 and
                          s.diagnostics.scalar_census("c")["mg_levels"] > 1, f"{nm} n={n}: full table")
                its[n][pe] = gadv_row(f"{nm} n={n}", s, "c", pe, bd)
        gadv_growth(nm, its[32], its[64])
        print(f"  [{nm}: {time.time() - t0:.0f} s]")
    t0 = time.time()
    for pe, bd in ((1.0, 16), (10.0, 18)):
        s = gadv_c_case(pe)
        gadv_row("(c) G9c channel, open faces", s, "d", pe, bd, rate=1.0)
    print(f"  [(c): {time.time() - t0:.0f} s]")


# --------------------------------------------------------------------------- conjugate (WO-7) ----
def fmt(v, spec):
    """A list of numbers formatted with `spec` (no nested f-string quotes: Python 3.10)."""
    return "[" + ", ".join(format(x, spec) for x in v) + "]"


def conj_box(n, L):
    """A walled box of physical size L (no flow)."""
    return walled((n, n, n), (L, L, L))


def sealed_check(tag, cen):
    """Ruling D-WO7-1: the sealed solid volume (kappa_s > 0, conjugate, no open solid face and no
    conjugate facet: not a solid unknown) is <= 1e-6 of the conjugate solid volume (as D-WO2-1)."""
    r = cen["sealed_solid_volume"] / cen["solid_volume"] if cen["solid_volume"] > 0 else 0.0
    check(r <= 1e-6, f"{tag}: sealed solid volume {r:.1e} <= 1e-6 of the solid volume")
    return r


def solid_isolated(s, name="c"):
    """Solid unknowns that no solid face and no facet couple: kappa_s > 0, every solid aperture
    1 - a^snap = 0 (all six fluid apertures snapped to 1, hence no facet). Ruling D-WO7-1 makes
    them sealed (not unknowns): this count must be 0."""
    geo = s.diagnostics.scalar_geometry(name)
    us = geo["solid_unknown"] > 0.5
    ax, ay, az = geo["aperture_x"], geo["aperture_y"], geo["aperture_z"]
    allopen = np.ones(us.shape, dtype=bool)
    for a, ap in ((0, ax), (1, ay), (2, az)):
        allopen &= (ap >= 1.0) & (np.roll(ap, -1, axis=a) >= 1.0)
    return us & allopen


def maxwell_fields(X, Y, Z, c0, R, G, ratio):
    """Maxwell's conducting sphere in a uniform gradient G (psi form, Lam_f = 1, Lam_s = ratio):
    outside G.x' (1 + A R^3/r^3), inside B G.x', A = (1 - ratio)/(ratio + 2), B = 3/(ratio + 2),
    x' = x - c0."""
    xs, ys, zs = X - c0[0], Y - c0[1], Z - c0[2]
    r = np.sqrt(xs * xs + ys * ys + zs * zs)
    gx = G[0] * xs + G[1] * ys + G[2] * zs
    A, B = (1.0 - ratio) / (ratio + 2.0), 3.0 / (ratio + 2.0)
    out = gx * (1.0 + A * R**3 / np.maximum(r, 1e-300) ** 3)
    return np.where(r >= R, out, B * gx), B * gx, out


def g4_case(Rh, off, ratio, K=1.0, rtol=None, G=(0.3, -0.5, 0.8)):
    """G4: a conjugate sphere R = 1 (Lam_s = ratio, C_s = 1, partition K) in a box 6R, Lam_f = D =
    1, the exact exterior field as the box Dirichlet profile, steady."""
    R, D = 1.0, 1.0
    h = R / Rh
    n = 6 * Rh
    L = n * h
    s = conj_box(n, L)
    c0 = np.array([0.5 * L, 0.5 * L, 0.5 * L]) + np.asarray(off) * h
    X, Y, Z = grid(s)
    r = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2)
    s.set_solid(np.asfortranarray(r - R))
    add_cc(s, D, box="neumann")
    for f in FACES:
        P = face_points(s, f)
        _, _, out = maxwell_fields(P[0], P[1], P[2], c0, R, G, ratio)
        s.set_scalar_bc("c", f, "dirichlet", out)
    s.set_scalar_solid("c", diffusivity=ratio * D / K, capacity=1.0, partition=K)
    if rtol is not None:
        s.set_scalar_tolerance("c", rtol)
    s.solve_scalar_steady("c")
    geo = s.diagnostics.scalar_geometry("c")
    cen = s.diagnostics.scalar_census("c")
    return s, X, Y, Z, c0, geo, cen


def interior_gradient(X, Y, Z, psi, mask):
    """Least-squares fit psi = a + g.x over the masked cells; returns g."""
    A = np.stack([np.ones(mask.sum()), X[mask], Y[mask], Z[mask]], axis=1)
    sol, *_ = np.linalg.lstsq(A, psi[mask], rcond=None)
    return sol[1:]


def gate_conj_unit():
    """The k_s = k_f checks (the brief's early unit test): with Lam_s = Lam_f, K = 1, R_c = 0 the
    composite must reproduce the homogeneous solution — a linear field to round-off (§1.6.3, G4(a))
    and, in the closure mode, k* = 1 (theta = const)."""
    print("conjugate unit: Lam_s = Lam_f, K = 1, R_c = 0 reproduces the homogeneous solution")
    G = (0.3, -0.5, 0.8)
    for K in (1.0, 3.0):
        s, X, Y, Z, c0, geo, cen = g4_case(6, OFFSETS[0], 1.0, K=K, rtol=1e-12, G=G)
        ex = G[0] * (X - c0[0]) + G[1] * (Y - c0[1]) + G[2] * (Z - c0[2])
        uf = geo["unknown"] > 0.5
        us = geo["solid_unknown"] > 0.5
        iso = solid_isolated(s)
        ef = float(np.max(np.abs(s.get_field("c") - ex)[uf]))
        cs = s.get_scalar_solid("c")
        es = float(np.max(np.abs(cs / K - ex)[us]))
        scale = float(np.linalg.norm(G)) * 6.0
        print(f"  K={K}: max|psi_f - G.x| {ef:.2e}, max|psi_s - G.x| {es:.2e} (every solid unknown); "
              f"{int(us.sum())} solid unknowns, sealed solid volume {cen['sealed_solid_volume']:.1e} "
              f"of {cen['solid_volume']:.3e}; {cen['krylov_iterations']} it, rungs {cen['probe_rungs']}")
        check(cen["krylov_converged"], f"K={K}: converged")
        check(int(iso.sum()) == 0, f"K={K}: no uncoupled solid unknown (D-WO7-1; {int(iso.sum())})")
        sealed_check(f"K={K}", cen)
        check(ef <= 1e-10 * scale and es <= 1e-10 * scale,
              f"K={K}: linear field reproduced to 1e-10 |G| L in both phases ({ef:.1e}, {es:.1e})")
        check(bool(np.all(np.isnan(cs[~us]))), f"K={K}: get_scalar_solid NaN outside the solid unknowns")
    n, D = 24, 0.6
    s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    R = 0.5 * (0.3 * 6.0 / math.pi) ** (1.0 / 3.0)
    X, Y, Z = grid(s)
    s.set_solid(np.asfortranarray(np.sqrt((X - 0.51) ** 2 + (Y - 0.49) ** 2 + (Z - 0.505) ** 2) - R))
    s.add_scalar("c", diffusivity=D, cutcell=True)
    s.set_scalar_solid("c", diffusivity=D)
    s.set_scalar_mean_gradient("c", (1.0, 0.0, 0.0))
    s.solve_scalar_steady("c")
    J = np.asarray(s.scalar_mean_flux("c"))
    th = s.get_field("c")
    print(f"  closure, SC array c = 0.3: k* - 1 = {-J[0] / D - 1.0:+.2e}, J_y,z = {J[1]:+.1e}, {J[2]:+.1e},"
          f" max|theta| {float(np.max(np.abs(th))):.1e}")
    check(abs(-J[0] / D - 1.0) <= 1e-12 and abs(J[1]) <= 1e-12 and abs(J[2]) <= 1e-12,
          "closure at Lam_s = Lam_f: k* = 1 to 1e-12 (the mean-gradient conjugate terms, §1.5)")


def gate_g4():
    print("G4 Maxwell conjugate sphere (steady), box 6R, R/h in {6, 12, 24}")
    G = (0.3, -0.5, 0.8)
    Gn = float(np.linalg.norm(G))
    e, its = {}, {}
    for ratio in (1e-2, 1.0, 10.0, 1e3):
        e[ratio], its[ratio] = {}, {}
        for Rh in (6, 12, 24):
            rows = []
            for off in OFFSETS:
                s, X, Y, Z, c0, geo, cen = g4_case(Rh, off, ratio, G=G)
                us = geo["solid_unknown"] > 0.5
                full = us & (geo["kappa_solid"] >= 1.0)
                gfit = interior_gradient(X, Y, Z, s.get_field("c_solid"), full)
                Bg = 3.0 / (ratio + 2.0) * np.asarray(G)
                rows.append(dict(err=float(np.linalg.norm(gfit - Bg) / np.linalg.norm(Bg)),
                                 it=cen["krylov_iterations"], conv=cen["krylov_converged"],
                                 rungs=cen["probe_rungs"], iso=int(solid_isolated(s).sum()),
                                 sealed=sealed_check(f"ratio={ratio} R/h={Rh}", cen)))
            e[ratio][Rh] = rms([r["err"] for r in rows])
            its[ratio][Rh] = [r["it"] for r in rows]
            print(f"  ratio={ratio:g} R/h={Rh:3d}  interior gradient rel err {fmt([r['err'] for r in rows], '.3e')}"
                  f"  iters {its[ratio][Rh]}  rungs {rows[0]['rungs']}  isolated solid {[r['iso'] for r in rows]}")
            for r in rows:
                check(r["conv"], f"ratio={ratio} R/h={Rh}: converged ({r['it']} iterations)")
                check(r["it"] <= 30, f"ratio={ratio} R/h={Rh}: G-iter conjugate <= 30 ({r['it']})")
        if ratio == 1.0:  # (a): the composite is exact here (a linear field), so no order exists
            check(max(e[ratio].values()) <= 1e-9,
                  f"ratio=1: interior gradient exact to round-off ({max(e[ratio].values()):.1e} <= 1e-9)")
            continue
        o1, o2 = order(e[ratio][6], e[ratio][12]), order(e[ratio][12], e[ratio][24])
        check(o1 >= 1.7 and o2 >= 1.7, f"ratio={ratio}: interior-gradient orders {o1:.2f}, {o2:.2f} >= 1.7")
        check(e[ratio][24] <= 5e-3, f"ratio={ratio}: |err| {e[ratio][24]:.2e} <= 5e-3 at R/h = 24 (prov.)")
    for ratio in (1e-2, 1e3):
        for Rh in (6, 12, 24):
            b = 2 * max(its[1.0][Rh]) + 5
            check(max(its[ratio][Rh]) <= b,
                  f"(d) ratio={ratio} R/h={Rh}: iterations {max(its[ratio][Rh])} <= 2 x ratio-1 + 5 = {b}")
    print("G4(c) K = 3 with the same Lam_s: psi identical, c_s = 3 psi_s")
    for ratio in (10.0,):
        s1, X, Y, Z, c0, geo, _ = g4_case(12, OFFSETS[0], ratio, K=1.0, rtol=1e-13, G=G)
        s3, _, _, _, _, _, _ = g4_case(12, OFFSETS[0], ratio, K=3.0, rtol=1e-13, G=G)
        us = geo["solid_unknown"] > 0.5
        iso = solid_isolated(s1)
        p1, p3 = s1.get_field("c_solid"), s3.get_field("c_solid")
        d = float(np.max(np.abs(p1 - p3)[us & ~iso]) / np.max(np.abs(p1)[us]))
        df = float(np.max(np.abs(s1.get_field("c") - s3.get_field("c"))) / np.max(np.abs(s1.get_field("c"))))
        c3 = s3.get_scalar_solid("c")
        k3 = float(np.max(np.abs(c3[us] - 3.0 * p3[us])))
        print(f"  ratio={ratio}: max|psi_s(K=3) - psi_s(K=1)| {d:.1e}, fluid {df:.1e} (rel); "
              f"max|c_s - 3 psi_s| {k3:.1e}")
        check(d <= 1e-9 and df <= 1e-9, f"K = 3: psi identical to K = 1 within 1e-9 ({d:.1e}, {df:.1e})")
        check(k3 == 0.0, "K = 3: get_scalar_solid = 3 psi_s exactly")


def a_t_keff_sc(c):
    """k* of perfectly conducting spheres on the simple-cubic lattice, Andrianov & Topol eq. 157
    (L3 §2 B6)."""
    a1, a2, a3, a4, a5, a6 = 1.305, 0.231, 0.405, 0.0723, 0.153, 0.0105
    den = (-1.0 + c + a1 * c ** (10 / 3) * (1.0 + a2 * c ** (11 / 3)) / (1.0 - a3 * c ** (7 / 3)) +
           a4 * c ** (14 / 3) + a5 * c**6 + a6 * c ** (22 / 3))
    return 1.0 - 3.0 * c / den


def g5a_case(n, off, D=0.6, ratio=1e4):
    L = 1.0
    h = L / n
    s = pf.Solver((n, n, n), extent=(L, L, L))
    s.set_rho(1.0)
    s.set_mu(1.0)
    R = 0.5 * L * (0.3 * 6.0 / math.pi) ** (1.0 / 3.0)
    c0 = np.array([0.5 * L, 0.5 * L, 0.5 * L]) + np.asarray(off) * h
    X, Y, Z = grid(s)
    r = np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2)
    s.set_solid(np.asfortranarray(r - R))
    s.add_scalar("c", diffusivity=D, cutcell=True)
    s.set_scalar_solid("c", diffusivity=ratio * D)
    s.set_scalar_mean_gradient("c", (1.0, 0.0, 0.0))
    s.solve_scalar_steady("c")
    J = np.asarray(s.scalar_mean_flux("c"))
    c = s.diagnostics.scalar_census("c")
    return dict(k=-J[0] / D, its=c["krylov_iterations"], conv=c["krylov_converged"], nd=2.0 * R / h,
                iso=int(solid_isolated(s).sum()), res=c["krylov_residual"],
                sealed=sealed_check(f"G5a n={n}", c))


def gate_g5a():
    ref = a_t_keff_sc(0.3)
    print(f"G5a conducting periodic SC array, c = 0.3, Lam_s/Lam_f = 1e4, mean gradient e_x; "
          f"reference k* (Andrianov-Topol eq. 157, perfect conductor) = {ref:.5f}")
    k, its = {}, {}
    for n in (20, 40, 80):
        rows = [g5a_case(n, off) for off in OFFSETS]
        k[n] = float(np.mean([r["k"] for r in rows]))
        its[n] = [r["its"] for r in rows]
        print(f"  n={n} (ND {rows[0]['nd']:.1f})  k* {fmt([r['k'] for r in rows], '.6f')}  "
              f"rel err vs ref {k[n] / ref - 1.0:+.2e}  iters {its[n]}  residual {fmt([r['res'] for r in rows], '.1e')}"
              f"  isolated solid {[r['iso'] for r in rows]}")
        for r in rows:
            check(r["conv"], f"n={n}: converged ({r['its']} iterations)")
            check(r["its"] <= 30, f"n={n}: G-iter singular steady <= 30 ({r['its']})")
    # ruling D-WO7-2: (i) order >= 1.7 over the ladder, (ii) the Richardson extrapolate (the
    # ladder's own order) within 5e-4 of the reference (orchestrator 2026-10-04: the reference is the
    # perfectly conducting limit while the run uses contrast 1e4, an offset of ~2e-4, plus a 1-2e-4
    # spread between the fitted-order and order-2 extrapolates; 1e-4 was tighter than the reference
    # itself), (iii) a regression bound of 2x the value
    # measured after D-WO7-1 at each rung (the provisional 3e-3 at ND 32 is retired; log WO-7)
    p = math.log2(abs(k[20] - k[40]) / abs(k[40] - k[80])) if k[40] != k[80] else float("inf")
    check(p >= 1.7, f"(i) self-convergence order {p:.2f} >= 1.7")
    kx = k[80] + (k[80] - k[40]) / (2.0**p - 1.0)
    check(abs(kx / ref - 1.0) <= 5e-4,
          f"(ii) extrapolate {kx:.6f}: |k*/ref - 1| {abs(kx / ref - 1.0):.2e} <= 5e-4 (order-2 extrapolate "
          f"{k[80] + (k[80] - k[40]) / 3.0:.6f}, INFO)")
    for n, b in ((20, 6.64e-2), (40, 1.82e-2), (80, 4.86e-3)):  # 2 x 3.32e-2, 9.09e-3, 2.43e-3
        check(abs(k[n] / ref - 1.0) <= b, f"(iii) n={n}: |k*/ref - 1| {abs(k[n] / ref - 1.0):.2e} <= {b:.2e}")


def g6_exact(Ls, CK, Rc, R=1.0, Ro=2.0):
    """The slowest decay rate mu and the mode (P, B, C) of the composite sphere in the psi form
    (Lam_f = C_f = 1): solid r < R, psi_s = P sin(k_s r)/r; fluid R < r < Ro, psi_f = (B sin(k_f r)
    + C cos(k_f r))/r; psi_f(Ro) = 0, Lam_s psi_s' = psi_f' and psi_s - psi_f + R_c Lam_s psi_s' = 0
    at R (the l = 0 analogue of conj.py's exact())."""
    def mat(mu):
        ks, kf = math.sqrt(CK * mu / Ls), math.sqrt(mu)
        S = lambda k, r: math.sin(k * r) / r
        C = lambda k, r: math.cos(k * r) / r
        dS = lambda k, r: (k * math.cos(k * r) * r - math.sin(k * r)) / (r * r)
        dC = lambda k, r: (-k * math.sin(k * r) * r - math.cos(k * r)) / (r * r)
        return np.array([[0.0, S(kf, Ro), C(kf, Ro)],
                         [Ls * dS(ks, R), -dS(kf, R), -dC(kf, R)],
                         [S(ks, R) + Rc * Ls * dS(ks, R), -S(kf, R), -C(kf, R)]])
    det = lambda m: float(np.linalg.det(mat(m)))
    mus = np.linspace(1e-4, 40.0, 40000)
    d = [det(m) for m in mus]
    for i in range(len(mus) - 1):
        if d[i] * d[i + 1] < 0:
            a, b = mus[i], mus[i + 1]
            for _ in range(200):
                m = 0.5 * (a + b)
                if det(a) * det(m) <= 0:
                    b = m
                else:
                    a = m
            mu = 0.5 * (a + b)
            _, _, vt = np.linalg.svd(mat(mu))
            return mu, vt[-1], math.sqrt(CK * mu / Ls), math.sqrt(mu)
    raise RuntimeError("no root")


G6_CASES = ((10.0, 1.0, 1.0, 0.0), (0.1, 1.0, 1.0, 0.0), (100.0, 0.5, 1.0, 0.0), (3.0, 3.0, 3.0, 0.0),
            (1.0, 1.0, 1.0, 0.2), (5.0, 1.0, 0.5, 0.5))


# D-WO7-2 (iii): 2x the RMS error measured after D-WO7-1 (log WO-7), per case and rung
G6_BOUND = ({8: 1.48e-2, 16: 3.77e-3, 32: 9.56e-4}, {8: 3.37e-2, 16: 8.45e-3, 32: 2.14e-3},
            {8: 1.62e-2, 16: 4.16e-3, 32: 1.06e-3}, {8: 3.49e-3, 16: 9.12e-4, 32: 2.37e-4},
            {8: 7.55e-3, 16: 2.01e-3, 32: 5.16e-4}, {8: 7.41e-3, 16: 1.90e-3, 32: 4.82e-4})


def g6_case(Rh, off, case, nsteps=30):
    """G6: a conjugate core R = 1 (instance 0) inside an immersed Dirichlet sphere Ro = 2 (instance
    1, psi = 0); psi form (Lam_s, C_s K, K, R_c), Lam_f = C_f = 1; BE late-time ratio of the total
    mass (fluid + solid) with dt mu = 1, the exact mode as the initial condition."""
    Ls, CK, K, Rc = case
    R, Ro = 1.0, 2.0
    h = R / Rh
    n = int(math.ceil(2.0 * Ro / h)) + 6
    L = n * h
    s = walled((n, n, n), (L, L, L))
    c0 = np.array([0.5 * L, 0.5 * L, 0.5 * L]) + np.asarray(off) * h
    s.set_scene(*two_body_scene(c0, R, Ro), periodic=False)
    s.set_solid_from_scene(cutcell_pressure=False)
    mu, (P, B, C), ks, kf = g6_exact(Ls, CK, Rc)
    dt = 1.0 / mu
    s.set_dt(dt)
    add_cc(s, 1.0, box="neumann")
    s.set_scalar_wall("c", "dirichlet", 0.0, instance=1)
    s.set_scalar_solid("c", diffusivity=Ls / CK, capacity=CK / K, partition=K, contact_resistance=Rc,
                       instance=0)
    X, Y, Z = grid(s)
    r = np.maximum(np.sqrt((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2), 1e-12)
    s.set_field("c", np.asfortranarray((B * np.sin(kf * r) + C * np.cos(kf * r)) / r))
    s.set_field("c_solid", np.asfortranarray(P * np.sin(ks * r) / r))
    masses, mfl, its, idmax = [], [], [], 0.0
    iso = int(solid_isolated(s).sum())
    for _ in range(nsteps):
        s.advance_scalars()
        b = s.diagnostics.scalar_budget("c")
        masses.append(b["mass"] + b["mass_solid"])
        mfl.append(b["mass"])
        idmax = max(idmax, abs(b["identity_error"]) / max(abs(b["d_mass"]), 1e-300))
        cen = s.diagnostics.scalar_census("c")
        its.append(cen["krylov_iterations"])
        if not cen["krylov_converged"]:
            return dict(err=float("nan"), its=its, conv=False, idrel=idmax, rungs=cen["probe_rungs"])
    mu_h = (masses[-2] / masses[-1] - 1.0) / dt
    mu_f = (mfl[-2] / mfl[-1] - 1.0) / dt  # INFO: the fluid mass alone (same mode, no solid cells)
    sealed_check(f"G6 case {G6_CASES.index(case) + 1} R/h={Rh}", cen)
    return dict(err=mu_h / mu - 1.0, errf=mu_f / mu - 1.0, its=its, conv=True, idrel=idmax,
                rungs=cen["probe_rungs"], nsolid=cen["num_solid_unknowns"], iso=iso)


def gate_g6(cases=None):
    print("G6 transient composite sphere (conjugate core R, immersed Dirichlet sphere 2R), "
          "BE late-time ratio, dt mu = 1, 30 steps; R/h in {8, 16, 32}")
    for ci, case in enumerate(G6_CASES):
        if cases is not None and ci not in cases:
            continue
        mu = g6_exact(case[0], case[1], case[3])[0]
        e = {}
        for Rh in (8, 16, 32):
            t0 = time.time()
            rows = [g6_case(Rh, off, case) for off in OFFSETS]
            e[Rh] = rms([r["err"] for r in rows])
            print(f"  case {ci + 1} (Lam_s, CK, K, Rc) = {case} mu = {mu:.6f}  R/h={Rh:3d}  "
                  f"mu rel err {fmt([r['err'] for r in rows], '+.3e')}  (INFO fluid mass: "
                  f"{fmt([r.get('errf', float('nan')) for r in rows], '+.3e')}; isolated solid "
                  f"{[r.get('iso') for r in rows]})  identity/|dM| <= "
                  f"{max(r['idrel'] for r in rows):.1e}  iters/step {min(min(r['its']) for r in rows)}.."
                  f"{max(max(r['its']) for r in rows)}  rungs {rows[0]['rungs']}  ({time.time() - t0:.0f} s)")
            for r in rows:
                check(r["conv"], f"case {ci + 1} R/h={Rh}: every step converged")
                check(r["idrel"] <= 1e-13, f"case {ci + 1} R/h={Rh}: budget identity {r['idrel']:.1e} <= 1e-13 |d_mass|")
                check(max(r["its"]) <= 30, f"case {ci + 1} R/h={Rh}: G-iter conjugate <= 30 ({max(r['its'])})")
        o1, o2 = order(e[8], e[16]), order(e[16], e[32])
        check(o1 >= 1.7 and o2 >= 1.7, f"case {ci + 1}: (i) orders {o1:.2f}, {o2:.2f} >= 1.7")
        # ruling D-WO7-2 (iii): a regression bound of 2x the value measured after D-WO7-1, at every
        # rung (the provisional 2e-3 at R/h 16 is retired; old and measured values in the log)
        for Rh in (8, 16, 32):
            b = G6_BOUND[ci][Rh]
            check(e[Rh] <= b, f"case {ci + 1}: (iii) |err| {e[Rh]:.3e} <= {b:.2e} at R/h = {Rh}")


def gate_gadv_conj():
    """WO-7's extra G-adv row (design A2): G-adv(b) with conducting spheres, Lam_s/Lam_f = 10, K = 1,
    at Pe_h 1 and 10 (the mean-gradient mode, G = e_x): converged, in <= 1.5x the insulating (b)
    count of the same field."""
    print("G-adv(b) conjugate: conducting spheres Lam_s/Lam_f = 10 vs insulating, Pe_h 1 and 10")
    for n in (32, 64):
        for pe in (1.0, 10.0):
            si = gadv_b_case(n, pe, "neumann")
            ci = si.diagnostics.scalar_census("c")
            s = sc_array(n)
            s.add_scalar("c", diffusivity=0.05, cutcell=True)
            s.set_scalar_solid("c", diffusivity=0.5)
            s.set_scalar_mean_gradient("c", (1.0, 0.0, 0.0))
            rescale_peclet(s, "c", pe)
            s.solve_scalar_steady("c")
            cc = s.diagnostics.scalar_census("c")
            b = s.diagnostics.scalar_budget("c")
            print(f"  n={n} Pe_h {cc['max_cell_peclet']:.3g}: conjugate {cc['krylov_iterations']} it "
                  f"(converged {cc['krylov_converged']}, residual {cc['krylov_residual']:.1e}), insulating "
                  f"{ci['krylov_iterations']} it; identity_error {b['identity_error']:.1e}")
            check(cc["krylov_converged"], f"n={n} Pe_h={pe}: conjugate converged")
            check(cc["krylov_iterations"] <= 1.5 * ci["krylov_iterations"],
                  f"n={n} Pe_h={pe}: {cc['krylov_iterations']} <= 1.5 x {ci['krylov_iterations']}")


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


def raises_naming(exc, fn, needles, msg):
    """`raises`, and the message names every one of `needles` (the remedy)."""
    try:
        fn()
    except exc as e:
        missing = [w for w in needles if w not in str(e)]
        check(not missing, f"{msg}" + (f" (message lacks {missing}: {e})" if missing else ""))
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
    # dt D/h^2 = 14.4 -> kappa_A > 13: the full ScalarMG table, 12 -> 6 -> 3
    check(c["krylov_converged"] and c["mg_levels"] == 3,
          f"one step converged ({c['krylov_iterations']} iterations, {c['mg_levels']} MG levels)")
    s.set_scalar_bc("c", "-x", "dirichlet", 0.5)  # the flow -x face is periodic
    raises(RuntimeError, s.advance_scalars, "a scalar Dirichlet face against a periodic flow face is refused")
    s.set_scalar_bc("c", "-x", "periodic")
    raises(ValueError, lambda: s.set_scalar_mean_gradient("c", (float("nan"), 0.0, 0.0)),
           "a non-finite mean gradient is a ValueError")
    raises(ValueError, lambda: s.set_scalar_mean_gradient("legacy", (1.0, 0.0, 0.0)),
           "set_scalar_mean_gradient on a legacy scalar is a ValueError")
    s.set_scalar_mean_gradient("c", (1.0, 0.0, 0.0))
    s.advance_scalars()  # periodic box: the closure mode steps
    sw = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
    sw.set_rho(1.0)
    sw.set_mu(1.0)
    sw.set_domain_bc("-y", "wall")
    sw.set_domain_bc("+y", "wall")
    sw.set_solid(np.asfortranarray(np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) - 0.2))
    sw.add_scalar("c", 1.0, cutcell=True)
    sw.set_scalar_bc("c", "-y", "neumann")
    sw.set_scalar_bc("c", "+y", "neumann")
    sw.set_scalar_mean_gradient("c", (1.0, 0.0, 0.5))  # x, z periodic: accepted
    sw.solve_scalar_steady("c")
    sw.set_scalar_mean_gradient("c", (0.0, 1.0, 0.0))
    raises_naming(RuntimeError, lambda: sw.solve_scalar_steady("c"), ("along y", "-y"),
                  "a mean gradient along a non-periodic axis is refused, naming the axis and face")
    s.set_scalar_mean_gradient("c", (0.0, 0.0, 0.0))
    s.set_porous_continuity(True)
    raises(RuntimeError, s.advance_scalars, "porous continuity is refused")
    # ruling D-WO5-2: a moving fluid without the cut-cell projection is refused, naming the remedy
    # (a fluid at rest keeps WO-3/WO-4's behaviour: the periodic box above stepped)
    s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(0.1)
    X, Y, Z = grid(s)
    s.set_solid(np.asfortranarray(np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) - 0.2))
    s.add_scalar("c", 1.0, cutcell=True)
    s.set_field("u", np.asfortranarray(np.full((n, n, n), 0.5)))
    raises_naming(RuntimeError, s.advance_scalars, ("cutcell_pressure=True", "set_pressure_geometry"),
                  "a moving fluid without cutcell_pressure is refused, naming the remedy (D-WO5-2)")
    # ruling D-WO5-4: the ghost projection is refused, the message listing the accepted ones
    s = pf.SolverColocated((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(0.1)
    s.set_collocated_scheme("ghost")
    s.set_solid(np.asfortranarray(np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) - 0.2),
                cutcell_pressure=True)
    s.add_scalar("c", 1.0, cutcell=True)
    raises_naming(RuntimeError, s.advance_scalars,
                  ("'gauge-exact'", "'plain'", "'embed'", "staggered Solver"),
                  "the 'ghost' scheme is refused, listing gauge-exact, plain, embed and the staggered "
                  "Solver (D-WO5-4)")
    # review finding 1 (ruling D-WOR-1): collocated, a raw set_field('u') never builds the projected
    # face field, so a moving fluid before the first step() is refused, naming the remedy (the
    # reviewer's probe: it silently returned the pure-diffusion solution); seeded through
    # set_velocity, the same case advects
    for seed in ("set_field", "set_velocity"):
        s = pf.SolverColocated((n, n, n), extent=(1.0, 1.0, 1.0))
        s.set_rho(1.0)
        s.set_mu(1.0)
        s.set_dt(0.1)
        s.set_collocated_scheme("gauge-exact")
        s.set_solid(np.asfortranarray(np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) - 0.2),
                    cutcell_pressure=True)
        s.add_scalar("c", 1.0, cutcell=True)
        s.set_scalar_wall("c", "dirichlet", 1.0)
        if seed == "set_field":
            s.set_field("u", np.asfortranarray(np.full((n, n, n), 0.5)))
            for fn, nm in ((lambda: s.solve_scalar_steady("c"), "solve_scalar_steady"),
                           (s.advance_scalars, "advance_scalars")):
                raises_naming(RuntimeError, fn, ("FACE field", "set_velocity", "step()"),
                              f"collocated {nm} with an unbuilt face field and a moving fluid is "
                              "refused, naming the remedy (D-WOR-1)")
        else:
            s.set_velocity(0, np.asfortranarray(np.full((n, n, n), 0.5)))
            s.solve_scalar_steady("c")
            cen = s.diagnostics.scalar_census("c")
            check(cen["max_cell_peclet"] > 0.0 and cen["num_flux_faces"] > 0,
                  f"collocated, seeded by set_velocity: the steady solve advects (Pe_h "
                  f"{cen['max_cell_peclet']:.3g}, {cen['num_flux_faces']} flux faces)")
    # review finding 4 (ruling D-WOR-4): a level table that stops above the pressure's bottom extent
    # (an odd box cannot halve: level 0 alone, 15^3) warns once per scalar; a factor-of-two box not
    for m, want in ((15, True), (16, False)):
        s = pf.Solver((m, m, m), extent=(1.0, 1.0, 1.0))
        Xm, Ym, Zm = grid(s)
        s.set_solid(np.asfortranarray(np.sqrt((Xm - 0.5) ** 2 + (Ym - 0.5) ** 2 + (Zm - 0.5) ** 2) - 0.2))
        s.add_scalar("c", 1.0, cutcell=True)
        s.set_scalar_wall("c", "dirichlet", 1.0)
        _, err = capture_stderr(lambda: s.solve_scalar_steady("c"))
        _, err2 = capture_stderr(lambda: s.solve_scalar_steady("c"))
        got = "ScalarMG level table stops" in err
        check(got == want and "ScalarMG" not in err2,
              f"{m}^3 steady: the shallow-bottom warning is {'printed once' if want else 'absent'} "
              f"(D-WOR-4; {err.strip()[:90]!r})")
    # review finding 7 (ruling D-WOR-7): koren above bulk Courant 1/2 warns once (fou never); a
    # rebuilt geometry re-issues the §9 resolution warnings (a 0.4 h plate: thin solid)
    for scheme in ("koren", "fou"):
        s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
        s.set_dt(0.05)
        s.set_solid(np.asfortranarray(np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) - 0.2),
                    cutcell_pressure=True)
        s.add_scalar("c", 0.01, scheme=scheme, cutcell=True)
        s.set_field("u", np.asfortranarray(np.full((n, n, n), 0.75)))
        _, err = capture_stderr(s.advance_scalars)
        _, err2 = capture_stderr(s.advance_scalars)
        cb = s.diagnostics.scalar_census("c")["bulk_courant"]
        want = scheme == "koren"
        check(cb > 0.5 and ("'koren' at bulk Courant" in err) == want and "Courant" not in err2,
              f"{scheme} at bulk Courant {cb:.3f}: the koren warning is "
              f"{'printed once' if want else 'absent'} (D-WOR-7)")
    h = 1.0 / n
    plate = np.asfortranarray(np.abs(X - (n // 2 + 0.55) * h) - 0.2 * h)
    s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_solid(plate)
    s.add_scalar("c", 1.0, cutcell=True)
    errs = [capture_stderr(lambda: s.solve_scalar_steady("c"))[1]]
    errs.append(capture_stderr(lambda: s.solve_scalar_steady("c"))[1])
    s.set_solid(plate)
    errs.append(capture_stderr(lambda: s.solve_scalar_steady("c"))[1])
    check(["thin-solid" in e for e in errs] == [True, False, True],
          f"the thin-solid warning: once per geometry, again after set_solid (D-WOR-7; "
          f"{['thin-solid' in e for e in errs]})")
    # WO-7 (§8.1): set_scalar_solid's validation, the registered solid field, get_scalar_solid
    s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(0.1)
    X, Y, Z = grid(s)
    s.set_solid(np.asfortranarray(np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) - 0.2))
    s.add_scalar("legacy", 1.0)
    s.add_scalar("c", 1.0, cutcell=True)
    raises(ValueError, lambda: s.set_scalar_solid("legacy", 1.0), "set_scalar_solid on a legacy scalar")
    for kw, nm in (({"diffusivity": -1.0}, "diffusivity < 0"), ({"capacity": 0.0}, "capacity = 0"),
                   ({"partition": 0.0}, "partition = 0"), ({"contact_resistance": -1.0}, "R_c < 0"),
                   ({"instance": 0}, "instance= without a scene")):
        args = {"diffusivity": 1.0}
        args.update(kw)
        raises(ValueError, lambda a=args: s.set_scalar_solid("c", **a), f"set_scalar_solid: {nm} is a ValueError")
    check("c_solid" not in s.field_names(), "no solid field before set_scalar_solid")
    s.set_scalar_solid("c", 2.0, capacity=0.5, partition=3.0)
    check("c_solid" in s.field_names(), "set_scalar_solid registers 'c_solid'")
    check(bool(np.all(np.isnan(s.get_scalar_solid("c")))), "get_scalar_solid before a solve: all NaN")
    s.set_field("c", np.asfortranarray(np.ones((n, n, n))))
    s.set_field("c_solid", np.asfortranarray(np.ones((n, n, n))))  # psi = 1 in both: equilibrium
    s.advance_scalars()
    c = s.diagnostics.scalar_census("c")
    cs = s.get_scalar_solid("c")
    geo = s.diagnostics.scalar_geometry("c")
    us = geo["solid_unknown"] > 0.5
    check(c["krylov_converged"] and c["num_solid_unknowns"] == int(us.sum()) > 0,
          f"a conjugate step converged ({c['krylov_iterations']} it, {c['num_solid_unknowns']} solid unknowns)")
    check(float(np.max(np.abs(cs[us] - 3.0))) <= 1e-9 and float(np.max(np.abs(s.get_field("c")[geo["unknown"] > 0.5] - 1.0))) <= 1e-9,
          "psi = 1 in both phases is an equilibrium: c = 1, c_s = K = 3")
    s.set_scalar_wall("c", "neumann", 0.0)  # every body a wall again: the single-phase path
    s.advance_scalars()
    check(s.diagnostics.scalar_census("c")["num_solid_unknowns"] == 0, "set_scalar_wall(instance=None) ends the conjugate path")


# --------------------------------------------------------------------------------------- G13 ----
G13_CASES = ("gap", "contact", "conj")


def sphere_pair_scene(c1, c2, R):
    """Two sphere instances (body ids 0 and 1) of radius R centred at c1, c2."""
    ni = np.array([1, -1, -1], dtype=np.int32)
    nr = np.zeros(16)
    nr[0] = R
    nr[14] = nr[15] = 1.0  # rotation w, scale
    ii = np.array([0, -1, 0, -1], dtype=np.int32)
    ir = np.zeros(2 * 18)
    for k, c in enumerate((c1, c2)):
        ir[18 * k:18 * k + 3] = c
        ir[18 * k + 6] = 1.0
        ir[18 * k + 7] = 1.0
    return ni, nr, ii, ir


def g13_case(Rh, off, case):
    """§11 G13: two spheres R = 1 along x in a box (6R, 4R, 4R) with Dirichlet box faces, steady.
    'gap' / 'contact': Dirichlet spheres c = 1 at a surface gap R/32 / an overlap R/64, far field 0;
    the functional is the total wall flux into the fluid. 'conj': the contacting pair conjugate at
    Lam_s/Lam_f = 100 (K = 1), the box faces at the far-field profile c = x - x_c (the pair alone at
    rest would carry no flux); the functional is the heat entering through the -x box face,
    Q = sum 2 D (g - c_0) h over its face cells (the operator's own boundary row; no solid touches
    the box faces)."""
    R, D = 1.0, 0.7
    h = R / Rh
    cells = (6 * Rh, 4 * Rh, 4 * Rh)
    L = (6.0 * R, 4.0 * R, 4.0 * R)
    s = walled(cells, L)
    xc = np.array([0.5 * L[0], 0.5 * L[1], 0.5 * L[2]]) + np.asarray(off) * h
    d = 2.0 * R + (R / 32.0 if case == "gap" else -R / 64.0)
    c1, c2 = xc - [0.5 * d, 0.0, 0.0], xc + [0.5 * d, 0.0, 0.0]
    s.set_scene(*sphere_pair_scene(c1, c2, R), periodic=False)
    s.set_solid_from_scene(cutcell_pressure=False)
    if case == "conj":
        add_cc(s, D, box="neumann")
        for f in FACES:
            P = face_points(s, f)
            s.set_scalar_bc("c", f, "dirichlet", P[0] - xc[0])
        s.set_scalar_solid("c", diffusivity=100.0 * D)
    else:
        add_cc(s, D, box="dirichlet")
        s.set_scalar_wall("c", "dirichlet", 1.0)
    s.solve_scalar_steady("c")
    cen = s.diagnostics.scalar_census("c")
    if case == "conj":
        P = face_points(s, "-x")
        c = s.get_field("c")
        F = float(np.sum(2.0 * D * ((P[0] - xc[0]) - c[0, :, :]) * h))
    else:
        F = float(np.sum(s.scalar_wall_flux("c")))
    b = s.diagnostics.scalar_budget("c")
    return dict(F=F, it=cen["krylov_iterations"], conv=cen["krylov_converged"], cen=cen, b=b,
                finite=bool(np.all(np.isfinite(s.get_field("c")[s.diagnostics.scalar_geometry("c")["unknown"] > 0.5]))))


def g13_rungs(cen):
    """(R2 fraction, rungs) over both phases' probes (the solid tuple on a conjugate scalar)."""
    pr = cen["probe_rungs"]
    tot = sum(pr["fluid"]) + (sum(pr["solid"]) if "solid" in pr else 0)
    r2 = pr["fluid"][3] + (pr["solid"][3] if "solid" in pr else 0)
    return (r2 / tot if tot else 0.0), pr


def g13_thin_plate():
    """§11 G13 thin plate: a solid slab 0.4 h thick across the box, its mid-plane 0.05 h off a
    cell-centre plane (a slab between two centre planes leaves every PL sample positive); the census
    counts thin-solid cells and the §9 warning is printed (stderr, captured at the fd)."""
    import os
    import tempfile
    n, L = 16, 1.0
    h = L / n
    s = walled((n, n, n), (L, L, L))
    X, _, _ = grid(s)
    x0 = (n // 2 + 0.5) * h + 0.05 * h
    s.set_solid(np.asfortranarray(np.abs(X - x0) - 0.2 * h))
    add_cc(s, 0.7, box="dirichlet")
    s.set_scalar_wall("c", "dirichlet", 1.0)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        sys.stderr.flush()
        fd = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            s.solve_scalar_steady("c")
        finally:
            os.dup2(fd, 2)
            os.close(fd)
        tmp.seek(0)
        err = tmp.read().decode(errors="replace")
    cen = s.diagnostics.scalar_census("c")
    return cen, err


def gate_g13(cases=G13_CASES, rungs=(8, 16, 32)):
    print("G13 contacts: two spheres R = 1 in a box (6R, 4R, 4R), R/h in {8, 16, 32}, steady")
    for case in cases:
        # ruling D-WO8-2: the conjugate contact row is INFO (its discrete solid bridge is
        # pre-asymptotic; §13 Q4 triggered); the Dirichlet rows and the thin plate stay gated
        info = case == "conj"
        if info:
            print("  OPEN: contact model pending (§13 Q4, Frank) -- the conjugate contact row is INFO")

        def check(cond, msg, _info=info):
            if _info:
                print(("  info  ok    " if cond else "  info  MISS  ") + msg)
            else:
                globals()["check"](cond, msg)
        F, its = {}, {}
        for Rh in rungs:
            rows = [g13_case(Rh, off, case) for off in OFFSETS]
            F[Rh] = float(np.mean([r["F"] for r in rows]))
            its[Rh] = [r["it"] for r in rows]
            fr = [g13_rungs(r["cen"]) for r in rows]
            print(f"  {case:7s} R/h={Rh:3d}  F = {[f'{r['F']:.6e}' for r in rows]} (mean {F[Rh]:.6e})  iters {its[Rh]}"
                  f"  two-sided {[r['cen']['num_two_sided'] for r in rows]}  thin {[r['cen']['num_thin_solid'] for r in rows]}"
                  f"  rungs {[p for _, p in fr]}")
            for r, (r2, _) in zip(rows, fr):
                check(r["conv"] and r["finite"], f"{case} R/h={Rh}: converged and finite ({r['it']} iterations)")
                check(r["it"] <= 50, f"{case} R/h={Rh}: <= 50 iterations ({r['it']})")
                if case == "gap" and Rh >= 32:  # gap R/32 >= h: no cell holds both walls (reading, WO-8)
                    print(f"  info  gap R/h={Rh}: two-sided {r['cen']['num_two_sided']} (the gap is >= h)")
                else:
                    check(r["cen"]["num_two_sided"] > 0, f"{case} R/h={Rh}: census two-sided > 0 ({r['cen']['num_two_sided']})")
                check(r2 <= 0.01, f"{case} R/h={Rh}: R2 <= 1 % of the probes ({100 * r2:.2f} %)")
        if len(rungs) == 3:
            a, b, c = (F[k] for k in rungs)
            p = order(abs(a - b), abs(b - c))
            check(p >= 1.0, f"{case}: total-flux self-convergence order {p:.2f} >= 1.0 "
                            f"(|F8 - F16| {abs(a - b):.3e}, |F16 - F32| {abs(b - c):.3e})")
    cen, err = g13_thin_plate()
    print(f"  thin plate 0.4h: num_thin_solid {cen['num_thin_solid']}, two-sided {cen['num_two_sided']}, "
          f"rungs {cen['probe_rungs']}; stderr: {err.strip()!r}")
    check(cen["num_thin_solid"] > 0, f"thin plate: num_thin_solid > 0 ({cen['num_thin_solid']})")
    check("thin" in err, "thin plate: the §9 thin-solid warning is printed")


GATES = {"api": gate_api, "g1": gate_g1, "g2": gate_g2, "g3a": gate_g3a, "g3b": gate_g3b, "g7": gate_g7,
         "giter": gate_giter, "g9": gate_g9, "g9b": gate_g9b, "g9c": gate_g9c, "gadv": gate_gadv,
         "g8": gate_g8, "g5b": gate_g5b, "conj_unit": gate_conj_unit, "g4": gate_g4, "g5a": gate_g5a,
         "g6": gate_g6, "gadv_conj": gate_gadv_conj, "g13": gate_g13}
for _ci in range(len(G6_CASES)):  # one ctest per G6 case (each runs the full ladder)
    GATES[f"g6_{_ci + 1}"] = (lambda c=_ci: gate_g6(cases=(c,)))

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
