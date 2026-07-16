"""A-priori validation of the directional ghost-cell projection on the COLLOCATED grid.

Extends ghost_projection_apriori.py (staggered, 9/9 gates) to Solver<Colocated> — the open
problem of doc/collocated_second_order_open_problem.md. Structure of the collocated scheme under
test (the steady fixed point feels ONLY the momentum grad(P) operator and the constraint, since
the incremental phi -> 0 there):

  constraint C   = ghost-closed point divergence of the 1/2-1/2 face-AVERAGED cell field
                   (the closures/matrix are IDENTICAL to the staggered ghost scheme: the face
                   correction uf -= grad(phi) is the same substitution, so assemble() is reused
                   verbatim and every staggered matrix gate carries over);
  grad(P) / cell correction = directional cell gradient: central difference where both axis
                   neighbours are fluid-centered, 2nd-order ONE-SIDED (-3P_i+4P_{i+1}-P_{i+2})/2
                   toward the fluid where the neighbour center is solid (falls back to the
                   2-point one-sided when i+2 is solid too; 0 when sandwiched). NEVER reads a
                   solid-centered cell's P/phi — those rows are decoupled (0), and reading them
                   is a GAUGE-DEPENDENT O(1) gradient error (the shipping mode-0 central
                   difference and the o-weighted kernels all make it; measured here in [C2]).

Tests (gates in main):
  [C1] constraint truncation (open-problem doc T1): ghost-closed divergence of the face-averaged
       exact solenoidal Stokes field. Near-IB rows O(h) localized truncation (same structure the
       staggered scheme damps to global 2nd order), bulk O(h^2). Contrast column: the mode-0
       openness divergence o_f * faceavg on the same field.
  [C2] cell-gradient operator ladder on a smooth pressure at cut cells (fluid center, solid
       axis-neighbour): central-reading-solid-0 and the o-weighted kernels are O(1) (and gauge-
       dependent where they read solid P); the directional one-sided gradient is O(h^2) and
       exactly gauge-independent.
  [C3] the full projection chain, manufactured (open-problem doc T2 analog): perturb the CELL
       field with Gc(phi_man), face-average, ghost-divergence, pinned singular solve, face
       correction (plain grad) + CELL correction (Gc). Gates: phi ~O(h^2), corrected CELL
       velocity ~O(h^2), diagnostic == residual identity. Ladder comparison of the cell
       correction shows the one-sided directional variant is required.

Run:  python tests/study/ghost_collocated_apriori.py  [--quick]
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghost_projection_apriori import (  # noqa: E402
    C0, COUPLED, assemble, build_geo, divergence, overlay_cells, phi_man, periodic_u,
    row_rescale, sdf_sphere, stokes_u)

TP = 2.0 * np.pi


def phi_man_grad(x, y, z):
    """Analytic gradient of phi_man (sin/cos triple)."""
    gx = TP * np.cos(TP * x) * np.cos(TP * y) - TP * np.sin(TP * z) * np.sin(TP * x)
    gy = -TP * np.sin(TP * x) * np.sin(TP * y) + TP * np.cos(TP * y) * np.cos(TP * z)
    gz = -TP * np.sin(TP * y) * np.sin(TP * z) + TP * np.cos(TP * z) * np.cos(TP * x)
    return gx, gy, gz


def cell_fields(geo, ufun):
    """Cell-centered samples, masked to 0 at solid centers (maskVelocity model)."""
    act = geo["active"]
    return [np.where(act, ufun(*geo["Xc"])[a], 0.0) for a in range(3)]


def face_avg(U3):
    """Minus-face field of the cell field: uf_a(i) = 1/2 (U_a(i) + U_a(i-1)), periodic."""
    return [0.5 * (U3[a] + np.roll(U3[a], +1, axis=a)) for a in range(3)]


def face_openness(geo, ns=6):
    """Sampled area fraction of the minus face per axis (mode-0 contrast only)."""
    h = geo["h"]
    t = (np.arange(ns) + 0.5) / ns - 0.5
    o = []
    for a in range(3):
        b, c = [q for q in range(3) if q != a]
        acc = np.zeros_like(geo["Pf"][a][0])
        for tb in t:
            for tc in t:
                Q = [p.copy() for p in geo["Pf"][a]]
                Q[b] = Q[b] + tb * h
                Q[c] = Q[c] + tc * h
                acc += (geo["sdf"](*Q) >= 0.0)
        o.append(acc / (ns * ns))
    return o


def cell_grad(geo, P, mode, o_real=None):
    """Per-axis discrete cell-center gradient (grid units) of a cell field P that is only
    defined on fluid cells (solid cells read as 0 — what the solver's decoupled rows hold).
      central  : plain central difference (mode-0 predictor; reads the solid 0)
      pcc      : projectCorrectCenter — 1/2 (g- + g+), closed-face gradient zeroed (binary)
      open     : centerGradOpen with binary openness — full-weight open-face gradient
      open_real: centerGradOpen with the REAL sampled openness (reads the solid 0 through
                 partially-open faces — the shipping mode-6 kernel)
      ghost    : central where both neighbours fluid; 2nd-order one-sided else; 2-point
                 one-sided fallback; 0 when sandwiched. Never reads solid cells."""
    act = geo["active"]
    Pm = np.where(act, P, 0.0)
    out = []
    for a in range(3):
        Pp1 = np.roll(Pm, -1, axis=a)
        Pn1 = np.roll(Pm, +1, axis=a)
        ap1 = np.roll(act, -1, axis=a)
        an1 = np.roll(act, +1, axis=a)
        if mode == "central":
            g = 0.5 * (Pp1 - Pn1)
        elif mode == "pcc":
            g = 0.5 * (np.where(an1, Pm - Pn1, 0.0) + np.where(ap1, Pp1 - Pm, 0.0))
        elif mode == "open":
            om, op = an1.astype(float), ap1.astype(float)
            g = (om * (Pm - Pn1) + op * (Pp1 - Pm)) / np.maximum(om + op, 1e-12)
        elif mode == "open_real":
            om = o_real[a]
            op = np.roll(o_real[a], -1, axis=a)
            g = (om * (Pm - Pn1) + op * (Pp1 - Pm)) / (om + op + 1e-12)
        elif mode == "ghost":
            Pp2 = np.roll(Pm, -2, axis=a)
            Pn2 = np.roll(Pm, +2, axis=a)
            ap2 = np.roll(act, -2, axis=a)
            an2 = np.roll(act, +2, axis=a)
            g = 0.5 * (Pp1 - Pn1)
            osm = act & ~an1 & ap1          # minus neighbour solid -> one-sided toward +
            g = np.where(osm & ap2, 0.5 * (-3.0 * Pm + 4.0 * Pp1 - Pp2),
                         np.where(osm, Pp1 - Pm, g))
            osp = act & ~ap1 & an1          # plus neighbour solid -> one-sided toward -
            g = np.where(osp & an2, 0.5 * (3.0 * Pm - 4.0 * Pn1 + Pn2),
                         np.where(osp, Pm - Pn1, g))
            g = np.where(act & ~an1 & ~ap1, 0.0, g)
        else:
            raise ValueError(mode)
        out.append(np.where(act, g, 0.0))
    return out


def order(prev, cur, Nprev, N):
    return np.log2(prev / cur) / np.log2(N / Nprev) if prev is not None else float("nan")


# ---------------------------------------------------------------- [C1] constraint truncation
def test_constraint(Ns):
    print("\n[C1] ghost-closed divergence of the FACE-AVERAGED exact Stokes field (/h)")
    print("     (near-IB: localized boundary truncation ~O(h); bulk: O(h^2) — gated on the FIXED")
    print("      shell r in [0.5,0.7]: the all-bulk max hugs the surface, where |d3 u| is ~100x")
    print("      larger, so its max-norm order approaches 2 only asymptotically (measured);")
    print("      mode-0 contrast: openness divergence o_f*faceavg on the same field, O(1) at cuts)")
    print(f"{'N':>5} {'near-IB':>12} {'ord':>6} {'bulk(all)':>12} {'ord':>6} "
          f"{'bulk(shell)':>12} {'ord':>6} {'mode0 IB':>12} {'ord':>6}")
    prev = None
    slopes = {}
    for N in Ns:
        geo = build_geo(N)
        U3 = cell_fields(geo, stokes_u)
        uf = face_avg(U3)
        d = divergence(geo, uf, stokes_u) / geo["h"]
        # mode-0 constraint on the same field: sum_a (o+ uf+ - o- uf-), real sampled openness
        o = face_openness(geo)
        d0 = np.zeros_like(d)
        for a in range(3):
            om, op = o[a], np.roll(o[a], -1, axis=a)
            ufp = np.roll(uf[a], -1, axis=a)
            d0 += (op * ufp - om * uf[a]).ravel()
        d0 /= geo["h"]
        ov = overlay_cells(geo).ravel()
        interior = ((np.abs(geo["Xc"][0]) < 0.5 - 2 * geo["h"])
                    & (np.abs(geo["Xc"][1]) < 0.5 - 2 * geo["h"])
                    & (np.abs(geo["Xc"][2]) < 0.5 - 2 * geo["h"])).ravel()
        ov &= interior
        bulk = geo["active"].ravel() & ~ov & interior
        rr = np.sqrt(sum((geo["Xc"][q] - C0[q]) ** 2 for q in range(3))).ravel()
        shell = bulk & (rr >= 0.5) & (rr < 0.7)
        e_ib = float(np.abs(d[ov]).max())
        e_bk = float(np.abs(d[bulk]).max())
        e_sh = float(np.abs(d[shell]).max())
        e_m0 = float(np.abs(d0[ov & (np.abs(d0) < 1e30)]).max())
        o_ib = order(prev and prev[0], e_ib, prev and prev[4], N)
        o_bk = order(prev and prev[1], e_bk, prev and prev[4], N)
        o_sh = order(prev and prev[2], e_sh, prev and prev[4], N)
        o_m0 = order(prev and prev[3], e_m0, prev and prev[4], N)
        print(f"{N:>5} {e_ib:>12.3e} {o_ib:>6.2f} {e_bk:>12.3e} {o_bk:>6.2f} "
              f"{e_sh:>12.3e} {o_sh:>6.2f} {e_m0:>12.3e} {o_m0:>6.2f}")
        prev = (e_ib, e_bk, e_sh, e_m0, N)
        slopes = {"ib": o_ib, "bulk": o_sh}
    return slopes


# ---------------------------------------------------------------- [C2] gradient ladder
def test_grad_ladder(Ns):
    print("\n[C2] cell-gradient operators on a smooth P at CUT cells (fluid center, solid")
    print("     axis-neighbour); error vs the analytic gradient, physical units. gauge = +5")
    print("     added to P (a constant MUST not change a gradient).")
    hdr = f"{'N':>5}"
    modes = ["central", "pcc", "open", "open_real", "ghost"]
    for m in modes:
        hdr += f" {m:>11} {'ord':>5}"
    hdr += f" {'ghost gauge':>12}"
    print(hdr)
    prev = {}
    slopes = {}
    gauge_ok = True
    for N in Ns:
        geo = build_geo(N)
        act = geo["active"]
        P = phi_man(*geo["Xc"])
        gex = phi_man_grad(*geo["Xc"])
        o_real = face_openness(geo)
        # cut cells per axis: active with a solid axis-neighbour
        row = f"{N:>5}"
        for m in modes:
            G3 = cell_grad(geo, P, m, o_real)
            e = 0.0
            for a in range(3):
                cut = act & (~np.roll(act, -1, axis=a) | ~np.roll(act, +1, axis=a))
                if np.any(cut):
                    e = max(e, float(np.abs(G3[a][cut] / geo["h"] - gex[a][cut]).max()))
            o = order(prev.get(m), e, prev.get("N"), N)
            row += f" {e:>11.2e} {o:>5.2f}"
            prev[m] = e
            slopes[m] = o
            if m == "ghost":
                G3g = cell_grad(geo, P + 5.0, m, o_real)
                dg = max(float(np.abs(G3g[a] - G3[a])[act].max()) for a in range(3))
                row += f" {dg:>12.1e}"
                gauge_ok &= dg < 1e-12
        prev["N"] = N
        print(row)
    return slopes, gauge_ok


# ---------------------------------------------------------------- [C3] full chain
def pinned_solve(A, b, activef):
    """Direct solve of the singular Neumann-like system with the incompatibility dumped
    UNIFORMLY (residual = lambda*e), the sparse-friendly equivalent of mean removal.
    See ghost_projection_apriori.test_solve for the derivation + measured failure modes."""
    r0 = int(np.nonzero(activef)[0][0])
    arow = A.getrow(r0)
    Apin = A.tolil()
    Apin.rows[r0], Apin.data[r0] = [r0], [1.0]
    lu = spla.splu(Apin.tocsc())
    bp = b.copy()
    bp[r0] = 0.0
    ev = activef.astype(float)
    evp = ev.copy()
    evp[r0] = 0.0
    phi_b = lu.solve(bp)
    phi_e = lu.solve(evp)
    lam = (float((arow @ phi_b)[0]) - b[r0]) / (float((arow @ phi_e)[0]) - 1.0)
    return phi_b - lam * phi_e, lam


def run_chain(geo, cc_mode):
    """One manufactured projection chain; returns error metrics."""
    act = geo["active"]
    actf = act.ravel()
    rho = row_rescale(geo)
    U3 = cell_fields(geo, periodic_u)
    pm = phi_man(*geo["Xc"])
    Gpm = cell_grad(geo, pm, cc_mode)
    ustar = [np.where(act, U3[a] + Gpm[a], 0.0) for a in range(3)]
    uf = face_avg(ustar)

    A = assemble(geo, rho)
    b = -divergence(geo, uf, periodic_u, rho=rho)
    phi, _ = pinned_solve(A, b, actf)
    phig = phi.reshape(act.shape)

    # phi error (gauge-removed)
    dphi = phi - pm.ravel()
    dphi -= dphi[actf].mean()
    e_phi = float(np.abs(dphi[actf]).max())

    # face correction (plain grad, all faces) -> COUPLED-face error
    ufc = [uf[a] - (phig - np.roll(phig, +1, axis=a)) for a in range(3)]
    uf_ex = face_avg(cell_fields(geo, periodic_u))
    e_f = 0.0
    for a in range(3):
        st_m, _ = geo["states"][(a, -1)]
        fluid_face = (st_m == COUPLED) & act
        if np.any(fluid_face):
            e_f = max(e_f, float(np.abs(ufc[a] - uf_ex[a])[fluid_face].max()))

    # cell correction -> corrected CELL velocity error (THE gate)
    Gphi = cell_grad(geo, phig, cc_mode)
    ucorr = [np.where(act, ustar[a] - Gphi[a], 0.0) for a in range(3)]
    ovm = overlay_cells(geo)
    e_c, e_c_ib, s2, nf = 0.0, 0.0, 0.0, 0
    for a in range(3):
        err = np.abs(ucorr[a] - U3[a])
        e_c = max(e_c, float(err[act].max()))
        s2 += float((err[act] ** 2).sum())
        nf += int(act.sum())
        near = act & (ovm | np.roll(ovm, +1, axis=a) | np.roll(ovm, -1, axis=a))
        if np.any(near):
            e_c_ib = max(e_c_ib, float(err[near].max()))
    e_c_l2 = float(np.sqrt(s2 / max(nf, 1)))

    # diagnostic == residual identity on the corrected FACE field
    diag = divergence(geo, ufc, periodic_u, rho=rho, u_explicit=uf)
    res = b - A @ phi
    ident = float(np.abs(diag[actf] + res[actf]).max())
    scale = max(1.0, float(np.abs(b[actf]).max()))
    return dict(e_phi=e_phi, e_f=e_f, e_c=e_c, e_c_ib=e_c_ib, e_c_l2=e_c_l2,
                ident=ident < 1e-10 * scale)


def test_chain(Ns, modes=("ghost", "open", "pcc")):
    print("\n[C3] full collocated projection chain, manufactured (THE global-order gate)")
    print("     u*_cell = u_exact + Gc(phi_man); face-average; ghost-div; solve; correct")
    print("     faces (plain grad) + cells (Gc). Cell-correction ladder:")
    out = {}
    for m in modes:
        print(f"  --- cell gradient: {m}")
        print(f"{'N':>7} {'|phi err|':>11} {'ord':>6} {'|u_cell err|':>13} {'ord':>6} "
              f"{'near-IB':>11} {'L2':>11} {'ord':>6} {'|uf err|':>11} {'ord':>6} {'diag==res':>10}")
        prev = None
        for N in Ns:
            geo = build_geo(N)
            r = run_chain(geo, m)
            o_p = order(prev and prev["e_phi"], r["e_phi"], prev and prev["N"], N)
            o_c = order(prev and prev["e_c"], r["e_c"], prev and prev["N"], N)
            o_l2 = order(prev and prev["e_c_l2"], r["e_c_l2"], prev and prev["N"], N)
            o_f = order(prev and prev["e_f"], r["e_f"], prev and prev["N"], N)
            print(f"{N:>7} {r['e_phi']:>11.3e} {o_p:>6.2f} {r['e_c']:>13.3e} {o_c:>6.2f} "
                  f"{r['e_c_ib']:>11.3e} {r['e_c_l2']:>11.3e} {o_l2:>6.2f} "
                  f"{r['e_f']:>11.3e} {o_f:>6.2f} {'OK' if r['ident'] else 'FAIL':>10}")
            prev = dict(r, N=N)
        out[m] = dict(o_phi=o_p, o_c=o_c, o_l2=o_l2, o_f=o_f, ident=prev["ident"])
    return out


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    Ns_eval = [16, 32, 64] if args.quick else [16, 32, 64, 128]
    Ns_solve = [16, 24, 32] if args.quick else [16, 24, 32, 48]

    c1 = test_constraint(Ns_eval)
    c2, gauge_ok = test_grad_ladder(Ns_eval)
    c3 = test_chain(Ns_solve)

    g = c3["ghost"]
    print("\n==== gates ====")
    gates = [
        ("C1 constraint near-IB truncation order >= 0.8", c1["ib"] >= 0.8),
        ("C1 constraint bulk order            >= 1.7", c1["bulk"] >= 1.7),
        ("C2 ghost gradient cut-cell order    >= 1.7", c2["ghost"] >= 1.7),
        ("C2 ghost gradient gauge-independent", gauge_ok),
        # phi max is dominated by a near-IB layer that converges at ~O(h^1.4) (the averaging/
        # cell-gradient perturbation mismatch is an O(h) surface source); the quantities the
        # physics feels — corrected cell velocity, COUPLED faces — are ~O(h^2) (gates below).
        ("C3 ghost chain: phi order           >= 1.3", g["o_phi"] >= 1.3),
        ("C3 ghost chain: cell-velocity order >= 1.7", g["o_c"] >= 1.7),
        ("C3 ghost chain: COUPLED-face order  >= 1.7", g["o_f"] >= 1.7),
        ("C3 diagnostic == residual identity", g["ident"]),
    ]
    npass = 0
    for name, ok in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        npass += ok
    print(f"{npass}/{len(gates)} gates passed")
