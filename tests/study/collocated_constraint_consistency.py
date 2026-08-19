#!/usr/bin/env python
"""Is the COLLOCATED constraint operator consistent? A-priori, no solver, no time stepping.

The permeability ladders show the collocated schemes converging to a fixed offset rather than to
the staggered/continuum answer. If that offset is real the operator is ZEROTH order -- plain
inconsistent -- so test the operator directly rather than inferring it from a global functional.

Method. Take Stokes flow past a sphere: exactly solenoidal, exactly no-slip at r=R. For a
solenoidal field the EXACT flux balance over the fluid part of any cell is zero,
    sum_faces  Int_{open part of face} u.n dA  =  0
(the wall fragment contributes nothing, u.n = 0 there). So for each near-wall fluid cell compute
that sum with the scheme's MODEL flux in place of the exact one, and the residual IS the operator's
consistency error -- no second discretisation, no periodicity, no solver in the loop.

Model fluxes compared, all on the same aperture geometry:
    exact    Int over the open part, by k x k quadrature on the face  (the reference)
    stag     alpha_f * h^2 * u_a(face centre)          -- face value is a free unknown
    col      alpha_f * h^2 * 1/2 (u_i + u_j)           -- interpolated from cells, solid cells
                                                          masked to 0 as the solver does
    col_nm   as col but WITHOUT the solid-cell mask (uses the analytic continuation) -- isolates
             how much of the defect is the masking rather than the interpolation

Reported per resolution: the L1 flux defect summed over near-wall cells, normalised by the
through-flux u_inf * pi * R^2, i.e. directly comparable to a permeability error in %.

    python tests/study/collocated_constraint_consistency.py [N ...]
"""
import sys

import numpy as np

R_SPH = 0.3102
C0 = np.array([0.013, -0.007, 0.004])
KQ = 8                       # quadrature points per face direction


def stokes_u(x, y, z, comp):
    dx, dy, dz = x - C0[0], y - C0[1], z - C0[2]
    r2 = dx * dx + dy * dy + dz * dz
    r = np.sqrt(np.maximum(r2, 1e-300))
    A = 3.0 * R_SPH / (4.0 * r)
    B = R_SPH ** 3 / (4.0 * r ** 3)
    if comp == 0:
        return 1.0 - A * (1.0 + dx * dx / r2) - B * (1.0 - 3.0 * dx * dx / r2)
    d1 = (dy, dz)[comp - 1]
    return -(A - 3.0 * B) * dx * d1 / r2


def sdf(x, y, z):
    return np.sqrt((x - C0[0]) ** 2 + (y - C0[1]) ** 2 + (z - C0[2]) ** 2) - R_SPH


def face_quadrature(fc, a, h):
    """Exact open-area flux and the planar aperture for faces whose centres are `fc` (3, M)."""
    t = [k for k in range(3) if k != a]
    off = (np.arange(KQ) + 0.5) / KQ - 0.5
    P, Q = np.meshgrid(off * h, off * h, indexing="ij")
    pts = [fc[k][None, :] + 0.0 for k in range(3)]
    pts[t[0]] = fc[t[0]][None, :] + P.ravel()[:, None]
    pts[t[1]] = fc[t[1]][None, :] + Q.ravel()[:, None]
    s = sdf(*pts)
    ua = stokes_u(*pts, a)
    openm = s >= 0.0
    exact = (h * h / (KQ * KQ)) * np.where(openm, ua, 0.0).sum(axis=0)
    alpha = openm.mean(axis=0)                      # exact area fraction from the same quadrature
    return exact, alpha


def run(N, band=3.0):
    h = 1.0 / N
    c = (np.arange(N) + 0.5) * h - 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    S = sdf(X, Y, Z)
    uc = [np.where(S >= 0.0, stokes_u(X, Y, Z, k), 0.0) for k in range(3)]   # masked, as the solver
    uc_nm = [stokes_u(X, Y, Z, k) for k in range(3)]                          # unmasked

    sel = (S >= 0.0) & (np.abs(S) < band * h)        # near-wall FLUID cells
    sel[:2, :, :] = sel[-2:, :, :] = False           # keep the stencil inside the box
    sel[:, :2, :] = sel[:, -2:, :] = False
    sel[:, :, :2] = sel[:, :, -2:] = False
    idx = np.array(np.nonzero(sel))                  # (3, M)
    M = idx.shape[1]
    if M == 0:
        return None

    res = {k: np.zeros(M) for k in ("exact", "stag", "col", "col_nm")}
    for a in range(3):
        for side in (0, 1):                          # 0 = minus face of the cell, 1 = plus face
            j = idx.copy()
            j[a] += side                             # face index: minus face of cell i+side
            fc = [c[j[k]] if k != a else c[j[a]] - 0.5 * h for k in range(3)]
            fc = [np.asarray(v) for v in fc]
            ex, al = face_quadrature(fc, a, h)
            nb = idx.copy(); nb[a] += 2 * side - 1   # the cell on the other side of that face
            ui = uc[a][tuple(idx)]
            uj = uc[a][tuple(nb)]
            ui_n = uc_nm[a][tuple(idx)]
            uj_n = uc_nm[a][tuple(nb)]
            uf_c = 0.5 * (ui + uj)
            uf_n = 0.5 * (ui_n + uj_n)
            uf_s = stokes_u(*fc, a)
            sgn = 1.0 if side == 1 else -1.0
            res["exact"] += sgn * ex
            res["stag"] += sgn * al * h * h * uf_s
            res["col"] += sgn * al * h * h * uf_c
            res["col_nm"] += sgn * al * h * h * uf_n

    Q = np.pi * R_SPH ** 2                           # u_inf * projected area: the through-flux scale
    out = dict(N=N, hR=h / R_SPH, ncell=M,
               quad=float(np.abs(res["exact"]).sum() / Q))
    for k in ("stag", "col", "col_nm"):
        d = res[k] - res["exact"]
        out[k] = float(np.abs(d).sum() / Q)          # L1: no cancellation
        out[k + "_net"] = float(d.sum() / Q)         # SIGNED: what a flux/permeability feels
    return out


if __name__ == "__main__":
    Ns = [int(x) for x in (sys.argv[1:] or [32, 48, 64, 96, 128])]
    print("L1 flux defect over near-wall cells, normalised by u_inf*pi*R^2 (so read it as a "
          "relative flux error, i.e. comparable to the permeability error in %).\n")
    print(f"{'N':>5} {'h/R':>7} | {'stag L1':>9} {'ord':>6} {'stag NET':>10} {'ord':>6} | "
          f"{'col L1':>9} {'ord':>6} {'col NET':>10} {'ord':>6}")
    prev = None
    for N in Ns:
        r = run(N)
        o = {}
        if prev:
            lr = np.log(N / prev["N"])
            for k in ("stag", "col", "col_nm", "stag_net", "col_net", "col_nm_net"):
                with np.errstate(all="ignore"):
                    o[k] = f"{np.log(abs(prev[k]/r[k]))/lr:+.2f}"
        print(f"{r['N']:>5} {r['hR']:>7.4f} | {r['stag']:>9.3e} {o.get('stag',''):>6} "
              f"{r['stag_net']:>+10.3e} {o.get('stag_net',''):>6} | {r['col']:>9.3e} "
              f"{o.get('col',''):>6} {r['col_net']:>+10.3e} {o.get('col_net',''):>6}", flush=True)
        prev = r
    print("\n'quad' is the residual of the EXACT open-area balance -- the quadrature floor; every")
    print("other column must be read against it. A column converging to zero is consistent; one")
    print("that flattens above the quadrature floor is the zeroth-order signature.")
