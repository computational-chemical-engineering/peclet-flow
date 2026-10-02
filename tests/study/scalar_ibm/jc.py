"""2-D prototype, round 3: Johansen-Colella / Schwartz-Barad-Colella-Ligocki (2006) style cut-cell FV
(the AMReX EB approach), with Robin eliminated at the wall.  Disc gates as in hybrid.py.

  values         : cell-centred, every kappa>0 cell is an unknown, mass kappa
  face flux      : aperture * gradient at the open-face CENTROID, linearly interpolated between the
                   face's own two-point gradient and the parallel neighbouring face's (if open)
                   [toggle fc=False -> plain aperture two-point flux]
  wall flux      : quadratic in the wall-normal distance through u_Gamma, u(p1), u(p2); p1,p2 where
                   the normal ray from the wall centroid crosses the next two cell-centre lines of the
                   dominant normal axis; u(p) by 3-point quadratic interpolation along that line
                   [toggle wall='lin' -> 2-point (u(p1)-u_G)/d1]
  Robin          : D du/ds = k u_G  (s into the fluid) -> u_G eliminated per wall:
                   du/ds = S - beta u_G,  u_G = D S / (k + D beta);  wall flux into cell = -k A_w u_G
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, sys
from disc_bc import Geo, R
from hybrid import mu_ref


def lagr(xs, x):
    w = np.ones(len(xs))
    for a in range(len(xs)):
        for b in range(len(xs)):
            if a != b:
                w[a] *= (x - xs[b]) / (xs[a] - xs[b])
    return w


REACH = [0]

def build(G, k, fc=True, wall="quad"):
    n, h = G.n, G.h
    valid = G.kap > 1e-12
    idx = -np.ones((n, n), int); idx[valid] = np.arange(valid.sum())
    N = valid.sum()
    rows, cols, vals = [], [], []
    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)
    # ---- faces: open interval centroid offsets
    def xface_cent(i, j):  # x-face at xe[i], row j: open part centroid y offset in units of h
        s = np.sqrt(max(R**2 - G.xe[i] ** 2, 0)); y0, y1 = G.ye[j], G.ye[j + 1]
        lo, hi = max(y0, -s), min(y1, s)
        return (0.5 * (lo + hi) - G.yc[j]) / h
    def yface_cent(i, j):
        s = np.sqrt(max(R**2 - G.ye[j] ** 2, 0)); x0, x1 = G.xe[i], G.xe[i + 1]
        lo, hi = max(x0, -s), min(x1, s)
        return (0.5 * (lo + hi) - G.xc[i]) / h
    # flux through x-face (i-1,j)|(i,j) as a list of (cell, coef) giving (grad u . e_x) at centroid
    def flux_x(i, j):
        a = G.ax[i, j]
        if a <= 0 or idx[i - 1, j] < 0 or idx[i, j] < 0:
            return a, []
        terms = [((i, j), 1 / h), ((i - 1, j), -1 / h)]
        if fc and a < 1:
            d = xface_cent(i, j)
            jj = j + (1 if d > 0 else -1)
            if 0 <= jj < n and G.ax[i, jj] > 0 and idx[i - 1, jj] >= 0 and idx[i, jj] >= 0:
                f = abs(d)
                terms = [(c, (1 - f) * v) for c, v in terms] + \
                        [((i, jj), f / h), ((i - 1, jj), -f / h)]
        return a, terms
    def flux_y(i, j):
        a = G.ay[i, j]
        if a <= 0 or idx[i, j - 1] < 0 or idx[i, j] < 0:
            return a, []
        terms = [((i, j), 1 / h), ((i, j - 1), -1 / h)]
        if fc and a < 1:
            d = yface_cent(i, j)
            ii = i + (1 if d > 0 else -1)
            if 0 <= ii < n and G.ay[ii, j] > 0 and idx[ii, j - 1] >= 0 and idx[ii, j] >= 0:
                f = abs(d)
                terms = [(c, (1 - f) * v) for c, v in terms] + \
                        [((ii, j), f / h), ((ii, j - 1), -f / h)]
        return a, terms
    # -L u * kappa h^2 = -(sum of outward fluxes) ; we assemble K = -L*(kappa h^2)/h^2 so K u = mu kappa u
    for i, j in zip(*np.nonzero(valid)):
        r = idx[i, j]
        for (a, terms, sgn) in (flux_x(i + 1, j) + (+1,), flux_x(i, j) + (-1,),
                                flux_y(i, j + 1) + (+1,), flux_y(i, j) + (-1,)):
            for (c, v) in terms:
                add(r, idx[c], -sgn * a * h * v / h**2)
        if G.Aw[i, j] <= 1e-14 * h or np.isinf(k) and False:
            pass
        if G.Aw[i, j] > 1e-14 * h and k > 0:
            # wall centroid: radial projection of the fluid centroid (exact enough for a circle)
            cx, cy = G.cx[i, j], G.cy[i, j]
            rr = np.hypot(cx, cy); xw = np.array([cx, cy]) * R / rr
            nn = -xw / R  # into the fluid
            ax_ = 0 if abs(nn[0]) >= abs(nn[1]) else 1
            sg = 1 if nn[ax_] > 0 else -1
            ci = (i, j)
            pts = []
            for m in (1, 2):
                if ax_ == 0:
                    col = i + sg * m; s = (G.xc[col] - xw[0]) / nn[0]
                    py = xw[1] + s * nn[1]; jn = int(round((py - G.yc[0]) / h))
                    cand = [(col, jn - 1), (col, jn), (col, jn + 1)]; coords = [G.yc[q] for _, q in cand]; t = py
                else:
                    row = j + sg * m; s = (G.yc[row] - xw[1]) / nn[1]
                    px = xw[0] + s * nn[0]; im = int(round((px - G.xc[0]) / h))
                    cand = [(im - 1, row), (im, row), (im + 1, row)]; coords = [G.xc[q] for q, _ in cand]; t = px
                ok = all(idx[c] >= 0 for c in cand)
                if not ok:
                    break
                pts.append((s, cand, lagr(np.array(coords), t)))
            if wall.startswith("probe"):
                c = float(wall[5:] or 1.0)
                sp_ = c * h
                p = xw + sp_ * nn
                fi = (p[0] - G.xc[0]) / h; fj = (p[1] - G.yc[0]) / h
                i0, j0 = int(np.floor(fi)), int(np.floor(fj)); tx, ty = fi - i0, fj - j0
                cand = [(i0, j0), (i0 + 1, j0), (i0, j0 + 1), (i0 + 1, j0 + 1)]
                ww = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
                if not all(idx[q] >= 0 for q in cand):
                    raise RuntimeError("probe hits invalid cell")
                reach = max(max(abs(q[0] - i), abs(q[1] - j)) for q in cand)
                REACH[0] = max(REACH[0], reach)
                pts = [(sp_, cand, np.array(ww))]
            use_quad = (wall == "quad" and len(pts) == 2)
            if len(pts) == 0:
                raise RuntimeError("no interpolation point")
            # du/ds = sum_c S_c u_c - beta u_G
            if use_quad:
                (d1, c1, w1), (d2, c2, w2) = pts
                den = d1 * d2 * (d2 - d1)
                S = [(c, d2 * d2 * w / den) for c, w in zip(c1, w1)] + \
                    [(c, -d1 * d1 * w / den) for c, w in zip(c2, w2)]
                beta = (d1 + d2) / (d1 * d2)
            else:
                (d1, c1, w1) = pts[0]
                S = [(c, w / d1) for c, w in zip(c1, w1)]; beta = 1 / d1
            # flux out of fluid through wall = D du/ds_out... wall flux INTO cell = -D du/ds (D=1) ;
            # with Robin: D du/ds = k u_G -> u_G = S.u/(k + beta); wall term into cell = -k A_w u_G
            fac = 1.0 if np.isinf(k) else k / (k + beta)
            # inf: u_G=0 -> into-cell flux = -(S.u)
            for c, v in S:
                add(r, idx[c], fac * G.Aw[i, j] * v / h**2)
    K = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return K, G.kap[valid]


def eig(K, M, which):
    s = 1 / np.sqrt(M)
    A = sp.diags(s) @ K @ sp.diags(s)
    w = spla.eigs(A.tocsc(), k=3, sigma=-1e-3, which="LM", return_eigenvectors=False)
    w = np.sort(w.real)
    return w[which]


if __name__ == "__main__":
    rng = np.random.default_rng(1); offs = rng.random((3, 2))
    NDs = [16, 32, 64, 128]
    variants = {"JC-lin": dict(fc=False, wall="lin"), "probe1.0": dict(fc=False, wall="probe1.0"),
                "probe1.5": dict(fc=False, wall="probe1.5"), "probe0.7": dict(fc=False, wall="probe0.7")}
    for Bi in (0.0, 0.1, 1.0, 10.0, 100.0, np.inf):
        ref = mu_ref(Bi); which = 1 if Bi == 0 else 0
        for tag, kw in variants.items():
            line = f"Bi={Bi:6g} {tag:8s}"; prev = None
            for ND in NDs:
                e = []
                for o in offs:
                    G = Geo(ND, o)
                    K, M = build(G, Bi, **kw)
                    e.append(eig(K, M, which) / ref - 1)
                em = np.sqrt(np.mean(np.square(e)))
                line += f" | {ND:3d}: {np.mean(e):+9.2e}" + (f" p={np.log(prev/em)/np.log(2):4.1f}" if prev else "       ")
                prev = em
            print(line + f'  reach={REACH[0]}', flush=True); REACH[0] = 0
