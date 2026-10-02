"""2-D prototype, round 7: Frank Peters' directional conjugate IBM (~/Codes/conjugate/
IBM_conjugated_transport.tex) against the other conjugate candidates, on HIS test problem.

Test (tex sec. Results): steady -div(k grad T) = 0, disc r < R = 1 with k2, k1 = 1 outside, in a
square of side 5 with Dirichlet values from the exact infinite-domain solution
    inside : T = 2 k1/(k1+k2) x ;   outside: T = x + (k1-k2)/(k1+k2) R^2 x / r^2.
Errors at cell centres (L1, L2, Linf as in the tex), k2 = 0.01, 0.5, 2, 100.

Methods (all finite-volume on cell-centred unknowns; one cell = the phase of its centre unless noted)
  OF-h   one field, harmonic mean of the two cell conductivities on a face ("one field")
  OF-a   one field, face conductivity = aperture-weighted arithmetic mean a k1 + (1-a) k2
  GFM    one field, series coefficient across the crossing (Liu-Fedkiw-Kang 2000):
         k_f = 1/((1/2+xi)/k_P + (1/2-xi)/k_Q) on faces whose centre line crosses the interface
  DIR    Peters' directional scheme: on a crossing line, quadratics through (-3/2, -1/2, xi) and
         (xi, 1/2, 3/2) sharing T_xi; T_xi minimises T''(xi-)^2 + T''(xi+)^2 (eq. 12); the face
         flux is k(side of x=0) T'(0) (eq. 13/14). Non-crossing faces use the phase k.
         Fallback (no second same-phase cell on a side): GFM coefficient.
  DIR-B  DIR + the appendix "second approach" effective conductivity on faces that are CUT but whose
         centre line does not cross (k_eff with a_s, l_s = 0, n_x).
  P2F    two-field probe-flux cut-cell FV (conj.py): fluid and solid unknowns in cut cells, probe
         0.7h each side, 2x2 elimination per facet (error read from the DOF of the centre's phase).
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, sys

R = 1.0
L = 5.0


def exact(x, y, k1, k2):
    r2 = x * x + y * y
    return np.where(r2 <= R * R, 2 * k1 / (k1 + k2) * x,
                    x + (k1 - k2) / (k1 + k2) * R * R * x / np.maximum(r2, 1e-300))


class Box:
    def __init__(self, ND, off):
        h = 2 * R / ND; n = int(round(L / h)); self.h, self.n = h, n
        x0 = -L / 2 + off[0] * h; y0 = -L / 2 + off[1] * h       # shift the disc off the grid
        self.xe = x0 + h * np.arange(n + 1); self.ye = y0 + h * np.arange(n + 1)
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1]); self.yc = 0.5 * (self.ye[1:] + self.ye[:-1])
        self.X, self.Y = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.solid = np.hypot(self.X, self.Y) < R
        # solid (disc) apertures of x-faces [i, j] (face at xe[i], row j) and y-faces [i, j]
        s = np.sqrt(np.maximum(R**2 - self.xe**2, 0))[:, None]
        self.asx = np.clip(np.minimum(self.ye[None, 1:], s) - np.maximum(self.ye[None, :-1], -s), 0, h) / h
        s = np.sqrt(np.maximum(R**2 - self.ye**2, 0))[None, :]
        self.asy = np.clip(np.minimum(self.xe[1:, None], s) - np.maximum(self.xe[:-1, None], -s), 0, h) / h


def lagr_d(nodes, x, order):
    """weights w with p^(order)(x) = sum w_j f_j for the Lagrange polynomial through nodes."""
    nodes = np.asarray(nodes, float); m = len(nodes)
    V = np.vander(nodes, m, increasing=True)          # p(t) = sum c_k t^k,  V c = f
    Vi = np.linalg.inv(V)
    if order == 1:
        row = np.array([k * x ** (k - 1) if k >= 1 else 0.0 for k in range(m)])
    else:
        row = np.array([k * (k - 1) * x ** (k - 2) if k >= 2 else 0.0 for k in range(m)])
    return row @ Vi


def dir_face(xi):
    """Peters' face derivative T'(0) = sum w_j T_j over (T_-3/2, T_-1/2, T_1/2, T_3/2), unit spacing."""
    left = [-1.5, -0.5, xi]; right = [xi, 0.5, 1.5]
    a = lagr_d(left, xi, 2); b = lagr_d(right, xi, 2)     # T''(xi-) = a.(T-3/2, T-1/2, Txi) ...
    # minimise (a0 Tm3 + a1 Tm1 + a2 Txi)^2 + (b0 Txi + b1 Tp1 + b2 Tp3)^2  ->  Txi = c . (Tm3,Tm1,Tp1,Tp3)
    den = a[2] ** 2 + b[0] ** 2
    cxi = -np.array([a[2] * a[0], a[2] * a[1], b[0] * b[1], b[0] * b[2]]) / den
    if xi >= 0:   # face x = 0 lies on the left branch
        d = lagr_d(left, 0.0, 1)
        return np.array([d[0], d[1], 0.0, 0.0]) + d[2] * cxi
    d = lagr_d(right, 0.0, 1)
    return np.array([0.0, 0.0, d[1], d[2]]) + d[0] * cxi


def dir_face_both(xi):
    """(left-branch row, right-branch row) of T'(0) over (T-3/2, T-1/2, T1/2, T3/2), shared min-curvature T_xi."""
    left = [-1.5, -0.5, xi]; right = [xi, 0.5, 1.5]
    a = lagr_d(left, xi, 2); b = lagr_d(right, xi, 2)
    den = a[2] ** 2 + b[0] ** 2
    cxi = -np.array([a[2] * a[0], a[2] * a[1], b[0] * b[1], b[0] * b[2]]) / den
    dl = lagr_d(left, 0.0, 1); dr = lagr_d(right, 0.0, 1)
    return (np.array([dl[0], dl[1], 0.0, 0.0]) + dl[2] * cxi,
            np.array([0.0, 0.0, dr[1], dr[2]]) + dr[0] * cxi)


def dir_face_flux(xi, kl, kr):
    """as dir_face_both, but T_xi from FLUX continuity kl T'_l(xi) = kr T'_r(xi) (k-aware)."""
    left = [-1.5, -0.5, xi]; right = [xi, 0.5, 1.5]
    a = lagr_d(left, xi, 1); b = lagr_d(right, xi, 1)
    # kl (a0 Tm3 + a1 Tm1 + a2 Txi) = kr (b0 Txi + b1 Tp1 + b2 Tp3)
    den = kl * a[2] - kr * b[0]
    cxi = np.array([-kl * a[0], -kl * a[1], kr * b[1], kr * b[2]]) / den
    dl = lagr_d(left, 0.0, 1); dr = lagr_d(right, 0.0, 1)
    return (np.array([dl[0], dl[1], 0.0, 0.0]) + dl[2] * cxi,
            np.array([0.0, 0.0, dr[1], dr[2]]) + dr[0] * cxi)


def dir_face_closed(xi):
    """the tex's closed form (eq. after 'with f_A ...'), to cross-check dir_face."""
    fA, fB, fC, fD = 0.5 - xi, 1.5 - xi, 0.5 + xi, 1.5 + xi
    W = fA**2 * fB**2 + fC**2 * fD**2
    txi = np.array([-fA**2 * fB**2 * fC, fA**2 * fB**2 * fD, fC**2 * fD**2 * fB, -fC**2 * fD**2 * fA]) / W
    if xi >= 0:
        return np.array([fA / fD, -fB / fC, 0, 0]) + 2 / (fC * fD) * txi
    return np.array([0, 0, fD / fA, -fC / fB]) - 2 / (fA * fB) * txi


def crossing(G, P, Q, axis):
    """xi in (-1/2, 1/2): crossing of the segment P->Q (unit spacing, face at 0) with the circle."""
    p = np.array([G.X[P], G.Y[P]]); q = np.array([G.X[Q], G.Y[Q]]); d = q - p
    a = d @ d; b = 2 * p @ d; c = p @ p - R * R
    disc = np.sqrt(max(b * b - 4 * a * c, 0))
    ts = [t for t in ((-b - disc) / (2 * a), (-b + disc) / (2 * a)) if 0 <= t <= 1]
    return ts[0] - 0.5 if ts else None


def solve_one_field(G, k1, k2, method):
    n, h = G.n, G.h
    kc = np.where(G.solid, k2, k1)
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n * n)
    def add(r, c, v): rows.append(r); cols.append(c); vals.append(v)
    nfb = 0
    for axis in (0, 1):
        for i in range(n):
            for j in range(n):
                P = (i, j); Q = (i + 1, j) if axis == 0 else (i, j + 1)
                if Q[axis] >= n: continue
                a_s = G.asx[i + 1, j] if axis == 0 else G.asy[i, j + 1]
                kP, kQ = kc[P], kc[Q]
                cross = G.solid[P] != G.solid[Q]
                stencil = None  # list of (cell, weight) for k * dT/dx * h (unit-spacing derivative)
                if method == "OF-h":
                    kf = 2 * kP * kQ / (kP + kQ); stencil = [(P, -kf), (Q, kf)]
                elif method in ("DIRq", "DIRq-side"):
                    kf = (1 - a_s) * k1 + a_s * k2 if method == "DIRq" else kP
                    stencil = [(P, -kf), (Q, kf)]
                    if cross:
                        xi = crossing(G, P, Q, axis)
                        sh = (1, 0) if axis == 0 else (0, 1)
                        Pm = (P[0] - sh[0], P[1] - sh[1]); Qp = (Q[0] + sh[0], Q[1] + sh[1])
                        gl, gr = dir_face_flux(xi, kP, kQ)
                        if method == "DIRq":
                            fP = a_s if G.solid[P] else 1 - a_s
                            w = kP * fP * gl + kQ * (1 - fP) * gr
                        else:
                            w = kP * gl if xi >= 0 else kQ * gr
                        stencil = [(Pm, w[0]), (P, w[1]), (Q, w[2]), (Qp, w[3])]
                elif method == "DIRc":
                    kf = (1 - a_s) * k1 + a_s * k2
                    stencil = [(P, -kf), (Q, kf)]
                    if cross:
                        xi = crossing(G, P, Q, axis)
                        sh = (1, 0) if axis == 0 else (0, 1)
                        Pm = (P[0] - sh[0], P[1] - sh[1]); Qp = (Q[0] + sh[0], Q[1] + sh[1])
                        gl, gr = dir_face_both(xi)
                        fP = a_s if G.solid[P] else 1 - a_s      # fraction of the face in P's phase
                        w = kP * fP * gl + kQ * (1 - fP) * gr
                        stencil = [(Pm, w[0]), (P, w[1]), (Q, w[2]), (Qp, w[3])]
                elif method == "OF-a":
                    kf = (1 - a_s) * k1 + a_s * k2; stencil = [(P, -kf), (Q, kf)]
                elif not cross:
                    kf = kP
                    if method == "DIR-B" and 0 < a_s < 1:
                        # appendix 'second approach', reference phase = the centre-line phase, l = 0
                        kr, ko = kP, (k1 if kP == k2 else k2)       # line phase, other phase
                        ao = a_s if kP == k1 else 1 - a_s          # fraction of the face in the other phase
                        # normal of the interface near the face centre
                        fx = G.xe[i + 1] if axis == 0 else G.xc[i]; fy = G.yc[j] if axis == 0 else G.ye[j + 1]
                        nrm = np.array([fx, fy]) / np.hypot(fx, fy); nx2 = nrm[axis] ** 2
                        S = ao / ko + (1 - ao) / kr
                        kf = (1 / kr * nx2 + S * (1 - nx2)) / ((1 / kr) ** 2 * nx2 + S / (ao * ko + (1 - ao) * kr) * (1 - nx2))
                    stencil = [(P, -kf), (Q, kf)]
                else:
                    xi = crossing(G, P, Q, axis)
                    sh = (1, 0) if axis == 0 else (0, 1)
                    Pm = (P[0] - sh[0], P[1] - sh[1]); Qp = (Q[0] + sh[0], Q[1] + sh[1])
                    ok = (method in ("DIR", "DIR-B") and min(Pm) >= 0 and max(Qp) < n and
                          G.solid[Pm] == G.solid[P] and G.solid[Qp] == G.solid[Q])
                    if ok:
                        w = dir_face(xi)
                        kf = kP if xi >= 0 else kQ
                        stencil = [(Pm, kf * w[0]), (P, kf * w[1]), (Q, kf * w[2]), (Qp, kf * w[3])]
                    else:
                        if method in ("DIR", "DIR-B"): nfb += 1
                        kf = 1 / ((0.5 + xi) / kP + (0.5 - xi) / kQ); stencil = [(P, -kf), (Q, kf)]
                # flux F = sum w T (into Q from P is -F... F is k dT/dx: flux in +x = -F)
                # cell P loses -F (outflow +x is -F) : div contribution for P: -(F)/h^2 ... assemble -div(k grad T)
                for c, w in stencil:
                    add(idx[P], idx[c], -w / h**2 * -1)   # P: -( -F_face_out ) ... see below
                    add(idx[Q], idx[c], -w / h**2)
    # sign bookkeeping: -div(k grad T) at P = -(F_east - F_west)/h^2 with F = k dT/dx.
    # The east face of P contributes -F/h^2 to P and the west face of Q contributes +F/h^2 to Q.
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))
    A = -A  # flip: the loop above added +F to P and -F to Q
    # outer Dirichlet faces: flux k1 (Tb - TP)/(h/2) through each boundary face
    diag = np.zeros(n * n)
    for (P, bx, by) in ([((0, j), G.xe[0], G.yc[j]) for j in range(n)] + [((n - 1, j), G.xe[n], G.yc[j]) for j in range(n)] +
                        [((i, 0), G.xc[i], G.ye[0]) for i in range(n)] + [((i, n - 1), G.xc[i], G.ye[n]) for i in range(n)]):
        c = 2 * k1 / h**2
        diag[idx[P]] += c; rhs[idx[P]] += c * exact(bx, by, k1, k2)
    A = A + sp.diags(diag)
    T = spla.spsolve(A.tocsc(), rhs)
    return T.reshape(n, n), nfb


def norms(G, T, k1, k2):
    e = (T - exact(G.X, G.Y, k1, k2)).ravel()
    return np.mean(np.abs(e)), np.sqrt(np.mean(e * e)), np.abs(e).max()



# ---------------------------------------------------------------------------------------------
# P2F: two-field probe-flux cut-cell FV on the same Box test (K = 1, no contact resistance)
def solve_p2f(G, k1, k2, c=0.7):
    from conj import bilin
    n, h = G.n, G.h
    gq, wq = np.polynomial.legendre.leggauss(24)
    ks = G.solid.astype(float)
    r = np.hypot(G.X, G.Y)
    cut = np.abs(r - R) < 1.5 * h
    for i, j in zip(*np.nonzero(cut)):
        XX, YY = np.meshgrid(G.xc[i] + 0.5 * h * gq, G.yc[j] + 0.5 * h * gq, indexing="ij")
        ks[i, j] = (np.outer(wq, wq) * 0.25 * (np.hypot(XX, YY) < R)).sum()
    kf = 1 - ks
    vf, vs = kf > 1e-12, ks > 1e-12
    idf = -np.ones((n, n), int); idf[vf] = np.arange(vf.sum())
    ids = -np.ones((n, n), int); ids[vs] = vf.sum() + np.arange(vs.sum())
    N = vf.sum() + vs.sum()
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    def add(r_, c_, v): rows.append(r_); cols.append(c_); vals.append(v)
    afx, afy = 1 - G.asx, 1 - G.asy
    for (idx, D, ax_, ay_) in ((idf, k1, afx, afy), (ids, k2, G.asx, G.asy)):
        for (a, P, Q) in ((ax_[1:-1, :], idx[:-1, :], idx[1:, :]), (ay_[:, 1:-1], idx[:, :-1], idx[:, 1:])):
            m = (a > 0) & (P >= 0) & (Q >= 0)
            for aa, p, q in zip(a[m], P[m], Q[m]):
                cf = D * aa / h**2
                add(p, p, cf); add(p, q, -cf); add(q, q, cf); add(q, p, -cf)
    Ax = -(G.asx[1:, :] - G.asx[:-1, :]) * h; Ay = -(G.asy[:, 1:] - G.asy[:, :-1]) * h
    Aw = np.hypot(Ax, Ay)
    s = c * h
    for i, j in zip(*np.nonzero(Aw > 1e-12 * h)):
        if idf[i, j] < 0 or ids[i, j] < 0: continue
        xw = np.array([G.X[i, j], G.Y[i, j]]); xw = xw * R / np.hypot(*xw); nn = xw / R
        for sf in (s, 1.0 * h, 1.5 * h):
            pf = bilin(G, idf, xw + sf * nn)
            if pf is not None: break
        for ss in (s, 1.0 * h, 1.5 * h):
            ps = bilin(G, ids, xw - ss * nn)
            if ps is not None: break
        gf, gs_ = k1 / sf, k2 / ss
        # continuity uGs = uGf; flux continuity gf (uG - upf) = gs (ups - uG) -> uG = (gf upf + gs ups)/(gf+gs)
        # q (solid -> fluid, along n) = gf (uG - upf) = gf gs (ups - upf)/(gf + gs)
        g = gf * gs_ / (gf + gs_)
        for (col, wv) in pf:
            add(idf[i, j], col, +Aw[i, j] * g * wv / h**2); add(ids[i, j], col, -Aw[i, j] * g * wv / h**2)
        for (col, wv) in ps:
            add(idf[i, j], col, -Aw[i, j] * g * wv / h**2); add(ids[i, j], col, +Aw[i, j] * g * wv / h**2)
    # outer Dirichlet (fluid only near the box)
    for (P, bx, by) in ([((0, j), G.xe[0], G.yc[j]) for j in range(n)] + [((n - 1, j), G.xe[n], G.yc[j]) for j in range(n)] +
                        [((i, 0), G.xc[i], G.ye[0]) for i in range(n)] + [((i, n - 1), G.xc[i], G.ye[n]) for i in range(n)]):
        cc = 2 * k1 / h**2
        add(idf[P], idf[P], cc); rhs[idf[P]] += cc * exact(bx, by, k1, k2)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    u = spla.spsolve(A.tocsc(), rhs)
    T = np.where(G.solid, u[np.maximum(ids, 0)], u[np.maximum(idf, 0)])
    return T, 0



if __name__ == "__main__":
    for xi in (-0.49, -0.2, 0.0, 0.3, 0.49):
        assert np.allclose(dir_face(xi), dir_face_closed(xi), atol=1e-10), xi
    print("closed-form face weights (tex) == generic Lagrange derivation: OK")
    methods = sys.argv[1:] or ["OF-h", "OF-a", "GFM", "DIR", "DIR-B"]
    rng = np.random.default_rng(3); offs = rng.random((2, 2))
    if "centred" in methods:
        methods = [m for m in methods if m != "centred"]; offs = np.zeros((1, 2))
    NDs = [16, 32, 64, 128]
    for k2 in (0.01, 0.5, 2.0, 100.0):
        for m in methods:
            line = f"k2={k2:6g} {m:6s}"; prev = None; fb = 0
            for ND in NDs:
                ee = []
                for o in offs:
                    G = Box(ND, o)
                    T, nfb = (solve_p2f(G, 1.0, k2) if m == "P2F" else solve_one_field(G, 1.0, k2, m)); fb += nfb
                    ee.append(norms(G, T, 1.0, k2))
                ee = np.mean(ee, axis=0)
                line += f" | {ND:3d}: L1 {ee[0]:.1e} Linf {ee[2]:.1e}" + (f" p={np.log(prev/ee[0])/np.log(2):4.1f}" if prev is not None else "      ")
                prev = ee[0]
            print(line + f"  fallbacks={fb}", flush=True)
