"""mac2d generalised to per-cell planes (architect follow-up Q-A, 2026-10-31).

Only the production configuration of mac2d: scheme 'sharp', SHARP_BASE 'facef', FCORR 'fit', DNN_W 'all'.
Geometry goes through a geo object so that a curved interface (circle) can carry one PLIC plane per mixed
cell, edge (node) owners, and a per-cell fit normal chosen by a rule.  PlanarGeo reproduces mac2d.
tau rules: 'band-avg' (mean over band cells among the 4), 'mixed-avg' (interfacial cells only), 'owner'.
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mac2d as M0  # noqa: E402

GL_X, GL_W = M0.GL_X, M0.GL_W
G8_X, G8_W = np.polynomial.legendre.leggauss(8)


def poly_frac(phi, x0, y0, h):
    poly = [(x0, y0), (x0 + h, y0), (x0 + h, y0 + h), (x0, y0 + h)]
    out = []
    for k in range(4):
        P, Q = poly[k], poly[(k + 1) % 4]
        fp, fq = phi(*P), phi(*Q)
        if fp < 0:
            out.append(P)
        if (fp < 0) != (fq < 0):
            w = fp / (fp - fq)
            out.append((P[0] + w * (Q[0] - P[0]), P[1] + w * (Q[1] - P[1])))
    if len(out) < 3:
        return 0.0
    A = 0.0
    for k in range(len(out)):
        A += out[k][0] * out[(k + 1) % len(out)][1] - out[(k + 1) % len(out)][0] * out[k][1]
    return abs(A) / 2 / (h * h)


def seg_frac(n, ref, s0, x0, y0, e, kern, h):
    p0 = n[0] * (x0 - ref[0]) + n[1] * (y0 - ref[1]) - s0
    g = n[0] * e[0] + n[1] * e[1]
    cdf = M0.cdf_box if kern == "box" else M0.cdf_hat
    if abs(g) < 1e-14:
        return 1.0 if p0 < 0 else 0.0
    ss = -p0 / g
    F = float(cdf(ss, h))
    return F if g > 0 else 1.0 - F


class PlanarGeo:
    def __init__(self, case):
        self.case, N, h = case, case.N, case.h
        self.N, self.h = N, h
        C = np.zeros((N + 2, N + 2))
        for i in range(-1, N + 1):
            for j in range(-1, N + 1):
                x0, y0 = i * h, j * h
                ph = [case.phi(x0, y0), case.phi(x0 + h, y0), case.phi(x0, y0 + h), case.phi(x0 + h, y0 + h)]
                if max(ph) < 0:
                    C[i + 1, j + 1] = 1.0
                elif min(ph) >= 0:
                    C[i + 1, j + 1] = 0.0
                else:
                    C[i + 1, j + 1] = case.frac_cell(x0, y0)
        self.Ca = C

    def C(self, i, j):
        return self.Ca[i + 1, j + 1]

    def node_frac(self, i, j, e, kern):
        return self.case.frac_line(i * self.h, j * self.h, e, kern)

    def mid_frac(self, i, j, e):
        return self.case.frac_line((i + 0.5) * self.h, (j + 0.5) * self.h, e, "box")

    def node_nt(self, i, j):
        return self.case.n, self.case.t

    cell_nt = node_nt
    fit_nt = node_nt

    def owner(self, node, cells, mixed):
        N, h = self.N, self.h
        cand = [(ci, cj) for (ci, cj) in cells if 0 <= ci <= N - 1 and 0 <= cj <= N - 1 and mixed[ci + 1, cj + 1]]
        if not cand:
            cand = [(ci, cj) for (ci, cj) in cells if 0 <= ci <= N - 1 and 0 <= cj <= N - 1]
        return min(cand, key=lambda q: (abs(self.case.phi((q[0] + 0.5) * h, (q[1] + 0.5) * h)), q))


class CircleCase:
    """Two-layer azimuthal Stokes flow U(r) e_theta; side A = inside r < r0.  'couette': g = 0; 'poiseuille':
    azimuthal body force g e_theta.  tau(r0) = tau0, p = const, D_nn = 0."""

    def __init__(self, N, cx, cy, r0, muA, muB, kind, g=1.0, tau0=1.0):
        self.N, self.h = N, 1.0 / N
        self.c = np.array([cx, cy])
        self.r0, self.muA, self.muB, self.kind, self.tau0 = r0, muA, muB, kind, tau0
        self.g = g if kind == "poiseuille" else 0.0
        beta = -(tau0 + self.g * r0 / 3.0) * r0 ** 2 / 2.0
        self.BA, self.BB, self.AB = beta / muA, beta / muB, 0.0
        UB0 = -self.g * r0 ** 2 / (3 * muB) + self.BB / r0
        self.AA = (UB0 + self.g * r0 ** 2 / (3 * muA) - self.BA / r0) / r0

    def phi(self, x, y):
        return np.hypot(x - self.c[0], y - self.c[1]) - self.r0

    def U(self, r):
        ins = r < self.r0
        mu = np.where(ins, self.muA, self.muB)
        A = np.where(ins, self.AA, self.AB)
        B = np.where(ins, self.BA, self.BB)
        return -self.g * r * r / (3 * mu) + A * r + B / r

    def vel(self, x, y):
        dx, dy = x - self.c[0], y - self.c[1]
        r = np.hypot(dx, dy)
        U = self.U(r)
        return -U * dy / r, U * dx / r

    def face_avg(self, comp, x0, y0, e):
        h = self.h
        d0, d1 = x0 - self.c[0], y0 - self.c[1]
        b = e[0] * d0 + e[1] * d1
        q = d0 * d0 + d1 * d1 - self.r0 ** 2
        disc = b * b - q
        cuts = [-0.5 * h, 0.5 * h]
        if disc > 0:
            for s in (-b - np.sqrt(disc), -b + np.sqrt(disc)):
                if -0.5 * h < s < 0.5 * h:
                    cuts.append(s)
        cuts.sort()
        tot = 0.0
        for k in range(len(cuts) - 1):
            a, bb = cuts[k], cuts[k + 1]
            sq = 0.5 * (a + bb) + 0.5 * (bb - a) * G8_X
            v = self.vel(x0 + sq * e[0], y0 + sq * e[1])[comp]
            tot += 0.5 * (bb - a) * np.dot(G8_W, v)
        return tot / h

    def row_force(self, comp, xf, yf):
        if self.g == 0.0:
            return 0.0
        h = self.h
        X, Y = np.meshgrid(xf + 0.5 * h * GL_X, yf + 0.5 * h * GL_X, indexing="ij")
        W = np.outer(GL_W, GL_W) / 4.0
        dx, dy = X - self.c[0], Y - self.c[1]
        r = np.hypot(dx, dy)
        f = (-dy / r) if comp == 0 else (dx / r)
        return float(self.g * np.sum(W * f))

    def cell_frac(self, x0, y0):
        h, cx, cy, r0 = self.h, self.c[0], self.c[1], self.r0
        bps = [x0, x0 + h]
        for yb in (y0, y0 + h):
            dd = r0 * r0 - (yb - cy) ** 2
            if dd > 0:
                for xs in (cx - np.sqrt(dd), cx + np.sqrt(dd)):
                    if x0 < xs < x0 + h:
                        bps.append(xs)
        for xs in (cx - r0, cx + r0):
            if x0 < xs < x0 + h:
                bps.append(xs)
        bps.sort()
        tot = 0.0
        for a, b in zip(bps[:-1], bps[1:]):
            xs = 0.5 * (a + b) + 0.5 * (b - a) * G8_X
            w = np.sqrt(np.maximum(r0 * r0 - (xs - cx) ** 2, 0.0))
            L = np.clip(np.minimum(y0 + h, cy + w) - np.maximum(y0, cy - w), 0.0, None)
            tot += 0.5 * (b - a) * np.dot(G8_W, L)
        return tot / (h * h)


class CircleGeo:
    """PLIC plane per mixed cell: exact normal at the cell centre's closest point, offset matching C exactly.
    Node (edge) owner: mixed cell among the 4 with the smallest |phi_J(node)|, ties in fixed offset order.
    fitrule for non-interfacial band cells: 'nearest' | 'mean' | 'oracle'."""

    def __init__(self, case, fitrule="nearest"):
        self.case, self.fitrule = case, fitrule
        N, h = case.N, case.h
        self.N, self.h = N, h
        C = np.zeros((N + 2, N + 2))
        for i in range(-1, N + 1):
            for j in range(-1, N + 1):
                f = case.cell_frac(i * h, j * h)
                C[i + 1, j + 1] = 0.0 if f < 1e-13 else (1.0 if f > 1 - 1e-13 else f)
        self.Ca = C
        self.pl = {}
        for i in range(-1, N + 1):
            for j in range(-1, N + 1):
                f = C[i + 1, j + 1]
                if 0.0 < f < 1.0:
                    xc, yc = (i + 0.5) * h, (j + 0.5) * h
                    n = np.array([xc - case.c[0], yc - case.c[1]])
                    n /= np.linalg.norm(n)
                    t = np.array([-n[1], n[0]])
                    ref = (xc, yc)
                    lo, hi = -0.75 * h, 0.75 * h
                    for _ in range(64):
                        s0 = 0.5 * (lo + hi)
                        fr = poly_frac(lambda x, y: n[0] * (x - ref[0]) + n[1] * (y - ref[1]) - s0, i * h, j * h, h)
                        if fr < f:
                            lo = s0
                        else:
                            hi = s0
                    self.pl[(i, j)] = (n, t, ref, 0.5 * (lo + hi))
        self.own = {}
        for i in range(0, N + 1):
            for j in range(0, N + 1):
                best, bd = None, None
                for q in ((i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j)):
                    if q in self.pl:
                        n, t, ref, s0 = self.pl[q]
                        d = abs(n[0] * (i * h - ref[0]) + n[1] * (j * h - ref[1]) - s0)
                        if bd is None or d < bd:
                            best, bd = q, d
                self.own[(i, j)] = best
                if best is None:
                    ph = {self.Ca[q[0] + 1, q[1] + 1] for q in ((i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j))}
                    assert len(ph) == 1, "grid-aligned interface: not supported in this harness"
        self.fn = {}

    def C(self, i, j):
        return self.Ca[i + 1, j + 1]

    def node_frac(self, i, j, e, kern):
        q = self.own[(i, j)]
        if q is None:
            return self.Ca[i + 1, j + 1]
        n, t, ref, s0 = self.pl[q]
        return seg_frac(n, ref, s0, i * self.h, j * self.h, e, kern, self.h)

    def mid_frac(self, i, j, e):
        if (i, j) not in self.pl:
            return self.Ca[i + 1, j + 1]
        n, t, ref, s0 = self.pl[(i, j)]
        return seg_frac(n, ref, s0, (i + 0.5) * self.h, (j + 0.5) * self.h, e, "box", self.h)

    _dummy = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    def node_nt(self, i, j):
        q = self.own[(i, j)]
        return self._dummy if q is None else self.pl[q][:2]

    def cell_nt(self, i, j):
        return self.pl[(i, j)][:2] if (i, j) in self.pl else self._dummy

    def fit_nt(self, i, j):
        if (i, j) in self.pl:
            return self.pl[(i, j)][:2]
        if (i, j) in self.fn:
            return self.fn[(i, j)]
        h = self.h
        xc, yc = (i + 0.5) * h, (j + 0.5) * h
        if self.fitrule == "oracle":
            n = np.array([xc - self.case.c[0], yc - self.case.c[1]])
            n /= np.linalg.norm(n)
        elif self.fitrule == "mean":
            s = np.zeros(2)
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    if (i + di, j + dj) in self.pl:
                        s += self.pl[(i + di, j + dj)][0]
            n = s / np.linalg.norm(s)
        else:
            best, bd = None, None
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    q = (i + di, j + dj)
                    if q in self.pl:
                        nn, tt, ref, s0 = self.pl[q]
                        d = abs(nn[0] * (xc - ref[0]) + nn[1] * (yc - ref[1]) - s0)
                        if bd is None or d < bd:
                            best, bd = q, d
            n = self.pl[best][0]
        self.fn[(i, j)] = (n, np.array([-n[1], n[0]]))
        return self.fn[(i, j)]

    def owner(self, node, cells, mixed):
        return self.own[node]


def build(case, geo, rule="band-avg"):
    N, h = case.N, case.h
    m = M0.Mac(N)
    muA, muB = case.muA, case.muB
    ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    Cc = geo.C
    mu_of = lambda f: f * muA + (1 - f) * muB
    R_of = lambda f: f / muA + (1 - f) / muB
    lowA = muA < muB
    mulo, muhi = (muA, muB) if lowA else (muB, muA)
    rr = mulo / muhi

    def cell_links(i, j):
        f = Cc(i, j)
        return [("xx", 0, 0, (m.iu(i + 1, j), m.iu(i, j)), f), ("yy", 1, 1, (m.iv(i, j + 1), m.iv(i, j)), f)]

    def node_links(i, j):
        thy = geo.node_frac(i, j, ey, "hat")
        thx = geo.node_frac(i, j, ex, "hat")
        return [("yu", 1, 0, (m.iu(i, j), m.iu(i, j - 1)), thy), ("xv", 0, 1, (m.iv(i, j), m.iv(i - 1, j)), thx)]

    mixed = np.zeros((N + 2, N + 2), bool)
    for i in range(-1, N + 1):
        for j in range(-1, N + 1):
            mixed[i + 1, j + 1] = 0.0 < Cc(i, j) < 1.0
    band = mixed.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            band[1:-1, 1:-1] |= mixed[1 + di:N + 1 + di, 1 + dj:N + 1 + dj]
    tauexpr, dnnexpr = {}, {}

    def tau_fit(i, j):
        if (i, j) in tauexpr:
            return tauexpr[(i, j)]
        n, t = geo.fit_nt(i, j)
        rows = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ci, cj = i + di, j + dj
                if not (0 <= ci <= N - 1 and 0 <= cj <= N - 1):
                    continue
                for L in cell_links(ci, cj):
                    rows.append(L)
        for ni in range(i - 1, i + 3):
            for nj in range(j - 1, j + 3):
                if not (0 <= ni <= N and 0 <= nj <= N):
                    continue
                for L in node_links(ni, nj):
                    rows.append(L)
        A = np.zeros((len(rows), 4))
        for r, (_, a, c, _, thA) in enumerate(rows):
            thlo = thA if lowA else 1 - thA
            A[r, 2 * a + c] += 1.0
            k = (1 - thlo) * n[a] * t[c] * (rr - 1)
            for p in range(2):
                for q in range(2):
                    A[r, 2 * p + q] += k * (t[p] * n[q] + t[q] * n[p])
        G = np.linalg.pinv(A)
        wv = np.array([[t[p] * n[q] + t[q] * n[p] for q in range(2)] for p in range(2)]).ravel()
        coef = mulo * (wv @ G)
        nn = np.array([[n[p] * n[q] for q in range(2)] for p in range(2)]).ravel()
        cdn = nn @ G
        expr, dexpr = {}, {}
        for r, (_, a, c, (ip, im), _) in enumerate(rows):
            expr[ip] = expr.get(ip, 0.0) + coef[r] / h
            expr[im] = expr.get(im, 0.0) - coef[r] / h
            dexpr[ip] = dexpr.get(ip, 0.0) + cdn[r] / h
            dexpr[im] = dexpr.get(im, 0.0) - cdn[r] / h
        tauexpr[(i, j)] = expr
        dnnexpr[(i, j)] = dexpr
        return expr

    def avg_over(cs):
        e = {}
        for cc in cs:
            for k, v in tau_fit(*cc).items():
                e[k] = e.get(k, 0.0) + v / len(cs)
        return e

    def tau_at(cells, node):
        ind = [(ci, cj) for (ci, cj) in cells if 0 <= ci <= N - 1 and 0 <= cj <= N - 1]
        if rule == "owner":
            if node is None:
                return tau_fit(*cells[0])
            q = geo.owner(node, cells, mixed)
            if q is not None and q in ind:
                return tau_fit(*q)
        if rule == "mixed-avg":
            cs = [q for q in ind if mixed[q[0] + 1, q[1] + 1]]
            if cs:
                return avg_over(cs)
        cs = [q for q in ind if band[q[0] + 1, q[1] + 1]]
        if not cs:
            cs = ind
        return avg_over(cs)

    def stress(links, beta, factor, cells_adj, nt, node):
        n, t = nt
        e, const = {}, 0.0
        for (nm, a, c, (ip, im), thA) in links:
            e[ip] = e.get(ip, 0.0) + factor * beta / h
            e[im] = e.get(im, 0.0) - factor * beta / h
        for (_, a, c, (ip, im), thA) in links:
            k = factor * n[a] * t[c] * (1.0 - beta * R_of(thA))
            if abs(k) < 1e-300 or thA in (0.0, 1.0) and abs(1.0 - beta * R_of(thA)) < 1e-12:
                continue
            for kk, v in tau_at(cells_adj, node).items():
                e[kk] = e.get(kk, 0.0) + k * v
        return e, const

    def beta_cell(i, j, row):
        return mu_of(geo.mid_frac(i, j, ey if row == "u" else ex))

    def beta_node(i, j, row):
        return mu_of(geo.node_frac(i, j, ex if row == "u" else ey, "box"))

    cache = {}

    def sig_cell(i, j, comp):
        key = ("c", i, j, comp)
        if key not in cache:
            L = cell_links(i, j)[comp]
            cache[key] = stress([L], beta_cell(i, j, "u" if comp == 0 else "v"), 2.0, [(i, j)], geo.cell_nt(i, j), None)
        return cache[key]

    def sig_node(i, j, row):
        key = ("n", i, j, row)
        if key not in cache:
            L = node_links(i, j)
            cache[key] = stress(L, beta_node(i, j, row), 1.0, [(i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j)],
                                geo.node_nt(i, j), (i, j))
        return cache[key]

    uidx = {}
    for i in range(N + 1):
        for j in range(-1, N + 1):
            if m.unknown_u(i, j):
                uidx[m.iu(i, j)] = len(uidx)
    for i in range(-1, N + 1):
        for j in range(N + 1):
            if m.unknown_v(i, j):
                uidx[m.iv(i, j)] = len(uidx)
    nvel = len(uidx)
    npr = N * N
    ntot = nvel + npr + 1
    pid = lambda i, j: nvel + i * N + j
    xall = np.zeros(m.nu + m.nv)
    for i in range(N + 1):
        for j in range(-1, N + 1):
            xall[m.iu(i, j)] = case.face_avg(0, i * h, (j + 0.5) * h, ey)
    for i in range(-1, N + 1):
        for j in range(N + 1):
            xall[m.iv(i, j)] = case.face_avg(1, (i + 0.5) * h, j * h, ex)
    planar_f = case.force() if hasattr(case, "force") else None
    rowf = (lambda comp, x, y: planar_f[comp]) if planar_f is not None else case.row_force
    R, Cl, V = [], [], []
    b = np.zeros(ntot)

    def add(row, expr, const, sgn):
        for k, v in expr.items():
            if k in uidx:
                R.append(row); Cl.append(uidx[k]); V.append(sgn * v)
            else:
                b[row] -= sgn * v * xall[k]
        b[row] -= sgn * const

    def add_J(row, ci, cj, w):
        tau_fit(ci, cj)
        add(row, {k: w * 2 * (muB - muA) * v for k, v in dnnexpr[(ci, cj)].items()}, 0.0, 1.0)

    for i in range(N + 1):
        for j in range(N):
            if not m.unknown_u(i, j):
                continue
            r = uidx[m.iu(i, j)]
            e, c0 = sig_cell(i, j, 0); add(r, e, c0, 1.0 / h)
            e, c0 = sig_cell(i - 1, j, 0); add(r, e, c0, -1.0 / h)
            e, c0 = sig_node(i, j + 1, "u"); add(r, e, c0, 1.0 / h)
            e, c0 = sig_node(i, j, "u"); add(r, e, c0, -1.0 / h)
            R += [r, r]; Cl += [pid(i, j), pid(i - 1, j)]; V += [-1.0 / h, 1.0 / h]
            b[r] -= rowf(0, i * h, (j + 0.5) * h)
            for (ci, sg) in ((i, 1.0), (i - 1, -1.0)):
                dq = geo.mid_frac(ci, j, ey) - Cc(ci, j)
                if abs(dq) > 1e-15:
                    add_J(r, ci, j, sg * dq / h)
    for i in range(N):
        for j in range(N + 1):
            if not m.unknown_v(i, j):
                continue
            r = uidx[m.iv(i, j)]
            e, c0 = sig_node(i + 1, j, "v"); add(r, e, c0, 1.0 / h)
            e, c0 = sig_node(i, j, "v"); add(r, e, c0, -1.0 / h)
            e, c0 = sig_cell(i, j, 1); add(r, e, c0, 1.0 / h)
            e, c0 = sig_cell(i, j - 1, 1); add(r, e, c0, -1.0 / h)
            R += [r, r]; Cl += [pid(i, j), pid(i, j - 1)]; V += [-1.0 / h, 1.0 / h]
            b[r] -= rowf(1, (i + 0.5) * h, j * h)
            for (cj, sg) in ((j, 1.0), (j - 1, -1.0)):
                dq = geo.mid_frac(i, cj, ex) - Cc(i, cj)
                if abs(dq) > 1e-15:
                    add_J(r, i, cj, sg * dq / h)
    for i in range(N):
        for j in range(N):
            r = pid(i, j)
            div = {m.iu(i + 1, j): 1.0 / h, m.iu(i, j): -1.0 / h, m.iv(i, j + 1): 1.0 / h, m.iv(i, j): -1.0 / h}
            add(r, div, 0.0, 1.0)
            R.append(r); Cl.append(ntot - 1); V.append(1.0)
            R.append(ntot - 1); Cl.append(r); V.append(1.0)
    A = sp.csr_matrix((V, (R, Cl)), shape=(ntot, ntot))
    return A, b, uidx, xall


def solve_err(case, geo, rule="band-avg"):
    A, b, uidx, xall = build(case, geo, rule)
    x = spla.spsolve(A.tocsc(), b)
    ex = np.array([xall[k] for k in uidx])
    err = np.abs(x[: len(uidx)] - ex)
    scale = np.max(np.abs(ex))
    return float(err.max() / scale), float(err.mean() / scale)
