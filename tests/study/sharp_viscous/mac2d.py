"""A-priori operator test for a sharp viscous stress on a 2-D MAC grid (architect, 2026-10-31).

Steady two-phase Stokes  div(2 mu D(u)) - grad p + f = 0, div u = 0  on [0,1]^2 with a planar interface
phi = n.(x - c) - s (side A: phi < 0).  Dirichlet data = exact FACE-AVERAGED velocities on every node outside
the unknown set (boundary faces and one ghost row).  The MAC unknown is the face-averaged velocity, so the
error is measured against exact face averages.  Direct sparse solve: this is the steady state of flow's
predictor + projection with any lagging of explicit terms.

Schemes (sample stress on cell centres = normal, on nodes = shear):
  arith  : beta_cell = mu(C), beta_node = mean of 4 cells          (flow default)
  harm4  : beta_node = harmonic of 4 cells                          (register Q4 option)
  facef  : exact PLIC face fraction per row, arithmetic             ("sub-cell mu from PLIC")
  linkh  : each link difference divided by its own resistance       (GFM / Liu-Fedkiw-Kang)
  sharp  : beta m + sum_l T_l(tau)(1 - beta Rbar_l), tau = interface traction
           tau: 'oracle' (exact) or 'fit' (kinked LS fit on the low-mu side gradient, 3x3 block)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

GL_X, GL_W = np.polynomial.legendre.leggauss(5)


def cdf_box(s, h):
    return np.clip((s + 0.5 * h) / h, 0.0, 1.0)


def cdf_hat(s, h):
    s = np.clip(s, -h, h)
    return np.where(s <= 0, (s + h) ** 2 / (2 * h * h), 1.0 - (h - s) ** 2 / (2 * h * h))


class Case:
    def __init__(self, N, alpha_deg, off_cells, muA, muB, kind, eps=1.0, tau0=1.0):
        self.N, self.h = N, 1.0 / N
        a = np.deg2rad(alpha_deg)
        self.n = np.array([np.cos(a), np.sin(a)])
        self.t = np.array([-np.sin(a), np.cos(a)])
        self.c = np.array([0.5, 0.5])
        self.s = off_cells * self.h
        self.muA, self.muB, self.kind, self.eps, self.tau0 = muA, muB, kind, eps, tau0

    def phi(self, x, y):
        return self.n[0] * (x - self.c[0]) + self.n[1] * (y - self.c[1]) - self.s

    def frac_line(self, x0, y0, e, kern):
        """side-A fraction of a segment through (x0,y0) along unit e, kernel 'box' (h) or 'hat' (2h)."""
        p0 = self.phi(x0, y0)
        g = self.n[0] * e[0] + self.n[1] * e[1]
        cdf = cdf_box if kern == "box" else cdf_hat
        if abs(g) < 1e-14:
            return 1.0 if p0 < 0 else 0.0
        ss = -p0 / g
        F = float(cdf(ss, self.h))
        return F if g > 0 else 1.0 - F

    def frac_cell(self, x0, y0):
        h = self.h
        poly = [(x0, y0), (x0 + h, y0), (x0 + h, y0 + h), (x0, y0 + h)]
        out = []
        for k in range(4):
            P, Q = poly[k], poly[(k + 1) % 4]
            fp, fq = self.phi(*P), self.phi(*Q)
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

    # exact solution -------------------------------------------------------------------------------------
    def vel(self, x, y):
        xi = self.phi(x, y)
        eta = self.t[0] * (x - self.c[0]) + self.t[1] * (y - self.c[1])
        mu = np.where(xi < 0, self.muA, self.muB)
        U = np.zeros_like(xi)
        ux = np.zeros_like(xi)
        uy = np.zeros_like(xi)
        if self.kind in ("couette",):
            U = self.tau0 * xi / mu
        if self.kind in ("poiseuille", "general"):
            U = (self.tau0 * xi - 0.5 * xi * xi) / mu
        ux = ux + U * self.t[0]
        uy = uy + U * self.t[1]
        if self.kind in ("extension", "general"):
            ux = ux + self.eps * (xi * self.n[0] - eta * self.t[0])
            uy = uy + self.eps * (xi * self.n[1] - eta * self.t[1])
        return ux, uy

    def force(self):
        if self.kind in ("poiseuille", "general"):
            return self.t.copy()
        return np.zeros(2)

    def tau_exact(self):
        return 0.0 if self.kind == "extension" else self.tau0

    def face_avg(self, comp, x0, y0, e):
        """average of velocity component over the face segment centred (x0,y0) along e (length h)."""
        h = self.h
        p0 = self.phi(x0, y0)
        g = self.n[0] * e[0] + self.n[1] * e[1]
        cuts = [-0.5 * h, 0.5 * h]
        if abs(g) > 1e-14:
            ss = -p0 / g
            if -0.5 * h < ss < 0.5 * h:
                cuts = [-0.5 * h, ss, 0.5 * h]
        tot = 0.0
        for k in range(len(cuts) - 1):
            a, b = cuts[k], cuts[k + 1]
            sq = 0.5 * (a + b) + 0.5 * (b - a) * GL_X
            xs, ys = x0 + sq * e[0], y0 + sq * e[1]
            v = self.vel(xs, ys)[comp]
            tot += 0.5 * (b - a) * np.dot(GL_W, v)
        return tot / h


class Mac:
    """index maps; u[i,j]: x=i h, y=(j+1/2)h, i=0..N, j=-1..N;  v[i,j]: x=(i+1/2)h, y=j h, i=-1..N, j=0..N."""

    def __init__(self, N):
        self.N = N
        self.nu = (N + 1) * (N + 2)
        self.nv = (N + 2) * (N + 1)

    def iu(self, i, j):
        return i * (self.N + 2) + (j + 1)

    def iv(self, i, j):
        return self.nu + (i + 1) * (self.N + 1) + j

    def unknown_u(self, i, j):
        return 1 <= i <= self.N - 1 and 0 <= j <= self.N - 1

    def unknown_v(self, i, j):
        return 0 <= i <= self.N - 1 and 1 <= j <= self.N - 1


FCORR = None
OWN_ONLY = None
DNN_W = "all"
CELL_VOL = False
TAU_RULE = "band-avg"
DNN_EPS = 1e-3


def build(case, scheme, tau_mode="fit"):
    N, h = case.N, case.h
    m = Mac(N)
    muA, muB = case.muA, case.muB
    ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    n, t = case.n, case.t
    # cell fractions incl. one ring of outside cells (index -1..N)
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
    Cc = lambda i, j: C[i + 1, j + 1]
    mu_of = lambda f: f * muA + (1 - f) * muB
    R_of = lambda f: f / muA + (1 - f) / muB
    lowA = muA < muB
    mulo, muhi = (muA, muB) if lowA else (muB, muA)
    rr = mulo / muhi

    # links: (name, a, c, nodes(+,-), thetaA)  link difference m = (u+ - u-)/h
    def cell_links(i, j):
        f = Cc(i, j)
        return [("xx", 0, 0, (m.iu(i + 1, j), m.iu(i, j)), f), ("yy", 1, 1, (m.iv(i, j + 1), m.iv(i, j)), f)]

    def node_links(i, j):
        thy = case.frac_line(i * h, j * h, ey, "hat")
        thx = case.frac_line(i * h, j * h, ex, "hat")
        return [("yu", 1, 0, (m.iu(i, j), m.iu(i, j - 1)), thy), ("xv", 0, 1, (m.iv(i, j), m.iv(i - 1, j)), thx)]

    # tau fit per band cell ----------------------------------------------------------------------------------
    mixed = np.zeros((N + 2, N + 2), bool)
    for i in range(-1, N + 1):
        for j in range(-1, N + 1):
            mixed[i + 1, j + 1] = 0.0 < Cc(i, j) < 1.0
    band = mixed.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            band[1:-1, 1:-1] |= mixed[1 + di:N + 1 + di, 1 + dj:N + 1 + dj]
    tauexpr = {}
    dnnexpr = {}

    def tau_fit(i, j):
        """tau_hat = mu_lo t.S_lo n with L_lo from an LS fit of all links in the 3x3 block (kinked model)."""
        if (i, j) in tauexpr:
            return tauexpr[(i, j)]
        rows, rhs = [], []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ci, cj = i + di, j + dj
                if not (0 <= ci <= N - 1 and 0 <= cj <= N - 1):
                    continue
                for L in cell_links(ci, cj):
                    rows.append(L); rhs.append(None)
        for ni in range(i - 1, i + 3):
            for nj in range(j - 1, j + 3):
                if not (0 <= ni <= N and 0 <= nj <= N):
                    continue
                for L in node_links(ni, nj):
                    rows.append(L); rhs.append(None)
        A = np.zeros((len(rows), 4))
        for r, (_, a, c, _, thA) in enumerate(rows):
            thlo = thA if lowA else 1 - thA
            A[r, 2 * a + c] += 1.0
            # (1-thlo) n_a t_c (rr-1) (t.S n),  t.S n = sum_pq (t_p n_q + t_q n_p) L_pq
            k = (1 - thlo) * n[a] * t[c] * (rr - 1)
            for p in range(2):
                for q in range(2):
                    A[r, 2 * p + q] += k * (t[p] * n[q] + t[q] * n[p])
        G = np.linalg.pinv(A)  # 4 x nl
        wv = np.array([[t[p] * n[q] + t[q] * n[p] for q in range(2)] for p in range(2)]).ravel()
        coef = mulo * (wv @ G)  # tau = coef . m
        nn = np.array([[n[p] * n[q] for q in range(2)] for p in range(2)]).ravel()
        if DNN_W == "hi":  # weight each link by its high-mu content: D_nn from the stiff side
            wl = np.array([(1 - thA) if lowA else thA for (_, _, _, _, thA) in rows]) + DNN_EPS
            cdn = nn @ (np.linalg.pinv(A * wl[:, None]) * wl[None, :])
        else:
            cdn = nn @ G  # D_nn = n.L n (continuous)
        expr, dexpr = {}, {}
        for r, (_, a, c, (ip, im), _) in enumerate(rows):
            expr[ip] = expr.get(ip, 0.0) + coef[r] / h
            expr[im] = expr.get(im, 0.0) - coef[r] / h
            dexpr[ip] = dexpr.get(ip, 0.0) + cdn[r] / h
            dexpr[im] = dexpr.get(im, 0.0) - cdn[r] / h
        tauexpr[(i, j)] = expr
        dnnexpr[(i, j)] = dexpr
        return expr

    def tau_at(cells):
        if TAU_RULE == "owner":
            cand = [(ci, cj) for (ci, cj) in cells if 0 <= ci <= N - 1 and 0 <= cj <= N - 1 and mixed[ci + 1, cj + 1]]
            if not cand:
                cand = [(ci, cj) for (ci, cj) in cells if 0 <= ci <= N - 1 and 0 <= cj <= N - 1]
            own = min(cand, key=lambda q: (abs(case.phi((q[0] + 0.5) * h, (q[1] + 0.5) * h)), q))
            return tau_fit(*own)
        if TAU_RULE == "mixed-avg":  # note section 5.7 Q-A: mean over the interfacial cells among the four
            cs = [(ci, cj) for (ci, cj) in cells if 0 <= ci <= N - 1 and 0 <= cj <= N - 1 and mixed[ci + 1, cj + 1]]
            if cs:
                e = {}
                for cc in cs:
                    for k, v in tau_fit(*cc).items():
                        e[k] = e.get(k, 0.0) + v / len(cs)
                return e
        cs = [(ci, cj) for (ci, cj) in cells if 0 <= ci <= N - 1 and 0 <= cj <= N - 1 and band[ci + 1, cj + 1]]
        if not cs:
            cs = [(ci, cj) for (ci, cj) in cells if 0 <= ci <= N - 1 and 0 <= cj <= N - 1]
        e = {}
        for cc in cs:
            for k, v in tau_fit(*cc).items():
                e[k] = e.get(k, 0.0) + v / len(cs)
        return e

    # sample stresses ---------------------------------------------------------------------------------------
    def stress(links, beta, factor, cells_adj):
        """sigma = factor*[ beta*sum m + sum_l T_l (1 - beta Rbar_l) ]  as (dict, const)."""
        e, const = {}, 0.0
        for (nm, a, c, (ip, im), thA) in links:
            if OWN_ONLY is not None and nm not in OWN_ONLY:
                continue
            e[ip] = e.get(ip, 0.0) + factor * beta / h
            e[im] = e.get(im, 0.0) - factor * beta / h
        if scheme == "sharp":
            for (_, a, c, (ip, im), thA) in links:
                k = factor * n[a] * t[c] * (1.0 - beta * R_of(thA))
                if abs(k) < 1e-300 or thA in (0.0, 1.0) and abs(1.0 - beta * R_of(thA)) < 1e-12:
                    continue
                if tau_mode == "oracle":
                    const += k * case.tau_exact()
                else:
                    for kk, v in tau_at(cells_adj).items():
                        e[kk] = e.get(kk, 0.0) + k * v
        return e, const

    def stress_linkh(links, factor):
        e = {}
        for (_, a, c, (ip, im), thA) in links:
            b = 1.0 / R_of(thA)
            e[ip] = e.get(ip, 0.0) + factor * b / h
            e[im] = e.get(im, 0.0) - factor * b / h
        return e, 0.0

    def side_mu(x, y):
        return muA if case.phi(x, y) < 0 else muB

    def beta_cell(i, j, row, side=None):
        if side is not None:
            return side
        if CELL_VOL:
            return mu_of(Cc(i, j))
        if scheme == "facef" or (scheme == "sharp" and SHARP_BASE == "facef"):
            xc, yc = (i + 0.5) * h, (j + 0.5) * h
            return mu_of(case.frac_line(xc, yc, ey if row == "u" else ex, "box"))
        return mu_of(Cc(i, j))

    def beta_node(i, j, row, side=None):
        if side is not None:
            return side
        cells = [Cc(i - 1, j - 1), Cc(i, j - 1), Cc(i - 1, j), Cc(i, j)]
        mus = [mu_of(f) for f in cells]
        if scheme == "harm4":
            return 4.0 / sum(1.0 / x for x in mus)
        if scheme == "facef" or (scheme == "sharp" and SHARP_BASE == "facef"):
            return mu_of(case.frac_line(i * h, j * h, ex if row == "u" else ey, "box"))
        return 0.25 * sum(mus)

    cache = {}
    GFM = scheme == "sharp" and SHARP_BASE == "gfm"

    def sig_cell(i, j, comp, side=None):
        key = ("c", i, j, comp, side)
        if key not in cache:
            L = cell_links(i, j)[comp]
            if scheme == "linkh":
                cache[key] = stress_linkh([L], 2.0)
            else:
                cache[key] = stress([L], beta_cell(i, j, "u" if comp == 0 else "v", side), 2.0, [(i, j)])
        return cache[key]

    def sig_node(i, j, row, side=None):
        key = ("n", i, j, row, side)
        if key not in cache:
            L = node_links(i, j)
            if OWN_ONLY is not None:
                L = [x for x in L if x[0] == ("yu" if row == "u" else "xv")]
            if scheme == "linkh":
                cache[key] = stress_linkh(L, 1.0)
            else:
                cache[key] = stress(L, beta_node(i, j, row, side), 1.0, [(i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j)])
        return cache[key]

    # assemble -----------------------------------------------------------------------------------------------
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
    # exact face averages for all nodes
    xall = np.zeros(m.nu + m.nv)
    for i in range(N + 1):
        for j in range(-1, N + 1):
            xall[m.iu(i, j)] = case.face_avg(0, i * h, (j + 0.5) * h, ey)
    for i in range(-1, N + 1):
        for j in range(N + 1):
            xall[m.iv(i, j)] = case.face_avg(1, (i + 0.5) * h, j * h, ex)
    f = case.force()
    R, Cl, V = [], [], []

    def add_J(row, ci, cj, w):
        """row += w * J(cell), J = 2 (muB - muA) D_nn."""
        if FCORR == "oracle":
            dnn = case.eps if case.kind in ("extension", "general") else 0.0
            b[row] -= w * 2 * (muB - muA) * dnn
        else:
            tau_fit(ci, cj)
            add(row, {k: w * 2 * (muB - muA) * v for k, v in dnnexpr[(ci, cj)].items()}, 0.0, 1.0)
    b = np.zeros(ntot)

    def add(row, expr, const, sgn):
        nonlocal b
        for k, v in expr.items():
            if k in uidx:
                R.append(row); Cl.append(uidx[k]); V.append(sgn * v)
            else:
                b[row] -= sgn * v * xall[k]
        b[row] -= sgn * const

    for i in range(N + 1):
        for j in range(N):
            if not m.unknown_u(i, j):
                continue
            r = uidx[m.iu(i, j)]
            sd = side_mu(i * h, (j + 0.5) * h) if GFM else None
            e, c0 = sig_cell(i, j, 0, sd); add(r, e, c0, 1.0 / h)
            e, c0 = sig_cell(i - 1, j, 0, sd); add(r, e, c0, -1.0 / h)
            e, c0 = sig_node(i, j + 1, "u", sd); add(r, e, c0, 1.0 / h)
            e, c0 = sig_node(i, j, "u", sd); add(r, e, c0, -1.0 / h)
            R += [r, r]; Cl += [pid(i, j), pid(i - 1, j)]; V += [-1.0 / h, 1.0 / h]
            b[r] -= f[0]
            if FCORR:
                for (ci, sg) in ((i, 1.0), (i - 1, -1.0)):
                    xc, yc = (ci + 0.5) * h, (j + 0.5) * h
                    dq = case.frac_line(xc, yc, ey, "box") - Cc(ci, j)
                    if abs(dq) > 1e-15:
                        add_J(r, ci, j, sg * dq / h)
    for i in range(N):
        for j in range(N + 1):
            if not m.unknown_v(i, j):
                continue
            r = uidx[m.iv(i, j)]
            sd = side_mu((i + 0.5) * h, j * h) if GFM else None
            e, c0 = sig_node(i + 1, j, "v", sd); add(r, e, c0, 1.0 / h)
            e, c0 = sig_node(i, j, "v", sd); add(r, e, c0, -1.0 / h)
            e, c0 = sig_cell(i, j, 1, sd); add(r, e, c0, 1.0 / h)
            e, c0 = sig_cell(i, j - 1, 1, sd); add(r, e, c0, -1.0 / h)
            R += [r, r]; Cl += [pid(i, j), pid(i, j - 1)]; V += [-1.0 / h, 1.0 / h]
            b[r] -= f[1]
            if FCORR:
                for (cj, sg) in ((j, 1.0), (j - 1, -1.0)):
                    xc, yc = (i + 0.5) * h, (cj + 0.5) * h
                    dq = case.frac_line(xc, yc, ex, "box") - Cc(i, cj)
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
    return A, b, uidx, xall, m


SHARP_BASE = "arith"


def solve_err(case, scheme, tau_mode="fit"):
    A, b, uidx, xall, m = build(case, scheme, tau_mode)
    x = spla.spsolve(A.tocsc(), b)
    ex = np.array([xall[k] for k in uidx])
    uh = x[: len(uidx)]
    err = np.abs(uh - ex)
    scale = np.max(np.abs(ex))
    return err.max() / scale, err.mean() / scale
