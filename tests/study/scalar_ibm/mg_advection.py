"""2-D prototype (architect, Amendment A2 to doc/scalar_ibm_design.md): how should the STEADY cut-cell
scalar solve with implicit first-order upwind (FOU) advection be preconditioned?

Operator (per unit cell volume, D = 1, steady, periodic box [0,1]^2, n x n cells):
  A = aperture 5-point diffusion (a D / h^2 per face)  +  lumped Dirichlet wall sink W = alpha / s0
      (A1's level-0 term; Neumann discs: W = 0, singular)  +  conservative FOU advection from a
      discretely divergence-free face flux f = F/V (node stream function, zero inside the discs,
      no-slip ramp outside them).  The probe off-diagonals are left out on purpose: they are the
      SAME gap at every Pe and the WO-3/4 evidence already prices them (A = S here).
Preconditioners (one V-cycle, BiCGStab on A, rtol 1e-10; the 3-D code's cycle: RB-GS from global
parity, pre nu R->B, post nu B->R, average restriction, bilinear prolongation, pinned re-zeroed,
coarse faces rediscretized, coarse wall = A1 average at s0; bottom 4x4 exact):
  l0only: level 0 = A (FOU couplings), coarse levels = the 'sym' ones (is coarse advection needed?)
  sym   : the shipped WO-5 surrogate - symmetric diffusion + wall + LUMPED outflow on the diagonal,
          the outflow restricted by average onto the coarse levels (WO-5's restrictAvg of omega)
  split : FOU couplings IN the surrogate; coarse advection = the piecewise-constant Galerkin value:
          per coarse face the summed positive parts of the sub-face fluxes, each direction kept
          (q+_C = sum_sub max(f, 0) / N_ratio, q-_C likewise), so every level is a conservative
          M-matrix with zero column sums
  net   : FOU couplings in the surrogate; coarse face flux = summed SIGNED sub-face fluxes, upwinded
  split-pc / split-nu3: piecewise-constant prolongation / 3 + 3 sweeps (variants, not shipped)
Cases: P1 periodic + Dirichlet disc R = L/4 (WO-5's C3 analogue), P2 Neumann 2x2 disc array (closure,
singular), P4 closed-streamline eddies without a solid (singular), P3 the P2 array with Dirichlet discs.
Output per case: BiCGStab iterations [V-cycle contraction rho(I - M^-1 A), power estimate];
"n!(r)" = not converged in 300, true relative residual r.
Run:  python mg_advection.py [n ...] [case prefixes] [variants] [pe=a,b,..] [gmres] [wcycle|fcycle]
      [bottom=sweeps] [nmin=16]        (OMP_NUM_THREADS=4; n = 64..256 for all cases: ~35 s)
Results: doc/scalar_ibm_design.md, Amendment A2 (under §6.7). 'split' is what A2 ships.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, sys, time

SNAP = 1e-3


# ------------------------------------------------------------------ geometry and flow (vectorized)
def edge_frac(pa, pb):
    """fluid fraction of an edge from its end-node SDF values (> 0 fluid), snapped at 1e-3."""
    out = np.where((pa > 0) & (pb > 0), 1.0, 0.0)
    mixed = (pa > 0) != (pb > 0)
    p = np.where(pa > 0, pa, pb); q = np.where(pa > 0, pb, pa)
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = np.where(mixed, p / (p - q), 0.0)
    out = np.where(mixed, fr, out)
    return np.where(out < SNAP, 0.0, np.where(out > 1 - SNAP, 1.0, out))


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


class Case:
    """n x n periodic cells on [0,1]^2; discs (cx, cy, R); flow direction d; wall 'dir' or 'neu'."""
    def __init__(self, n, discs, wall, d=(1.0, 0.3), delta_frac=0.8, cells=0.0):
        self.n, self.h = n, 1.0 / n
        h = self.h
        xn = np.arange(n + 1) * h
        X, Y = np.meshgrid(xn, xn, indexing="ij")          # nodes (n+1)^2
        sdf = np.full(X.shape, np.inf)
        Ux, Uy = d
        lin = lambda x, y: Ux * y - Uy * x
        psi = lin(X, Y)
        if cells:  # closed-streamline eddies (4 per period) on top of the mean flow
            psi = psi + cells * np.hypot(Ux, Uy) * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) / (2 * np.pi)
        for (cx, cy, R) in discs:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    r = np.hypot(X - cx - ox, Y - cy - oy)
                    sdf = np.minimum(sdf, r - R)
        # ramp width: delta_frac of the smallest clearance between disc surfaces (incl. images)
        cl = np.inf
        for k, (cx, cy, R) in enumerate(discs):
            for m_, (dx, dy, Rm) in enumerate(discs):
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        if k == m_ and ox == 0 and oy == 0:
                            continue
                        cl = min(cl, np.hypot(cx - dx - ox, cy - dy - oy) - R - Rm)
        self.delta = delta_frac * 0.5 * cl if np.isfinite(cl) else 1.0
        for (cx, cy, R) in discs:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    r = np.hypot(X - cx - ox, Y - cy - oy)
                    m = smoothstep((r - R) / self.delta)
                    psi = psi + (1 - m) * (lin(cx + ox, cy + oy) - lin(X, Y))
        # apertures of the LOW faces of cell (i, j): ax[i, j] = x-face at x_i, ay[i, j] = y-face at y_j
        self.ax = edge_frac(sdf[:-1, :-1], sdf[:-1, 1:])   # nodes (i, j) - (i, j+1)
        self.ay = edge_frac(sdf[:-1, :-1], sdf[1:, :-1])   # nodes (i, j) - (i+1, j)
        hi = lambda a, ax_: np.roll(a, -1, axis=ax_)
        self.unknown = (self.ax + hi(self.ax, 0) + self.ay + hi(self.ay, 1)) > 0
        Axv = (hi(self.ax, 0) - self.ax) * h; Ayv = (hi(self.ay, 1) - self.ay) * h
        Aw = np.hypot(Axv, Ayv)
        with np.errstate(invalid="ignore", divide="ignore"):
            nx = np.where(Aw > 0, Axv / np.maximum(Aw, 1e-300), 0); ny = np.where(Aw > 0, Ayv / np.maximum(Aw, 1e-300), 0)
        alpha = np.where(self.unknown & (Aw > 1e-14 * h), Aw / h**2, 0.0)
        s0 = 1.1 * 0.5 * (np.abs(nx) + np.abs(ny)) * h
        self.W0 = np.where(alpha > 0, alpha / np.maximum(s0, 1e-300), 0.0) if wall == "dir" else np.zeros((n, n))
        self.singular = wall == "neu"
        # face fluxes F (per depth) of the low faces, from node psi; f = F / h^2 per unit volume
        Fx = psi[:-1, 1:] - psi[:-1, :-1]
        Fy = -(psi[1:, :-1] - psi[:-1, :-1])
        unk = self.unknown
        gx = unk & np.roll(unk, 1, axis=0); gy = unk & np.roll(unk, 1, axis=1)   # both cells unknown
        self.dropped = max(np.abs(np.where(gx, 0, Fx)).max(), np.abs(np.where(gy, 0, Fy)).max()) / max(np.abs(Fx).max(), 1e-300)
        Fx = np.where(gx, Fx, 0.0); Fy = np.where(gy, Fy, 0.0)
        div = (np.roll(Fx, -1, 0) - Fx) + (np.roll(Fy, -1, 1) - Fy)
        self.divrel = np.abs(div).max() / np.abs(Fx).max()
        self.Fmax = max(np.abs(Fx).max(), np.abs(Fy).max())
        self.fx0 = Fx / h**2; self.fy0 = Fy / h**2

    def flux(self, pe):
        """per-unit-volume face fluxes scaled to peak cell Peclet pe = max|u| h / D (|u| = |F| / h)."""
        s = pe / self.Fmax if self.Fmax > 0 else 0.0  # max|F| = pe (D = 1): max f = pe / h^2
        return self.fx0 * s, self.fy0 * s


# ------------------------------------------------------------------ level operators
def level_op(n, h, tx, ty, px, mx, py, my, unk, W, omega=None):
    """per-volume operator; tx/ty diffusion coefficients (already / h^2), px/mx (py/my) advective
    coefficients of the LOW faces: p = flow toward +axis (low cell -> high cell), m = toward -axis.
    omega: lumped extra diagonal (the 'sym' baseline). Identity rows on pinned cells."""
    idx = np.arange(n * n).reshape(n, n)
    gx = unk & np.roll(unk, 1, 0); gy = unk & np.roll(unk, 1, 1)
    tx = np.where(gx, tx, 0); ty = np.where(gy, ty, 0)
    px = np.where(gx, px, 0); mx = np.where(gx, mx, 0); py = np.where(gy, py, 0); my = np.where(gy, my, 0)
    up = lambda a, ax_: np.roll(a, -1, ax_)   # the HIGH face of cell (i, j)
    diag = W + tx + up(tx, 0) + ty + up(ty, 1) + mx + up(px, 0) + my + up(py, 1)
    if omega is not None:
        diag = diag + omega
    rows, cols, vals = [], [], []
    for (coef, sh) in ((tx + px, (1, 0)), (up(tx, 0) + up(mx, 0), (-1, 0)),
                       (ty + py, (0, 1)), (up(ty, 1) + up(my, 1), (0, -1))):
        nb = np.roll(idx, sh, axis=(0, 1))       # neighbour index: roll(+1) gives the LOW neighbour
        m = unk & (coef != 0)
        rows.append(idx[m]); cols.append(nb[m]); vals.append(-coef[m])
    diag = np.where(unk, diag, 1.0)
    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())
    r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
    return sp.csr_matrix((v, (r, c)), shape=(n * n, n * n))


def transfers(n, prol):
    nc = n // 2
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    fine = (I * n + J).ravel()
    R = sp.csr_matrix((np.full(n * n, 0.25), (((I // 2) * nc + J // 2).ravel(), fine)), shape=(nc * nc, n * n))
    if prol == "pc":
        return R, (4 * R).T.tocsr()
    si = np.where(I % 2, 1, -1); sj = np.where(J % 2, 1, -1)
    rows, cols, vals = [], [], []
    for (dI, dJ, w) in ((0, 0, 9 / 16), (1, 0, 3 / 16), (0, 1, 3 / 16), (1, 1, 1 / 16)):
        II = (I // 2 + dI * si) % nc; JJ = (J // 2 + dJ * sj) % nc
        rows.append(fine); cols.append((II * nc + JJ).ravel()); vals.append(np.full(n * n, w))
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, nc * nc))
    return R, P


NMIN = 4  # bottom grid edge (the 3-D distributed table can stop at a few cells per RANK)


def hierarchy(C, pe, mode, prol, nmin=None):
    nmin = nmin or NMIN
    n, h = C.n, C.h
    fx, fy = C.flux(pe)
    ax, ay, unk = C.ax, C.ay, C.unknown
    W0 = C.W0
    px, mx, py, my = np.maximum(fx, 0), np.maximum(-fx, 0), np.maximum(fy, 0), np.maximum(-fy, 0)
    # the TRUE operator (A = S with FOU couplings, lumped wall)
    A = level_op(n, h, ax / h**2, ay / h**2, px, mx, py, my, unk, W0)
    levels = []
    if mode in ("sym", "l0only"):
        Z = np.zeros((n, n))
        om = (mx + np.roll(px, -1, 0) + my + np.roll(py, -1, 1)) * unk   # lumped outflow per cell
        S = level_op(n, h, ax / h**2, ay / h**2, Z, Z, Z, Z, unk, W0, omega=om)
    if mode != "sym":
        S = A
    hh, L = h, 0
    while True:
        levels.append(dict(A=S, unk=unk.ravel(), n=n))
        if n // 2 < nmin:
            break
        R, P = transfers(n, prol)
        levels[-1].update(R=R, P=P)
        nc = n // 2; H = hh * 2
        unk_c = unk.reshape(nc, 2, nc, 2).any(axis=(1, 3))
        axc = 0.5 * (ax[0::2, 0::2] + ax[0::2, 1::2]); ayc = 0.5 * (ay[0::2, 0::2] + ay[1::2, 0::2])
        f = 2 ** (L + 1)
        Wc = W0.reshape(nc, f, nc, f).mean(axis=(1, 3))
        if mode in ("sym", "l0only"):
            om = om.reshape(nc, 2, nc, 2).mean(axis=(1, 3))
            Z = np.zeros((nc, nc))
            S = level_op(nc, H, axc / H**2, ayc / H**2, Z, Z, Z, Z, unk_c, Wc, omega=om)
        else:
            if mode == "split":
                px = 0.25 * (px[0::2, 0::2] + px[0::2, 1::2]); mx = 0.25 * (mx[0::2, 0::2] + mx[0::2, 1::2])
                py = 0.25 * (py[0::2, 0::2] + py[1::2, 0::2]); my = 0.25 * (my[0::2, 0::2] + my[1::2, 0::2])
            else:  # net
                fx = 0.25 * (fx[0::2, 0::2] + fx[0::2, 1::2]); fy = 0.25 * (fy[0::2, 0::2] + fy[1::2, 0::2])
                px, mx, py, my = np.maximum(fx, 0), np.maximum(-fx, 0), np.maximum(fy, 0), np.maximum(-fy, 0)
            S = level_op(nc, H, axc / H**2, ayc / H**2, px, mx, py, my, unk_c, Wc)
        ax, ay, unk, n, hh, L = axc, ayc, unk_c, nc, H, L + 1
    return A, levels


# ------------------------------------------------------------------ cycle
def prep(levels):
    for lv in levels:
        n = lv["n"]
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        col = ((I + J) % 2).ravel()
        A = lv["A"]
        lv["sets"] = []
        for c in (0, 1):
            s = np.nonzero(col == c)[0]
            lv["sets"].append((s, A[s].tocsr(), A.diagonal()[s]))
        if "R" not in lv:
            lv["dense"] = np.linalg.pinv(A.toarray()) if True else None


def sweep(lv, x, b, order):
    for c in order:
        s, As, Ds = lv["sets"][c]
        x[s] += (b[s] - As @ x) / Ds
    return x


GAMMA = 1         # 2: W-cycle, 3: F-cycle (prototype levers only)
BOTTOM = "exact"   # or "sweeps": the 3-D code's 16 sweeps (8 R->B + 8 B->R) + mean removal


def vcycle(levels, l, b, singular, nu, kind=None):
    kind = kind or ("W" if GAMMA == 2 else ("F" if GAMMA == 3 else "V"))
    lv = levels[l]
    if "R" not in lv:
        if BOTTOM == "sweeps":
            x = np.zeros_like(b)
            for _ in range(8):
                x = sweep(lv, x, b, (0, 1))
            for _ in range(8):
                x = sweep(lv, x, b, (1, 0))
        else:
            x = lv["dense"] @ b
        if singular:
            u = lv["unk"]; x[u] -= x[u].mean()
        return x
    x = np.zeros_like(b)
    for _ in range(nu):
        x = sweep(lv, x, b, (0, 1))
    r = b - lv["A"] @ x
    rc = lv["R"] @ r
    if singular:
        u = levels[l + 1]["unk"]; rc[u] -= rc[u].mean(); rc[~u] = 0
    ec = vcycle(levels, l + 1, rc, singular, nu, kind)
    if kind in ("W", "F") and "R" in levels[l + 1]:  # W: second W visit; F: F then one V visit
        A1_ = levels[l + 1]["A"]
        ec = ec + vcycle(levels, l + 1, rc - A1_ @ ec, singular, nu, "W" if kind == "W" else "V")
    x += lv["P"] @ ec
    x[~lv["unk"]] = 0
    for _ in range(nu):
        x = sweep(lv, x, b, (1, 0))
    return x


def precond(levels, singular, nu):
    u = levels[0]["unk"]
    def apply(r):
        z = vcycle(levels, 0, r.copy(), singular, nu)
        if singular:
            z[u] -= z[u].mean()
        return z
    return apply


def contraction(A, levels, singular, nu, its=40):
    u = levels[0]["unk"]; M = precond(levels, singular, nu)
    rng = np.random.default_rng(3)
    e = np.where(u, rng.standard_normal(A.shape[0]), 0.0)
    hist = []
    for _ in range(its):
        if singular:
            e[u] -= e[u].mean()
        nrm = np.linalg.norm(e)
        e = e - M(A @ e); e[~u] = 0
        if singular:
            e[u] -= e[u].mean()
        hist.append(np.linalg.norm(e) / nrm)
        if not np.isfinite(hist[-1]) or hist[-1] > 1e6:
            return float("inf")
    return float(np.exp(np.mean(np.log(hist[-10:]))))


def krylov_its(A, levels, singular, nu, method="bicgstab", maxit=300):
    u = levels[0]["unk"]
    rng = np.random.default_rng(5)
    b = np.where(u, 1 + 0.3 * rng.standard_normal(A.shape[0]), 0.0)
    if singular:
        b[u] -= b[u].mean()
    M = spla.LinearOperator(A.shape, matvec=precond(levels, singular, nu))
    cnt = [0]
    cb = lambda _: cnt.__setitem__(0, cnt[0] + 1)
    if method == "bicgstab":
        x, info = spla.bicgstab(A, b, M=M, rtol=1e-10, atol=0.0, maxiter=maxit, callback=cb)
    else:
        x, info = spla.gmres(A, b, M=M, rtol=1e-10, atol=0.0, restart=maxit, maxiter=1, callback=cb, callback_type="pr_norm")
    rel = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    return cnt[0] if (info == 0 and rel < 1e-8) else f"{cnt[0]}!({rel:.0e})"


CASES = {
    # WO-5's C3 analogue: periodic, one Dirichlet disc R = L/4 (R/h = 16 at n = 64), uniform-ish flow
    "P1 Dirichlet disc R=L/4": dict(discs=[(0.5 + 0.37 / 64, 0.5 + 0.37 / 64, 0.25)], wall="dir"),
    # closure-like: periodic 2x2 array of Neumann discs, porosity ~0.50, singular
    "P2 Neumann 2x2 array (closure)": dict(discs=[(0.25 + i * 0.5 + 0.37 / 64, 0.25 + j * 0.5 + 0.37 / 64, 0.2) for i in (0, 1) for j in (0, 1)], wall="neu"),
    # closed streamlines (the hard case for advective MG): eddies of amplitude 5x the mean flow,
    # no solid, singular (closure-like)
    "P4 cellular eddies, no solid": dict(discs=[], wall="neu", cells=5.0),
    # reactive packed bed: the same array with Dirichlet discs
    "P3 Dirichlet 2x2 array": dict(discs=[(0.25 + i * 0.5 + 0.37 / 64, 0.25 + j * 0.5 + 0.37 / 64, 0.2) for i in (0, 1) for j in (0, 1)], wall="dir"),
}

VARIANTS = {  # name: (mode, prolongation, nu)
    "sym": ("sym", "bil", 2),
    "split": ("split", "bil", 2),
    "net": ("net", "bil", 2),
    "split-pc": ("split", "pc", 2),
    "split-nu3": ("split", "bil", 3),
    "l0only": ("l0only", "bil", 2),   # FOU couplings at level 0 only; coarse = the 'sym' levels
}

if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:] if a.isdigit()] or [64, 128]
    pes = [0.0, 0.1, 1.0, 10.0]
    for a in sys.argv[1:]:
        if a.startswith("pe="):
            pes = [float(v) for v in a[3:].split(",")]
    vsel = [a for a in sys.argv[1:] if a in VARIANTS] or list(VARIANTS)
    csel = [k for k in CASES if any(k.startswith(a) for a in sys.argv[1:])] or list(CASES)
    gm = "gmres" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("bottom="):
            BOTTOM = a[7:]
        if a == "wcycle":
            GAMMA = 2
        if a == "fcycle":
            GAMMA = 3
        if a.startswith("nmin="):
            NMIN = int(a[5:])
    for name in csel:
        for n in ns:
            C = Case(n, **CASES[name])
            print(f"\n{name}, n = {n}: div/max|F| = {C.divrel:.1e}, dropped flux {C.dropped:.1e}, "
                  f"unknowns {C.unknown.sum()}, ramp {C.delta / C.h:.1f} h", flush=True)
            for v in vsel:
                mode, prol, nu = VARIANTS[v]
                line = f"  {v:10s}"
                for pe in pes:
                    t = time.time()
                    A, lv = hierarchy(C, pe, mode, prol)
                    prep(lv)
                    its = krylov_its(A, lv, C.singular, nu)
                    rho = contraction(A, lv, C.singular, nu)
                    extra = ""
                    if gm:
                        extra = f" g{krylov_its(A, lv, C.singular, nu, 'gmres')}"
                    line += f" | Pe_h {pe:g}: {its} [{rho:.3f}]{extra}"
                print(line, flush=True)
