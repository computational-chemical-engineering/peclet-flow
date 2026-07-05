"""A-priori test of the fully-FV cut-cell wall viscous flux with 7-point ingredients.

Estimator (what the proposed FV collocated momentum assembly would compute):
  F_est = mu * sum_cells sum_a W_a * g_a
  - W_a = h^2 (o_{a-} - o_{a+}) : the wall fragment's area-weighted fluid-outward normal,
          FREE from the face openness by the divergence theorem (no surface reconstruction).
  - g_a : d(ux)/da at the wall from the UNIDIRECTIONAL wall-anchored quadratic along axis a
          through {u(wall)=0, U_c, U_far} -- strictly 7-point data. Central difference where
          the axis does not cross the wall; solid-center cells borrow the adjacent fluid
          cell's gradient vector.
Reference: F_exact = mu * closed-surface integral of grad(ux).n dA (n = fluid-outward = -rhat),
by Fibonacci quadrature of the analytic Stokes solution. Note grad(ux) = n * d(ux)/dn EXACTLY
on the wall (no-slip kills tangential derivatives), so the axis decomposition misses nothing
structurally; the test measures fit truncation + evaluation-point displacement + attribution.
Face openness computed semi-analytically (256-point chord integration of the exact disk cut),
so the fragment identity is near-exact and the GRADIENT estimator is what is being tested.
"""
import numpy as np

R = 0.3102
C0 = np.array([0.013, -0.007, 0.004])
MU = 0.1

def ux(x, y, z):
    dx, dy, dz = x - C0[0], y - C0[1], z - C0[2]
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    return 1.0 - 3*R/4*(1/r + dx*dx/r**3) - R**3/4*(1/r**3 - 3*dx*dx/r**5)

def sdf(x, y, z):
    return np.sqrt((x-C0[0])**2 + (y-C0[1])**2 + (z-C0[2])**2) - R

def reference(M=200000, d=1e-6):
    i = np.arange(M); ga = np.pi*(3-np.sqrt(5))
    zz = 1 - 2*(i+0.5)/M; rr = np.sqrt(1-zz*zz); th = ga*i
    nx, ny, nz = rr*np.cos(th), rr*np.sin(th), zz
    px, py, pz = C0[0]+R*nx, C0[1]+R*ny, C0[2]+R*nz
    gx = (ux(px+d,py,pz)-ux(px-d,py,pz))/(2*d)
    gy = (ux(px,py+d,pz)-ux(px,py-d,pz))/(2*d)
    gz = (ux(px,py,pz+d)-ux(px,py,pz-d))/(2*d)
    return MU*np.mean(gx*(-nx)+gy*(-ny)+gz*(-nz))*4*np.pi*R*R

def face_openness(N, axis, h):
    """Fluid area fraction of every face perpendicular to `axis`, shape (N+1,N,N) in
    (plane, t1, t2) order; t1,t2 = the other two axes in cyclic order. Semi-analytic:
    the sphere cuts the face plane in a disk; per cut face integrate the chord (256 pts)."""
    a1, a2 = (axis+1) % 3, (axis+2) % 3
    w = np.arange(N+1)*h - 0.5                    # plane coordinates
    t1lo = np.arange(N)*h - 0.5                   # square low edges
    t2lo = np.arange(N)*h - 0.5
    rho2 = R*R - (w - C0[axis])**2                # disk radius^2 per plane
    O = np.ones((N+1, N, N))
    c1, c2 = C0[a1], C0[a2]
    for k in np.nonzero(rho2 > 0)[0]:
        rho = np.sqrt(rho2[k])
        # candidate squares: within rho+h*sqrt2 of the disk center
        j1 = np.nonzero(np.abs(t1lo + h/2 - c1) < rho + h)[0]
        j2 = np.nonzero(np.abs(t2lo + h/2 - c2) < rho + h)[0]
        if len(j1) == 0 or len(j2) == 0:
            continue
        J1, J2 = np.meshgrid(j1, j2, indexing="ij")
        lo1 = t1lo[J1.ravel()]; lo2 = t2lo[J2.ravel()]
        # 256-point midpoint rule along t1; exact chord in t2
        t = lo1[:, None] + (np.arange(256)[None, :] + 0.5)*(h/256)
        s2 = rho*rho - (t - c1)**2                          # chord half-width^2
        half = np.sqrt(np.maximum(s2, 0.0))
        zlo = np.maximum(c2 - half, lo2[:, None])
        zhi = np.minimum(c2 + half, lo2[:, None] + h)
        solid_frac = np.clip(zhi - zlo, 0.0, None).mean(axis=1)/h
        O[k, J1.ravel(), J2.ravel()] = 1.0 - solid_frac
    return O  # (plane, t1, t2)

def run(N, use_far2=False):
    h = 1.0/N
    c = (np.arange(N)+0.5)*h - 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    S = sdf(X, Y, Z)
    fl = S >= 0
    U = np.where(fl, ux(X, Y, Z), 0.0)

    # face openness -> per-cell W_a = h^2 (o_minus - o_plus), axes ordered (x,y,z)
    W = []
    for a in range(3):
        O = face_openness(N, a, h)                 # (plane, t1, t2) with t1,t2 = cyclic
        om = O[:-1]; op = O[1:]                    # minus/plus face of each cell, cyclic order
        Wa = h*h*(om - op)                         # (N,N,N) in (a, a+1, a+2) axis order
        # reorder to (x,y,z)
        Wa = np.moveaxis(Wa, [0, 1, 2], [a, (a+1) % 3, (a+2) % 3])
        W.append(Wa)
    sumW = [float(W[a].sum()) for a in range(3)]
    area = float(np.sqrt(W[0]**2 + W[1]**2 + W[2]**2).sum())

    # unidirectional wall gradients per axis (fluid-center cells)
    g = [np.zeros((N, N, N)) for _ in range(3)]
    gc = [np.zeros((N, N, N)) for _ in range(3)]   # central-only baseline
    for a in range(3):
        up = np.roll(U, -1, a); um = np.roll(U, +1, a)
        sp = np.roll(S, -1, a); sm = np.roll(S, +1, a)
        solP = sp < 0; solM = sm < 0
        thP = np.clip(np.where(solP, S/np.maximum(S-sp, 1e-30), 1.0), 1e-3, 1.0)
        thM = np.clip(np.where(solM, S/np.maximum(S-sm, 1e-30), 1.0), 1e-3, 1.0)
        # wall-anchored quadratic through (wall=0 at th), U_c at 0, U_far at -1 (7-point):
        #   alpha = [U_c (th+1)^2 - U_far th^2] / [th (th+1)] ; du/ds|wall = -alpha (s toward solid)
        def alpha(th, Uc, Uf, farFluid):
            quad = (Uc*(th+1.0)**2 - Uf*th*th)/(th*(th+1.0))
            lin = Uc/th
            return np.where(farFluid, quad, lin)
        aP = alpha(thP, U, um, ~solM)   # solid at +a -> far is -a neighbor
        aM = alpha(thM, U, up, ~solP)
        cen = (up - um)/(2*h)
        both = solP & solM  # two-walled along this axis (rare): average the two one-sided estimates
        gA = np.where(solP & ~solM, -aP/h,
              np.where(solM & ~solP, +aM/h,
               np.where(both, 0.5*(-aP + aM)/h, cen)))
        g[a] = np.where(fl, gA, 0.0)
        gc[a] = np.where(fl, cen, 0.0)

    # solid-center cells with wall fragments: borrow the gradient vector of the first
    # fluid 6-neighbor (their fragments are small; measure their share)
    need = (~fl) & ((np.abs(W[0]) + np.abs(W[1]) + np.abs(W[2])) > 0)
    filled = np.zeros_like(need)
    for a in range(3):
        for d in (-1, +1):
            nb_fl = np.roll(fl, -d, a)
            m = need & ~filled & nb_fl
            for q in range(3):
                g[q][m] = np.roll(g[q], -d, a)[m]
                gc[q][m] = np.roll(gc[q], -d, a)[m]
            filled |= m
    orphan_share = float(np.sqrt(sum(W[a][need & ~filled]**2 for a in range(3))).sum()/max(area, 1e-30))

    Fest = MU*sum((W[a]*g[a]).sum() for a in range(3))
    Fcen = MU*sum((W[a]*gc[a]).sum() for a in range(3))

    # ---- Variant B: anchor each 1-D profile at the FRAGMENT CENTROID p* = x - sdf*grad(sdf)
    # (the SDF closest-point projection). The axis line through p* meets the wall AT p*, so the
    # wall-anchored one-sided quadratic needs only two fluid samples per axis at p* +/- sigma*h,
    # +/- 2 sigma*h (sigma = fluid side = sign(n_a); both samples are provably on the fluid side).
    # No solid-center borrowing needed: every fragment cell projects its own anchor. In-solver the
    # two samples come from trilinear interpolation of cell values (deferred-correction candidate);
    # here analytic u isolates the PLACEMENT question.
    m = (np.abs(W[0]) + np.abs(W[1]) + np.abs(W[2])) > 0
    xm, ym, zm = X[m], Y[m], Z[m]
    rx, ry, rz = xm - C0[0], ym - C0[1], zm - C0[2]
    rr = np.sqrt(rx*rx + ry*ry + rz*rz)
    nxh, nyh, nzh = rx/rr, ry/rr, rz/rr
    px, py, pz = C0[0] + R*nxh, C0[1] + R*nyh, C0[2] + R*nzh   # p* on the surface
    FB = 0.0
    nh = [nxh, nyh, nzh]
    for a in range(3):
        sg = np.where(nh[a] >= 0, 1.0, -1.0)
        d1 = [px.copy(), py.copy(), pz.copy()]; d1[a] = d1[a] + sg*h
        d2 = [px.copy(), py.copy(), pz.copy()]; d2[a] = d2[a] + 2*sg*h
        u1 = ux(d1[0], d1[1], d1[2])
        u2 = ux(d2[0], d2[1], d2[2])
        gB = sg*(2.0*u1 - 0.5*u2)/h            # du/da at the wall (one-sided quadratic, u(0)=0)
        FB += (W[a][m]*gB).sum()
    FB *= MU
    return dict(Fest=Fest, Fcen=Fcen, FB=FB, sumW=sumW, area=area, orphan=orphan_share)

# ------------------------------------------------------------------------------------------------
# Variant C: Basilisk `dirichlet_gradient` — TRUE-NORMAL image-point wall gradient.
# Instead of decomposing into three axis-anchored 1-D quadratics, reconstruct the single scalar
# wall-normal derivative d(ux)/dn along n̂ = ∇sdf directly, exactly as Basilisk embed.h does:
#   * boundary point p = SDF foot point x - sdf·n̂ (proxy for the fragment centroid), in cell units;
#   * along the DOMINANT axis (largest |n̂_a|) take two image points i=1,2 cells into the fluid;
#   * get u there by TRANSVERSE bi-quadratic interpolation of the cell-centred values (Basilisk's
#     `quadratic` Lagrange stencil), gated `defined` on the 3x3 transverse cells being fluid;
#   * fit the quadratic {u=0 at wall, v0 at d0, v1 at d1} for a 2nd-order d(ux)/dn;
#   * FALLBACKS (Basilisk): if only the near image point is defined -> 1-point linear (2nd order);
#     if neither is defined (sliver) -> the degenerate 1-point through the cell centre itself.
# The wall flux is then F = -mu * Σ_frag A_frag · d(ux)/dn  (A_frag = |W|, the fragment area;
# n̂ = fluid-outward so d(ux)/dn > 0 and F < 0, matching the reference). This is the ingredient the
# C++ embed operator will use; the test confirms it is O(h²) AND that the sliver fallback keeps it
# well-behaved (fallback-cell share reported).

def _quad(x, a1, a2, a3):
    # Lagrange quadratic through (-1,a1),(0,a2),(+1,a3), evaluated at x (Basilisk `quadratic`).
    return (a1 * (x - 1.0) + a3 * (x + 1.0)) * x / 2.0 - a2 * (x - 1.0) * (x + 1.0)

def run_normal(N, analytic=False):
    h = 1.0 / N
    c = (np.arange(N) + 0.5) * h - 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    S = sdf(X, Y, Z)
    fl = S >= 0
    U = np.where(fl, ux(X, Y, Z), 0.0)

    # fragment area per cell from the face openness (same W as variants A/B)
    W = []
    for a in range(3):
        O = face_openness(N, a, h)
        Wa = h * h * (O[:-1] - O[1:])
        W.append(np.moveaxis(Wa, [0, 1, 2], [a, (a + 1) % 3, (a + 2) % 3]))
    A = np.sqrt(W[0]**2 + W[1]**2 + W[2]**2)               # fragment area per cell
    m = A > 0                                              # fragment cells (fluid- AND solid-centred)

    # unit inward-to-fluid normal n̂ = ∇sdf (analytic here; ∇sdf in the solver) at fragment cells
    Im, Jm, Km = np.nonzero(m)
    xm, ym, zm = X[m], Y[m], Z[m]
    rx, ry, rz = xm - C0[0], ym - C0[1], zm - C0[2]
    rr = np.sqrt(rx * rx + ry * ry + rz * rz)
    nhat = np.stack([rx / rr, ry / rr, rz / rr], axis=1)   # (M,3), fluid-outward = ∇sdf
    sdm = S[m]
    pcell = -sdm[:, None] / h * nhat                       # foot point, cell units, per axis (M,3)
    cidx = np.stack([Im, Jm, Km], axis=1)                  # (M,3) integer cell coords
    Ucell = U[m]
    Aw = A[m]

    Mn = len(Im)
    grad = np.full(Mn, np.nan)                             # d(ux)/dn (into fluid)
    used = np.zeros(Mn, dtype=int)                         # 0=degenerate coef, 1=near-only, 2=two-pt
    bl = np.zeros(Mn, dtype=bool)                          # any image point used biased-linear interp
    da = np.argmax(np.abs(nhat), axis=1)                   # dominant axis per cell

    def gather(cc, a, oa, t1, ot1, t2, ot2):
        idx = [cc[:, 0].copy(), cc[:, 1].copy(), cc[:, 2].copy()]
        idx[a] = np.clip(idx[a] + oa, 0, N - 1)
        idx[t1] = np.clip(idx[t1] + ot1, 0, N - 1)
        idx[t2] = np.clip(idx[t2] + ot2, 0, N - 1)
        return U[idx[0], idx[1], idx[2]], (S[idx[0], idx[1], idx[2]] >= 0)

    for a in range(3):
        g = np.nonzero(da == a)[0]
        if len(g) == 0:
            continue
        t1, t2 = (a + 1) % 3, (a + 2) % 3
        cc = cidx[g]
        na, nt1, nt2 = nhat[g, a], nhat[g, t1], nhat[g, t2]
        pa, pt1, pt2 = pcell[g, a], pcell[g, t1], pcell[g, t2]
        sgn = np.where(na >= 0, 1.0, -1.0)
        v = [None, None]
        dd = [None, None]
        defd = [None, None]
        for l in (0, 1):
            i = (l + 1) * sgn.astype(int)                  # ±1, ±2 into the fluid
            dl = (i - pa) / na                             # distance (cells) to image plane
            y1 = pt1 + dl * nt1
            z1 = pt2 + dl * nt2
            j = np.clip(np.round(y1), -1, 1).astype(int)
            k = np.clip(np.round(z1), -1, 1).astype(int)
            ly, lz = y1 - j, z1 - k
            if analytic:  # sample the exact field at the image point (isolate geometry+fit)
                imx = [None, None, None]
                imx[a] = X[cc[:, 0], cc[:, 1], cc[:, 2]] + i * h if a == 0 else None
                # build physical image-point coords from cell center + (i, y1, z1)*h
                base = [X[cc[:, 0], cc[:, 1], cc[:, 2]], Y[cc[:, 0], cc[:, 1], cc[:, 2]],
                        Z[cc[:, 0], cc[:, 1], cc[:, 2]]]
                base[a] = base[a] + i * h
                base[t1] = base[t1] + y1 * h
                base[t2] = base[t2] + z1 * h
                v[l] = ux(base[0], base[1], base[2])
                dd[l] = dl
                defd[l] = np.ones(len(g), dtype=bool)
            else:
                # gather the 3x3 transverse stencil (values + fluid flags) at column a=i
                vv = {}
                sf = {}
                for dk in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        vv[(dj, dk)], sf[(dj, dk)] = gather(cc, a, i, t1, j + dj, t2, k + dk)
                full = np.ones(len(g), dtype=bool)
                for key in sf:
                    full &= sf[key]
                # (1) full 3x3 fluid -> bi-quadratic (Basilisk `quadratic` x `quadratic`)
                vbq = _quad(lz, _quad(ly, vv[(-1, -1)], vv[(0, -1)], vv[(1, -1)]),
                            _quad(ly, vv[(-1, 0)], vv[(0, 0)], vv[(1, 0)]),
                            _quad(ly, vv[(-1, 1)], vv[(0, 1)], vv[(1, 1)]))
                # (2) home cell fluid but stencil straddles -> Basilisk `embed_interpolate`
                #     biased-linear: anchored at the fluid home cell, each transverse gradient
                #     biased toward whichever side (+ or -) is fluid.
                home = sf[(0, 0)]

                def biased(coord, keyP, keyM):  # bias toward +side if fluid, else -side
                    return np.where(sf[keyP], np.abs(coord) * (vv[keyP] - vv[(0, 0)]),
                                    np.where(sf[keyM], np.abs(coord) * (vv[(0, 0)] - vv[keyM]), 0.0))
                # transverse t1 forward = sign(ly); t2 forward = sign(lz)
                spj = ly >= 0
                spk = lz >= 0
                cj = np.where(spj, biased(ly, (1, 0), (-1, 0)), biased(ly, (-1, 0), (1, 0)))
                ck = np.where(spk, biased(lz, (0, 1), (0, -1)), biased(lz, (0, -1), (0, 1)))
                vbl = vv[(0, 0)] + cj + ck
                v[l] = np.where(full, vbq, vbl)
                dd[l] = dl
                defd[l] = full | home
                bl[g] |= (home & ~full)                    # this image point fell to biased-linear
        d0, d1 = dd[0], dd[1]
        v0, v1 = v[0], v[1]
        two = defd[0] & defd[1]
        near = defd[0] & ~defd[1]
        deg = ~defd[0]
        gg = np.zeros(len(g))
        gg[two] = (v0[two] * d1[two] / d0[two] - v1[two] * d0[two] / d1[two]) / \
                  ((d1[two] - d0[two]) * h)
        gg[near] = v0[near] / (d0[near] * h)
        d0deg = np.maximum(1e-3, np.abs(pa[deg] / na[deg]))
        gg[deg] = Ucell[g][deg] / (d0deg * h)
        grad[g] = gg
        used[g] = np.where(two, 2, np.where(near, 1, 0))

    Fnorm = -MU * float(np.sum(Aw * grad))
    frac_bl = float(np.mean(bl))                           # share using biased-linear (straddling)
    frac_deg = float(np.mean(used == 0))                   # truly degenerate (coef path, u_cell)
    return dict(Fnorm=Fnorm, frac_bl=frac_bl, frac_deg=frac_deg,
                area=float(Aw.sum()), Mn=Mn)

Fex = reference()
print(f"F_exact = {Fex:+.6e}   (mu * closed-surface integral of grad(ux).n, n fluid-outward)")
print(f"{'N':>5} | {'A: axis-intercept':>17} {'ord':>5} | {'B: centroid-anchored':>20} {'ord':>5} | "
      f"{'central naive':>13} | {'sum|W|/4piR^2':>13}")
pA = pB = None
for N in (32, 64, 128, 192):
    r = run(N)
    eA = 100*(r["Fest"]-Fex)/abs(Fex); eB = 100*(r["FB"]-Fex)/abs(Fex)
    eC = 100*(r["Fcen"]-Fex)/abs(Fex)
    oA = np.log2(abs(pA)/abs(eA)) if pA else float("nan")
    oB = np.log2(abs(pB)/abs(eB)) if pB else float("nan")
    fac = np.log2(N/pN) if pA else 1.0
    print(f"{N:>5} | {eA:>+16.3f}% {oA/fac if pA else float('nan'):>5.2f} | "
          f"{eB:>+19.4f}% {oB/fac if pB else float('nan'):>5.2f} | {eC:>+12.2f}% | "
          f"{r['area']/(4*np.pi*R*R):>13.6f}")
    pA, pB, pN = eA, eB, N

# --- Variant C: true-normal Basilisk dirichlet_gradient (the C++ embed operator's ingredient) ---
print()
print("Variant C: true-normal image-point wall gradient (Basilisk dirichlet_gradient)")
print(f"{'N':>5} | {'C: true-normal':>16} {'ord':>5} | {'biased-lin %':>12} | {'degen %':>8} | "
      f"{'Ncut':>7}")
pC = pN2 = None
for N in (32, 64, 128, 192):
    r = run_normal(N)
    eC = 100 * (r["Fnorm"] - Fex) / abs(Fex)
    oC = np.log2(abs(pC) / abs(eC)) / np.log2(N / pN2) if pC else float("nan")
    print(f"{N:>5} | {eC:>+15.4f}% {oC:>5.2f} | {100*r['frac_bl']:>11.2f}% | "
          f"{100*r['frac_deg']:>7.3f}% | {r['Mn']:>7}")
    pC, pN2 = eC, N
