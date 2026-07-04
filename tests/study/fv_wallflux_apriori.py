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
