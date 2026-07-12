"""A-priori validation of the directional ghost-cell IBM projection (plan Phase 0).

The proposed scheme (staggered, point-based, NO openness): the divergence of a fluid-centered
pressure cell uses plain face differences; a face whose staggered velocity point is solid gets its
velocity from the momentum IBM's 1-D wall-anchored quadratic along the face's own axis
(poly_D/poly_Nc/poly_N_nb/poly_Nbc of src/cut_cell_ibm.hpp, reused verbatim):

    poly_D(th) * u_ghost = 2*u_bc + poly_Nc(th)*u_near + poly_N_nb(th)*u_far
    th = sdf_near/(sdf_near - sdf_ghost)  in (0,1], clamped [1e-4, 1]
    (wall at distance th below the near point; ghost point at distance 1; unit grid spacing)

Substituting corrected velocities u = u* - grad(phi) makes the closure implicit in phi: the
Poisson row gains couplings to phi(+/-1), phi(+/-2) along the axis (13-point, nonsymmetric).
Per-row conditioning rescale rho = min(1, min_f D_f) — the D_rescale analog.

Face-state cascade per (cell, axis, side)  [sdf >= 0 fluid]:
  COUPLED   face point fluid AND neighbor center fluid          -> standard +/- (phi_i - phi_nb)
  (sandwich) both face points of the axis solid, center fluid   -> BC_ONLY both sides
  GHOST_QUAD face point solid, near+far sources exist            -> quadratic closure, th in (0,1]
  GHOST_LIN  face point solid, only near source                  -> th*u_g = u_bc + (th-1)*u_near
  SLIVER     face point fluid but neighbor center solid: same quadratic with EXTENDED
             th = 1 + sdf_g/(sdf_g - sdf_beyond) in (1,2)  (evaluation INSIDE the data hull,
             D = th(1+th) > 2: well conditioned); falls back like QUAD->LIN
  BC_ONLY    no usable fluid source (or sandwich)                -> u_face = u_bc, no phi coupling
  EXPLICIT   sliver with no crossing on the u-line               -> face flux = u* (no phi term)

Tests (gates in main):
  1. extrapolation accuracy at ghost faces vs the analytic Stokes-sphere field, anchored at the
     linearized-SDF crossing (the scheme; expect O(h^2)) AND at the exact sphere crossing
     (pure polynomial truncation; expect O(h^3))
  2. closed divergence of the exact solenoidal field on near-IB cells (physical units):
     localized boundary truncation, expect O(h) — the SAME structure as the momentum IBM,
     global 2nd order then comes from elliptic damping and is measured by test 3
  3. assembled sparse solve: u* = u_periodic_exact + discrete_grad(phi_man); solve
     A phi = -div*(u*) with inhomogeneous u_bc = exact wall values; expect phi and the corrected
     velocity 2nd order; verify diagnostic div(u_corr) == residual identically (round-off)
  4. solver probes (dense, small N): A@1 = 0 on active rows; left-null compatibility gap;
     spectrum of the binary-openness-MG-surrogate-preconditioned operator (BiCGStab health);
     deferred-correction rate max|1-lambda|
  5. degenerate geometries: slab channel, sandwich slit, one-cell gap, wall through a face point

This file is the reference implementation for src/ghost_projection.hpp: the C++ gpFillEntry must
reproduce these closure coefficients to float tolerance.
"""
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------- analytic fields
R = 0.3102
C0 = np.array([0.013, -0.007, 0.004])   # off-lattice center (no symmetry luck)
THETA_MIN = 1e-4

def sdf_sphere(x, y, z):
    return np.sqrt((x - C0[0])**2 + (y - C0[1])**2 + (z - C0[2])**2) - R

def stokes_u(x, y, z):
    """Exact Stokes flow past the sphere, U_inf = (1,0,0). Solenoidal, no-slip at r=R."""
    dx, dy, dz = x - C0[0], y - C0[1], z - C0[2]
    r2 = dx*dx + dy*dy + dz*dz
    r = np.sqrt(r2)
    A = 3.0*R/(4.0*r)
    B = R**3/(4.0*r**3)
    ux = 1.0 - A*(1.0 + dx*dx/r2) - B*(1.0 - 3.0*dx*dx/r2)
    uy = -(A - 3.0*B)*dx*dy/r2
    uz = -(A - 3.0*B)*dx*dz/r2
    return ux, uy, uz

def periodic_u(x, y, z):
    """Manufactured periodic solenoidal field (period-1 box), nonzero at the sphere wall.
    Each component varies along its own axis (non-degenerate discrete divergence)."""
    tp = 2.0*np.pi
    return (np.sin(tp*x)*np.cos(tp*y),
            -np.cos(tp*x)*np.sin(tp*y) + np.sin(tp*y)*np.cos(tp*z),
            -np.cos(tp*y)*np.sin(tp*z))

def phi_man(x, y, z):
    tp = 2.0*np.pi
    return (np.sin(tp*x)*np.cos(tp*y) + np.sin(tp*y)*np.cos(tp*z)
            + np.sin(tp*z)*np.cos(tp*x))

# ---------------------------------------------------------------- closure polynomials (verbatim)
def poly_D(t):
    return t*(1.0 + t)

def poly_Nc(t):
    return 2.0*(t*t - 1.0)

def poly_Nnb(t):
    return t*(1.0 - t)

# state codes
COUPLED, QUAD, LIN, BC_ONLY, EXPLICIT = 0, 1, 2, 3, 4
STATE_NAMES = {COUPLED: "COUPLED", QUAD: "QUAD", LIN: "LIN",
               BC_ONLY: "BC_ONLY", EXPLICIT: "EXPLICIT"}

# ---------------------------------------------------------------- geometry + classification
def build_geo(N, sdf=sdf_sphere):
    """Classify every (cell, axis, side) face. Returns dict with sdf samples, states, thetas."""
    h = 1.0/N
    c = (np.arange(N) + 0.5)*h - 0.5
    Xc = np.meshgrid(c, c, c, indexing="ij")
    Sc = sdf(*Xc)
    active = Sc >= 0.0

    # face-point sdf per axis: Sf[a][i,j,k] = minus-face of cell (i,j,k) along axis a
    Sf = []
    Pf = []   # face-point coordinates (3 arrays each)
    for a in range(3):
        P = [Xc[0].copy(), Xc[1].copy(), Xc[2].copy()]
        P[a] = P[a] - 0.5*h
        Sf.append(sdf(*P))
        Pf.append(P)

    def cent(q, a):           # Sc[i+q] along axis a
        return np.roll(Sc, -q, axis=a)

    def face(m, a):           # Sf[a][i+m] along axis a
        return np.roll(Sf[a], -m, axis=a)

    states = {}
    for a in range(3):
        for side in (-1, +1):
            # roll offsets relative to cell index i (see derivation in module docstring)
            if side < 0:
                mg, mn, mf, mb = 0, 1, 2, -1      # ghost, near, far, beyond-ghost faces
                qnb = -1                          # neighbor center
                qn1, qn2 = 1, 2                   # phi cells needed by near/far face gradients
            else:
                mg, mn, mf, mb = 1, 0, -1, 2
                qnb = +1
                qn1, qn2 = -1, -2
            Sg, Sn, Sfar, Sb = face(mg, a), face(mn, a), face(mf, a), face(mb, a)
            Snb = cent(qnb, a)
            C1, C2 = cent(qn1, a), cent(qn2, a)

            st = np.full(Sc.shape, COUPLED, dtype=np.int8)
            th = np.ones(Sc.shape)

            coupled = (Sg >= 0) & (Snb >= 0)
            sandwich = (face(0, a) < 0) & (face(1, a) < 0)          # both faces of THIS cell solid
            ghost = (Sg < 0) & ~sandwich                            # near face fluid guaranteed
            sliver = (Sg >= 0) & (Snb < 0)

            # ghost theta (standard, wall between ghost and near face points)
            th_g = np.where(ghost, Sn/np.where(ghost, Sn - Sg, 1.0), 1.0)
            # sliver theta (extended, wall between ghost and beyond face points), needs Sb < 0
            has_x = sliver & (Sb < 0)
            th_s = np.where(has_x, 1.0 + Sg/np.where(has_x, Sg - Sb, 1.0), 1.0)

            src1 = (Sn >= 0) & (C1 >= 0)                            # near source usable
            src2 = (Sfar >= 0) & (C2 >= 0)                          # far source usable

            st[sandwich] = BC_ONLY
            st[ghost & ~src1] = BC_ONLY
            st[ghost & src1 & src2] = QUAD
            st[ghost & src1 & ~src2] = LIN
            st[sliver & ~has_x] = EXPLICIT
            st[sliver & has_x & ~src1] = BC_ONLY
            st[sliver & has_x & src1 & src2] = QUAD
            st[sliver & has_x & src1 & ~src2] = LIN
            th = np.where(ghost, np.clip(th_g, THETA_MIN, 1.0), th)
            th = np.where(sliver & has_x, np.clip(th_s, 1.0 + THETA_MIN, 2.0), th)
            st[coupled] = COUPLED
            states[(a, side)] = (st, th)

    return dict(N=N, h=h, Xc=Xc, Sc=Sc, Sf=Sf, Pf=Pf, active=active, states=states, sdf=sdf)

def overlay_cells(geo):
    """Active cells with at least one non-COUPLED face."""
    m = np.zeros_like(geo["active"])
    for (a, side), (st, _) in geo["states"].items():
        m |= (st != COUPLED)
    return m & geo["active"]

# crossing point (u_bc anchor) for a set of flat cell indices, given axis/side/state
def crossing_points(geo, a, side, cells_flat):
    N, h = geo["N"], geo["h"]
    st, th = geo["states"][(a, side)]
    stf, thf = st.ravel()[cells_flat], th.ravel()[cells_flat]
    mn = 1 if side < 0 else 0
    P = [np.roll(p, -mn, axis=a).ravel()[cells_flat] for p in geo["Pf"][a]]  # near-face point
    Pc = [p.copy() for p in P]
    Pc[a] = Pc[a] + side*thf*h                       # near - th*h (minus) / near + th*h (plus)
    # BC_ONLY (incl. sandwich): crossing between own face point and cell center
    bo = stf == BC_ONLY
    if np.any(bo):
        mg = 0 if side < 0 else 1
        Sg = np.roll(geo["Sf"][a], -mg, axis=a).ravel()[cells_flat]
        Scf = geo["Sc"].ravel()[cells_flat]
        t = np.clip(Scf/np.where(np.abs(Scf - Sg) > 0, Scf - Sg, 1.0), 0.0, 1.0)
        for q in range(3):
            Pc[q][bo] = geo["Xc"][q].ravel()[cells_flat][bo]
        Pc[a][bo] = Pc[a][bo] + side*t[bo]*(0.5*h)
    return Pc

def closure_weights(stf, thf):
    """(w_bc, w_n1, w_n2, D) for QUAD/LIN/BC_ONLY flat state/theta arrays."""
    D = np.ones_like(thf)
    wbc = np.zeros_like(thf)
    w1 = np.zeros_like(thf)
    w2 = np.zeros_like(thf)
    q = stf == QUAD
    Dq = poly_D(thf[q])
    D[q] = Dq
    wbc[q] = 2.0/Dq
    w1[q] = poly_Nc(thf[q])/Dq
    w2[q] = poly_Nnb(thf[q])/Dq
    l = stf == LIN
    D[l] = thf[l]
    wbc[l] = 1.0/thf[l]
    w1[l] = (thf[l] - 1.0)/thf[l]
    b = stf == BC_ONLY
    wbc[b] = 1.0
    return wbc, w1, w2, D

def row_rescale(geo):
    """rho = min(1, min over ghost faces of D_f) per cell (flat array)."""
    N = geo["N"]
    rho = np.ones(N**3)
    for (a, side), (st, th) in geo["states"].items():
        stf, thf = st.ravel(), th.ravel()
        _, _, _, D = closure_weights(stf, thf)
        m = (stf == QUAD) | (stf == LIN)
        rho[m] = np.minimum(rho[m], D[m])
    return rho

# ---------------------------------------------------------------- assembly + divergence
def gather_face(u, a, m, cells_flat):
    return np.roll(u, -m, axis=a).ravel()[cells_flat]

def divergence(geo, u3, ubc_fn, rho=None, u_explicit=None):
    """Closed point divergence (grid units: sum of face differences) on active cells,
    row-rescaled by rho. u3 = 3 face fields; ubc_fn(x,y,z)->(3 components); u_explicit
    supplies the field read at EXPLICIT faces (defaults to u3)."""
    N = geo["N"]
    if u_explicit is None:
        u_explicit = u3
    d = np.zeros(N**3)
    for a in range(3):
        for side in (-1, +1):
            st, th = geo["states"][(a, side)]
            stf = st.ravel()
            sgn = float(side) if side > 0 else -1.0
            # COUPLED faces: plain difference contribution
            mcp = stf == COUPLED
            mg = 0 if side < 0 else 1
            uf = np.roll(u3[a], -mg, axis=a).ravel()
            d[mcp] += sgn*uf[mcp]
            # closures
            for state in (QUAD, LIN, BC_ONLY):
                cells = np.nonzero(stf == state)[0]
                if len(cells) == 0:
                    continue
                thf = th.ravel()[cells]
                wbc, w1, w2, _ = closure_weights(np.full(len(cells), state, np.int8), thf)
                Pc = crossing_points(geo, a, side, cells)
                ub = ubc_fn(Pc[0], Pc[1], Pc[2])[a]
                val = wbc*ub
                if state != BC_ONLY:
                    mn = 1 if side < 0 else 0
                    mf = 2 if side < 0 else -1
                    val = val + w1*gather_face(u3[a], a, mn, cells)
                    if state == QUAD:
                        val = val + w2*gather_face(u3[a], a, mf, cells)
                d[cells] += sgn*val
            # explicit faces: read the supplied field at the own face point
            mex = np.nonzero(stf == EXPLICIT)[0]
            if len(mex):
                d[mex] += sgn*gather_face(u_explicit[a], a, mg, mex)
    if rho is not None:
        d *= rho
    d[~geo["active"].ravel()] = 0.0
    return d

def assemble(geo, rho):
    """Sparse A (N^3 x N^3): binary-openness base + closure deltas, overlay rows scaled by rho.
    Inactive rows = identity. Convention: A phi = -div(u*) (positive diagonal)."""
    N = geo["N"]
    n = N**3
    IDX = np.arange(n).reshape(N, N, N)
    rows, cols, vals = [], [], []
    activef = geo["active"].ravel()

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for a in range(3):
        for side in (-1, +1):
            st, th = geo["states"][(a, side)]
            stf, thf = st.ravel(), th.ravel()
            sgn = float(side) if side > 0 else -1.0
            # face at roll-offset m couples cells (i+m-1, i+m); div term sgn*c_f*u(face m)
            # A[r, cp] -= rho_r*sgn*c_f ; A[r, cm] += rho_r*sgn*c_f
            def add_face(cells, m, cf):
                r = cells
                cp = np.roll(IDX, -m, axis=a).ravel()[cells]
                cm = np.roll(IDX, -(m - 1), axis=a).ravel()[cells]
                w = rho[cells]*sgn*cf
                add(r, cp, -w)
                add(r, cm, +w)

            mcp = np.nonzero((stf == COUPLED) & activef)[0]
            add_face(mcp, 0 if side < 0 else 1, np.ones(len(mcp)))
            for state in (QUAD, LIN):
                cells = np.nonzero((stf == state) & activef)[0]
                if len(cells) == 0:
                    continue
                _, w1, w2, _ = closure_weights(np.full(len(cells), state, np.int8), thf[cells])
                mn = 1 if side < 0 else 0
                add_face(cells, mn, w1)
                if state == QUAD:
                    mf = 2 if side < 0 else -1
                    add_face(cells, mf, w2)
    # inactive rows: identity
    inact = np.nonzero(~activef)[0]
    add(inact, inact, np.ones(len(inact)))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows).astype(np.int64),
                                              np.concatenate(cols).astype(np.int64))),
                      shape=(n, n))
    A.sum_duplicates()
    return A

def binary_openness_op(geo):
    """The symmetric MG surrogate: 7-point op with o=1 on COUPLED faces, 0 otherwise."""
    N = geo["N"]
    n = N**3
    IDX = np.arange(n).reshape(N, N, N)
    rows, cols, vals = [], [], []
    activef = geo["active"].ravel()
    for a in range(3):
        for side in (-1, +1):
            st, _ = geo["states"][(a, side)]
            cells = np.nonzero((st.ravel() == COUPLED) & activef)[0]
            m = 0 if side < 0 else 1
            nb = np.roll(IDX, -(m - 1) if side < 0 else -m, axis=a).ravel()[cells]
            # neighbor cell across the face: minus side -> i-1 (m-1 roll of IDX at m=0), plus -> i+1
            rows += [cells, cells]
            cols += [cells, nb]
            vals += [np.ones(len(cells)), -np.ones(len(cells))]
    inact = np.nonzero(~activef)[0]
    rows.append(inact)
    cols.append(inact)
    vals.append(np.ones(len(inact)))
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows).astype(np.int64),
                                              np.concatenate(cols).astype(np.int64))),
                      shape=(n, n))
    M.sum_duplicates()
    return M

# ---------------------------------------------------------------- tests
def face_fields(geo, ufun):
    return [ufun(*geo["Pf"][a])[a] for a in range(3)]

def test_extrapolation(Ns):
    """QUAD ghost-face closures vs the smooth continuation of the analytic Stokes field.
    Variant 'scheme':     u_bc = 0 (what the solver knows)      -> O(h^2) (wall-anchoring error)
    Variant 'consistent': u_bc = field at the linearized anchor -> O(h^3) (pure poly truncation)"""
    print("\n[1] ghost-face extrapolation vs analytic Stokes field (QUAD faces)")
    print(f"{'N':>5} {'scheme ubc=0':>13} {'ord':>6} {'consistent':>12} {'ord':>6} "
          f"{'nQUAD':>7} {'nSLIV':>6} {'nLIN':>5}")
    prev = {}
    slopes = {}
    for N in Ns:
        geo = build_geo(N)
        u3 = face_fields(geo, stokes_u)
        errs = {"scheme": [], "consistent": []}
        nq = nl = ns = 0
        for a in range(3):
            for side in (-1, +1):
                st, th = geo["states"][(a, side)]
                stf, thf = st.ravel(), th.ravel()
                nl += int(np.sum(stf == LIN))
                cells = np.nonzero(stf == QUAD)[0]
                if len(cells) == 0:
                    continue
                tt = thf[cells]
                nq += int(np.sum(tt <= 1.0))
                ns += int(np.sum(tt > 1.0))
                wbc, w1, w2, _ = closure_weights(np.full(len(cells), QUAD, np.int8), tt)
                mg = 0 if side < 0 else 1
                mn = 1 if side < 0 else 0
                mf = 2 if side < 0 else -1
                base = (w1*gather_face(u3[a], a, mn, cells)
                        + w2*gather_face(u3[a], a, mf, cells))
                # truth = smooth continuation of the fluid solution at the closed face point
                truth = gather_face(u3[a], a, mg, cells)
                Pc = crossing_points(geo, a, side, cells)
                ub_c = stokes_u(Pc[0], Pc[1], Pc[2])[a]
                errs["scheme"].append(np.abs(base - truth))            # u_bc = 0
                errs["consistent"].append(np.abs(base + wbc*ub_c - truth))
        e_s = max(float(e.max()) for e in errs["scheme"])
        e_c = max(float(e.max()) for e in errs["consistent"])
        o_s = np.log2(prev["s"]/e_s)/np.log2(N/prev["N"]) if prev else float("nan")
        o_c = np.log2(prev["c"]/e_c)/np.log2(N/prev["N"]) if prev else float("nan")
        print(f"{N:>5} {e_s:>13.3e} {o_s:>6.2f} {e_c:>12.3e} {o_c:>6.2f} "
              f"{nq:>7} {ns:>6} {nl:>5}")
        prev = {"s": e_s, "c": e_c, "N": N}
        slopes = {"scheme": o_s, "consistent": o_c}
    return slopes

def test_divergence(Ns):
    print("\n[2] closed divergence of the exact solenoidal Stokes field, physical units (/h)")
    print("    (near-IB rows: localized boundary truncation, expect ~O(h); bulk: O(h^2))")
    print(f"{'N':>5} {'max near-IB':>13} {'ord':>6} {'max bulk':>12} {'ord':>6}")
    prev = None
    slope = float("nan")
    for N in Ns:
        geo = build_geo(N)
        u3 = face_fields(geo, stokes_u)
        d = divergence(geo, u3, stokes_u)/geo["h"]     # unscaled rows (rho=None): raw truncation
        ov = overlay_cells(geo).ravel()
        # the Stokes field is NOT periodic: exclude the box-boundary wrap layer from the metrics
        interior = ((np.abs(geo["Xc"][0]) < 0.5 - 2*geo["h"])
                    & (np.abs(geo["Xc"][1]) < 0.5 - 2*geo["h"])
                    & (np.abs(geo["Xc"][2]) < 0.5 - 2*geo["h"])).ravel()
        ov &= interior
        bulk = geo["active"].ravel() & ~ov & interior
        e_ib = float(np.abs(d[ov]).max())
        e_bk = float(np.abs(d[bulk]).max())
        o_ib = np.log2(prev[0]/e_ib)/np.log2(N/prev[2]) if prev else float("nan")
        o_bk = np.log2(prev[1]/e_bk)/np.log2(N/prev[2]) if prev else float("nan")
        print(f"{N:>5} {e_ib:>13.3e} {o_ib:>6.2f} {e_bk:>12.3e} {o_bk:>6.2f}")
        prev = (e_ib, e_bk, N)
        slope = o_ib
    return slope

def solve_system(A, b, n_active):
    try:
        return spla.spsolve(A.tocsc(), b)
    except MemoryError:
        ilu = spla.spilu(A.tocsc(), drop_tol=1e-5, fill_factor=20)
        M = spla.LinearOperator(A.shape, ilu.solve)
        x, info = spla.lgmres(A, b, M=M, rtol=1e-12, maxiter=2000)
        if info != 0:
            raise RuntimeError(f"lgmres failed: info={info}")
        return x

def test_solve(Ns):
    print("\n[3] assembled projection solve, manufactured field (THE global-order gate)")
    print("    u* = u_exact + Dgrad(phi_man); expect phi and corrected u ~ O(h^2)")
    print(f"{'N':>5} {'max|phi err|':>13} {'ord':>6} {'max|u err|':>12} {'ord':>6} "
          f"{'max near-IB':>12} {'L2|u err|':>12} {'ord':>6} {'diag==res':>10} {'rho_min':>8}")
    prev = None
    out = {}
    for N in Ns:
        geo = build_geo(N)
        h = geo["h"]
        rho = row_rescale(geo)
        activef = geo["active"].ravel()
        pm = phi_man(*geo["Xc"])                      # cell-centered manufactured phi
        u3 = face_fields(geo, periodic_u)
        ustar = [u3[a] + (pm - np.roll(pm, +1, axis=a)) for a in range(3)]

        A = assemble(geo, rho)
        b = -divergence(geo, ustar, periodic_u, rho=rho)
        # sanity: active rows never touch inactive columns
        sub = A[activef, :][:, ~activef]
        assert sub.nnz == 0, "active row references decoupled phi"
        # Singular (Neumann-like) system: dump the incompatibility UNIFORMLY (residual = lambda*e,
        # the direct-solve equivalent of the iterative mean-removal the real solver uses). Row
        # pinning alone dumps it LOCALLY (a spurious O(h^0) spike at the pin — measured), and a
        # bordered [A e; e^T 0] system is sparse-LU-hostile (the dense multiplier column explodes
        # the fill-in — also measured, catastrophically). So: factor the pinned matrix once, solve
        # for b and for the uniform vector e, and pick lambda to make the pinned row consistent:
        # phi = phi_b - lambda*phi_e,  lambda = (A_r0.phi_b - b_r0)/(A_r0.phi_e - 1)
        # => A phi = b - lambda*e exactly (uniform dump), sparse-friendly.
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
        phi = phi_b - lam * phi_e
        pmf = pm.ravel()

        dphi = phi - pmf
        dphi -= dphi[activef].mean()
        e_phi = float(np.abs(dphi[activef]).max())     # grid units, vs the O(1) target pm

        ucorr = [ustar[a] - (phi.reshape(geo["Sc"].shape)
                             - np.roll(phi.reshape(geo["Sc"].shape), +1, axis=a))
                 for a in range(3)]
        ovm = overlay_cells(geo)
        e_u, e_u_ib, s2, n_f = 0.0, 0.0, 0.0, 0
        for a in range(3):
            st_m, _ = geo["states"][(a, -1)]
            # faces the scheme constrains: COUPLED faces owned by an ACTIVE cell.
            # (COUPLED faces of solid-centered cells get a one-sided phi-0 correction in
            # projectCorrect — watch item for the coupled solver, not an a-priori gate.)
            fluid_face = (st_m == COUPLED) & geo["active"]
            near = fluid_face & (ovm | np.roll(ovm, +1, axis=a))
            err = np.abs(ucorr[a] - u3[a])
            if np.any(fluid_face):
                e_u = max(e_u, float(err[fluid_face].max()))
                s2 += float((err[fluid_face]**2).sum())
                n_f += int(fluid_face.sum())
            if np.any(near):
                e_u_ib = max(e_u_ib, float(err[near].max()))
        e_u_l2 = np.sqrt(s2/max(n_f, 1))

        # diagnostic == residual identity (closure consistency by construction): the closed
        # divergence of the corrected field is EXACTLY A phi - b = -(b - A phi). (Max-abs is what
        # the solver diagnostic reports, so the sign is irrelevant there.)
        diag = divergence(geo, ucorr, periodic_u, rho=rho, u_explicit=ustar)
        res = b - A@phi                               # = lambda*e: the uniform incompat. dump
        ident = float(np.abs(diag[activef] + res[activef]).max())
        scale = max(1.0, float(np.abs(b[activef]).max()))

        o_p = np.log2(prev[0]/e_phi)/np.log2(N/prev[3]) if prev else float("nan")
        o_u = np.log2(prev[1]/e_u)/np.log2(N/prev[3]) if prev else float("nan")
        o_l2 = np.log2(prev[2]/e_u_l2)/np.log2(N/prev[3]) if prev else float("nan")
        ok = "OK" if ident < 1e-10*scale else f"{ident:.1e}"
        print(f"{N:>5} {e_phi:>13.3e} {o_p:>6.2f} {e_u:>12.3e} {o_u:>6.2f} "
              f"{e_u_ib:>12.3e} {e_u_l2:>12.3e} {o_l2:>6.2f} {ok:>10} "
              f"{rho[overlay_cells(geo).ravel()].min():>8.1e}")
        prev = (e_phi, e_u, e_u_l2, N)
        out = {"o_phi": o_p, "o_u": o_u, "o_u_l2": o_l2, "ident": ident < 1e-10*scale}
    return out

def test_probes(N):
    print(f"\n[4] solver probes at N={N} (dense)")
    geo = build_geo(N)
    rho = row_rescale(geo)
    A = assemble(geo, rho)
    M = binary_openness_op(geo)
    activef = geo["active"].ravel()
    ii = np.nonzero(activef)[0]
    Aa = A[np.ix_(ii, ii)].toarray()
    Ma = M[np.ix_(ii, ii)].toarray()
    na = len(ii)

    e1 = float(np.abs(Aa.sum(axis=1)).max())
    print(f"    A@1 on active rows:            {e1:.2e}   (constants right-null: want ~0)")

    # left null vector + compatibility gap against a physical RHS
    w, V = np.linalg.eig(Aa.T)
    k = int(np.argmin(np.abs(w)))
    wn = np.real(V[:, k])
    u3 = face_fields(geo, periodic_u)
    pmf = phi_man(*geo["Xc"])
    ustar = [u3[a] + (pmf - np.roll(pmf, +1, axis=a)) for a in range(3)]
    b = (-divergence(geo, ustar, periodic_u, rho=rho))[ii]
    gap = abs(wn @ b)/(np.linalg.norm(wn)*np.linalg.norm(b))
    ones_ang = abs(wn @ np.ones(na))/(np.linalg.norm(wn)*np.sqrt(na))
    print(f"    left-null eigenvalue |lam|:    {abs(w[k]):.2e}")
    print(f"    |w.1|/(|w||1|):                {ones_ang:.4f}   (1.0 would mean w = constants)")
    print(f"    compatibility gap |w.b|/|w||b|: {gap:.2e}   (small => mean-removal-style OK)")

    # spectrum of the surrogate-preconditioned operator (constant mode pinned via rank-1)
    cshift = np.mean(np.diag(Ma))
    e = np.ones((na, 1))/np.sqrt(na)
    G = np.linalg.solve(Ma + cshift*(e@e.T), Aa + cshift*(e@e.T))
    lam = np.linalg.eigvals(G)
    lam = lam[np.argsort(np.abs(lam - 1.0))]        # drop the pinned ~1 constant mode last
    re, im = lam.real, lam.imag
    print(f"    spec(M^-1 A): Re in [{re.min():.3f}, {re.max():.3f}], max|Im| = "
          f"{np.abs(im).max():.3f}, n = {na}")
    dc = float(np.abs(1.0 - lam).max())
    print(f"    deferred-correction rate max|1-lam| = {dc:.3f}  (<1 => DC converges)")
    return dict(gap=gap, re_min=float(re.min()), dc=dc)

def test_degenerate():
    print("\n[5] degenerate geometries (classification + null-space sanity)")
    N = 24
    cases = {
        "slab channel |y|<0.30":   lambda x, y, z: 0.30 - np.abs(y),
        "offset slab (th generic)": lambda x, y, z: 0.30 + 0.31/N - np.abs(y),
        "sandwich slit gap=0.8h":  lambda x, y, z: 0.4/N - np.abs(y - 0.021),
        "one-cell gap=1.6h":       lambda x, y, z: 0.8/N - np.abs(y - 0.021),
        "wall AT a face point":    lambda x, y, z: -y,   # y=0 face plane exactly on the wall
    }
    ok = True
    for name, f in cases.items():
        geo = build_geo(N, sdf=f)
        rho = row_rescale(geo)
        A = assemble(geo, rho)
        activef = geo["active"].ravel()
        counts = {s: 0 for s in STATE_NAMES}
        for (a, side), (st, _) in geo["states"].items():
            stf = st.ravel()[activef]
            for s in STATE_NAMES:
                counts[s] += int(np.sum(stf == s))
        e1 = float(np.abs(np.asarray(A[activef].sum(axis=1))).max())
        cross = A[activef, :][:, ~activef].nnz
        finite = np.all(np.isfinite(A.data))
        stat = "ok" if (e1 < 1e-10 and cross == 0 and finite) else "FAIL"
        ok &= stat == "ok"
        cs = " ".join(f"{STATE_NAMES[s][:4]}={c}" for s, c in counts.items() if c)
        print(f"    {name:<26} A@1={e1:.1e} cross={cross} {stat}   [{cs}]")
    return ok

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grids")
    ap.add_argument("--probe-n", type=int, default=12)
    args = ap.parse_args()

    Ns_eval = [16, 32, 64] if args.quick else [16, 32, 64, 128]
    Ns_solve = [16, 24, 32] if args.quick else [16, 24, 32, 48]

    s1 = test_extrapolation(Ns_eval)
    s2 = test_divergence(Ns_eval)
    s3 = test_solve(Ns_solve)
    s4 = test_probes(args.probe_n)
    s5 = test_degenerate()

    print("\n==== gates ====")
    gates = [
        ("extrapolation order (consistent)  >= 2.5", s1["consistent"] >= 2.5),
        ("extrapolation order (scheme)      >= 1.7", s1["scheme"] >= 1.7),
        ("near-IB divergence order          >= 0.8", s2 >= 0.8),
        ("solve: phi order                  >= 1.5", s3["o_phi"] >= 1.5),
        ("solve: corrected-velocity order   >= 1.7", s3["o_u"] >= 1.7),
        ("diagnostic == residual identity", s3["ident"]),
        ("preconditioned spectrum Re > 0", s4["re_min"] > 0.0),
        ("compatibility gap < 1e-3", s4["gap"] < 1e-3),
        ("degenerate geometries sane", s5),
    ]
    npass = 0
    for name, okk in gates:
        print(f"  {'PASS' if okk else 'FAIL'}  {name}")
        npass += okk
    print(f"{npass}/{len(gates)} gates passed")
