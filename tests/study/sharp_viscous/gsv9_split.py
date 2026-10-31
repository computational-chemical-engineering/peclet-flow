"""Architect round 3, Q1 (doc/vof_sharp_viscous_stress.md §6.1): is the 3-D G-SV9(b) blow-up a
departure of the production step from the 2-D harness, or a genuine 3-D mode?

harness : 2-D dense rho(G), G = P (M+K)^-1 (M+K-A), for two implicit splits of the cell normal
          stress 2 mu(C) m_cc:
            'K2' -- all of it implicit (what stab.py / gsv9a.py / §6 measured);
            'K1' -- mu(C) m_cc implicit, mu(C) m_cc explicit on u^n (what the production kernels do:
                    the Laplacian band ibmBuildDiffusionVar + the normal element N_c of
                    full_stress.hpp, §5.4(b)).
          A = sharp (§5.7 interfacial-mean tau) or averaged (today's arithmetic operator).
prod    : the PRODUCTION step map on a 3-D case, linear with the colour frozen, by Arnoldi
          (scipy eigs) on u -> step(u). Non-incremental projection and converged solves by default
          (RB-GS to 1e-15, the Chebyshev pressure driver to 1e-13 -- the CG drivers return wrong
          fields at density ratio 1e4), so that it is the harness's G. --incremental runs the
          production incremental-rotational step on the full state (u, v, w, p), p through
          get_field/set_field("p") with its mean removed (G-SV9(e)); --prodsolve keeps gsv9.py's
          PCG(800). Geometries: 'ext' = the 2-D harness plane extruded in z (exact harness
          fractions, z periodic, walls on x and y; --kz0 restricts to z-independent fields with
          u_z = 0, the harness subspace); 'plane' = the G-SV9(b) plane n(22.5,22.5), offset 0.3,
          walls on all faces, exact fractions (--sub8 = gsv9.py's 8^3 sub-sampled colour);
          'film' = the G-SV9(d) slab.

  nice -n 10 python gsv9_split.py harness [--N 16]
  nice -n 10 python gsv9_split.py prod --geom ext --N 16 --Nz 4 --mode sharp --pair 1e-3,1e-4 --D 1e4 [--kz0]
  nice -n 10 python gsv9_split.py prod --geom plane --N 16 --mode sharp --pair 1e-3,1e-4 --D 1e4 --incremental
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def harness_case(N, ang, off, muA, rA, Ds, scheme):
    import mac2d as M
    from stab import blocks, rho_face
    M.TAU_RULE = "mixed-avg"
    case = M.Case(N, ang, off, muA, 1.0, "couette")
    rf = rho_face(case, *blocks(case, "facef", "arith", None, None)[3:], rA, 1.0)
    if scheme == "sharp":
        M.CELL_VOL = True
        K2 = blocks(case, "facef", "arith", None, {"xx", "yy", "yu", "xv"})[0]
        Kc = blocks(case, "facef", "arith", None, {"xx", "yy"})[0]
        M.CELL_VOL = False
        At, G, D, _, _ = blocks(case, "sharp", "facef", "fit", None)
    else:
        K2 = blocks(case, "arith", "arith", None, {"xx", "yy", "yu", "xv"})[0]
        Kc = blocks(case, "arith", "arith", None, {"xx", "yy"})[0]
        At, G, D, _, _ = blocks(case, "arith", "arith", None, None)
    K1 = K2 - 0.5 * Kc
    # 'Kb': the full 2 mu(C) m_cc implicit on INTERFACIAL cells only (eps < C < 1 - eps), K1 elsewhere
    _, _, _, uidx, mm = blocks(case, "arith", "arith", None, {"xx"})
    N, h = case.N, case.h
    nv = len(uidx)
    Kcf = np.zeros((nv, nv))
    Kcb = np.zeros((nv, nv))
    for i in range(N):
        for j in range(N):
            f = case.frac_cell(i * h, j * h)
            w = 2.0 * (f * muA + (1.0 - f) * 1.0) / h**2
            for ip, im in ((mm.iu(i + 1, j), mm.iu(i, j)), (mm.iv(i, j + 1), mm.iv(i, j))):
                g = np.zeros(nv)
                if ip in uidx:
                    g[uidx[ip]] += 1.0
                if im in uidx:
                    g[uidx[im]] -= 1.0
                Kcf += w * np.outer(g, g)
                if 1e-8 < f < 1.0 - 1e-8:
                    Kcb += w * np.outer(g, g)
    chk = float(np.max(np.abs(Kcf - Kc)) / np.max(np.abs(Kc)))
    assert chk < 1e-10, chk
    Kb = K1 + 0.5 * Kcb
    out = {}
    for Dn in Ds:
        dt = Dn * case.h**2 * (1.0 if muA < 1 else rA / muA)
        Mm = np.diag(rf / dt)
        Mi = np.diag(dt / rf)
        P = np.eye(len(rf)) - Mi @ G @ np.linalg.pinv(D @ Mi @ G) @ D
        for nm, K in (("K2", K2), ("K1", K1), ("Kb", Kb)):
            out[(nm, Dn)] = float(np.max(np.abs(np.linalg.eigvals(P @ np.linalg.solve(Mm + K, Mm + K - At)))))
    return out


def _harness_job(a):
    N, mu, r, ang, off, sch, Ds = a
    return a, harness_case(N, ang, off, mu, r, Ds, sch)


def cmd_harness(a):
    from multiprocessing import Pool
    geoms = ((0.0, 0.6), (0.0, 0.2), (22.5, 0.0), (22.5, 0.3), (45.0, 0.3), (67.5, 0.5))
    pairs = ((1e-3, 1e-4), (1e-2, 1e-3), (1e2, 1e3), (1e3, 1e4))
    Ds = (0.1, 1.0, 10.0, 100.0, 1e4)
    jobs = [(a.N, m, r, g, o, sch, Ds) for sch in ("sharp", "averaged") for (m, r) in pairs for (g, o) in geoms]
    with Pool(a.procs) as p:
        res = p.map(_harness_job, jobs)
    for (N, m, r, g, o, sch, _), out in res:
        print("HARN N=%d %-8s mu=%g rho=%g geom=(%g,%g) | %s" % (N, sch, m, r, g, o, " | ".join(
            "%s: %s" % (nm, " ".join("%.4f" % out[(nm, d)] for d in Ds)) for nm in ("K2", "K1", "Kb"))), flush=True)
    for sch in ("sharp", "averaged"):
        for nm in ("K2", "K1", "Kb"):
            w = [max(out[(nm, d)] for (j, out) in res if j[5] == sch) for d in Ds]
            print("HARN-WORST N=%d %s %s rho(G) at D_hi=%s: %s" % (a.N, sch, nm, Ds, " ".join("%.4f" % x for x in w)))


def cube_frac(m, al):
    import itertools
    m = np.array(m, float)
    al = al - sum(x for x in m if x < 0)
    m = np.abs(m)
    nz = [x for x in m if x > 1e-300]
    d = len(nz)
    if d == 0:
        return 1.0 if al > 0 else 0.0
    tot = 0.0
    for S in itertools.product((0, 1), repeat=d):
        r = al - sum(x for x, s in zip(nz, S) if s)
        if r > 0:
            tot += (-1) ** sum(S) * r**d
    return min(1.0, max(0.0, tot / (math.factorial(d) * float(np.prod(nz)))))


def colour(a, N, Nz):
    if a.geom == "ext":
        import mac2d as M
        case = M.Case(N, a.ang, a.off, 1.0, 1.0, "couette")
        h = case.h
        C2 = np.array([[case.frac_cell(i * h, j * h) for j in range(N)] for i in range(N)])
        return np.asfortranarray(np.repeat(C2[:, :, None], Nz, axis=2))
    if a.geom == "film":
        lowA = a.muA < 1.0
        C = np.zeros((N, N, N)) if lowA else np.ones((N, N, N))
        C[N // 2, :, :] = 1.0 if lowA else 0.0
        return np.asfortranarray(C)
    b = g = math.radians(22.5)
    n = np.array([math.cos(b) * math.cos(g), math.cos(b) * math.sin(g), math.sin(b)])
    x0 = np.array([0.5 * N + 0.3 * c for c in n])  # index units, as gsv9.py
    if a.sub8:
        sys.path.insert(0, HERE)
        import gsv9
        gsv9.N = N
        return gsv9.plane_fractions(N, n, x0)
    C = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j, k] = cube_frac(n, float(n @ (x0 - np.array([i, j, k]))))
    return np.asfortranarray(C)


def cmd_prod(a):
    import peclet.flow as pf
    from scipy.sparse.linalg import LinearOperator, eigs
    N = a.N
    Nz = a.Nz if a.geom == "ext" else N
    h = 1.0 / N
    muA, rA = a.muA, a.rA
    dt = a.D * h * h * (1.0 if muA < 1 else rA / muA)
    s = pf.Solver((N, N, Nz), extent=(1.0, 1.0, Nz * h))
    s.set_dt(dt)
    for f in ("-x", "+x", "-y", "+y"):
        s.set_domain_bc(f, "wall")
    for f in ("-z", "+z"):
        s.set_domain_bc(f, "periodic" if a.geom == "ext" else "wall")
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_pressure_geometry(np.full((N, N, Nz), 1.0, order="F"))
    s.enable_vof()
    C0 = colour(a, N, Nz)
    s.set_vof(C0)
    s.set_property_model("mu", "linear", "C", [1.0, muA - 1.0])
    s.set_property_model("rho", "linear", "C", [1.0, rA - 1.0])
    s.set_advection(False)
    dg = s.diagnostics
    dg.set_vof_viscous_interface(a.mode)
    dg.set_incremental_pressure(bool(a.incremental))
    if a.prodsolve:
        s.set_pressure_pcg(True, 800, 1e-12)
    else:
        {"pcg": s.set_pressure_pcg, "fcg": s.set_pressure_fcg, "cheb": s.set_pressure_chebyshev}[a.pdriver](
            True, 20000, 1e-13)
        dg.set_velocity_solver("gauss_seidel")
        dg.set_velocity_solver_params(200000, 1e-15, 2)
    zper = a.geom == "ext"
    shp = (N, N, Nz)
    lo = [1, 1, 0 if zper else 1]  # face index 0 is the wall face (zero) on a walled axis
    sls = [tuple(slice(lo[c], None) if d == c else slice(None) for d in range(3)) for c in range(3)]
    sizes = [int(np.prod([n - (lo[c] if d == c else 0) for d, n in enumerate(shp)])) for c in range(3)]
    npc = N * N * Nz if a.incremental else 0  # the incremental state carries p (internal units, mean removed)
    nd = sum(sizes) + npc
    amp = 1e-6 * h / dt  # colour CFL ~1e-6 for a unit-max vector; the map is linear
    stats = {"n": 0, "pit": 0, "leak": 0.0}

    def unpack(x):
        arrs, o = [], 0
        for c in range(3):
            A = np.zeros(shp)
            A[sls[c]] = x[o:o + sizes[c]].reshape(A[sls[c]].shape, order="F")
            o += sizes[c]
            arrs.append(A)
        return arrs

    def pack(arrs, P=None):
        out = [np.asarray(arrs[c])[sls[c]].ravel(order="F") for c in range(3)]
        if npc:
            P = np.asarray(P).ravel(order="F")
            out.append(P - P.mean())
        return np.concatenate(out)

    def proj(x):
        u = unpack(x)
        u = [np.repeat(u[0].mean(axis=2, keepdims=True), Nz, 2), np.repeat(u[1].mean(axis=2, keepdims=True), Nz, 2),
             np.zeros(shp)]
        return pack(u)

    def mv(x):
        x = np.real(np.asarray(x)).ravel()
        sc = amp / max(np.max(np.abs(x)), 1e-300)
        for c, A in enumerate(unpack(x * sc)):
            s.set_velocity(c, np.asfortranarray(A))
        if npc:
            P = x[nd - npc:] * sc
            s.set_field("p", np.asfortranarray((P - P.mean()).reshape(shp, order="F")))
        s.step()
        s.set_vof(C0)
        stats["n"] += 1
        stats["pit"] = max(stats["pit"], dg.last_pressure_iterations())
        y = pack([s.get_u(), s.get_v(), s.get_w()], s.get_field("p") if npc else None) / sc
        if a.kz0:
            yp = proj(y)
            stats["leak"] = max(stats["leak"], float(np.linalg.norm(y - yp) / max(np.linalg.norm(y), 1e-300)))
            y = yp
        return y

    rng = np.random.default_rng(7)
    v0 = rng.standard_normal(nd)
    if a.kz0:
        v0 = proj(v0)
    x1, x2 = rng.standard_normal(nd), rng.standard_normal(nd)
    if a.kz0:
        x1, x2 = proj(x1), proj(x2)
    lin = np.linalg.norm(mv(0.3 * x1 + 0.7 * x2) - 0.3 * mv(x1) - 0.7 * mv(x2)) / np.linalg.norm(mv(x1))
    op = LinearOperator((nd, nd), matvec=mv, dtype=float)
    vals = eigs(op, k=a.k, which="LM", v0=v0, ncv=a.ncv, tol=1e-9, maxiter=200, return_eigenvectors=False)
    vals = sorted(vals, key=lambda z: -abs(z))
    tag = "%s geom=%s%s N=%d Nz=%d mu=%g rho=%g D_hi=%g incr=%d prodsolve=%d kz0=%d p=%s" % (
        a.mode, a.geom, "(%g,%g)" % (a.ang, a.off) if a.geom == "ext" else ("/sub8" if a.sub8 else ""), N, Nz,
        muA, rA, a.D, a.incremental, a.prodsolve, a.kz0, a.pdriver)
    print("PROD %s | rho=%.6f | top: %s | matvecs=%d max_pcg_it=%d lin=%.1e kz0_leak=%.1e solver=%s" % (
        tag, abs(vals[0]), " ".join("%.5f%+.5fi" % (z.real, z.imag) for z in vals[:4]), stats["n"], stats["pit"],
        lin, stats["leak"], dg.velocity_solver()), flush=True)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    h = sp.add_parser("harness")
    h.add_argument("--N", type=int, default=16)
    h.add_argument("--procs", type=int, default=12)
    q = sp.add_parser("prod")
    q.add_argument("--geom", choices=("ext", "plane", "film"), default="ext")
    q.add_argument("--N", type=int, default=16)
    q.add_argument("--Nz", type=int, default=4)
    q.add_argument("--ang", type=float, default=22.5)
    q.add_argument("--off", type=float, default=0.3)
    q.add_argument("--mode", choices=("sharp", "averaged"), default="sharp")
    q.add_argument("--pair", default="1e-3,1e-4")
    q.add_argument("--D", type=float, default=1e4)
    q.add_argument("--kz0", action="store_true")
    q.add_argument("--incremental", action="store_true")
    q.add_argument("--prodsolve", action="store_true")
    q.add_argument("--sub8", action="store_true")
    q.add_argument("--pdriver", choices=("pcg", "fcg", "cheb"), default="cheb")
    q.add_argument("--k", type=int, default=4)
    q.add_argument("--ncv", type=int, default=30)
    a = p.parse_args()
    if a.cmd == "harness":
        return cmd_harness(a)
    a.muA, a.rA = (float(x) for x in a.pair.split(","))
    return cmd_prod(a)


if __name__ == "__main__":
    sys.exit(main())
