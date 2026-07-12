"""Zick & Homsy comparison with ANALYTIC-SDF exact IBM points: cut-cell vs ghost (1,1)/(1,2)/(2,2).

Exact geometry for the simple-cubic sphere lattice (phi = 0.125), all in grid units:
  * exact wall crossings t[c][k] (line-sphere intersection, nearest image) feed
    set_exact_crossings -> exact theta in BOTH the momentum cut-cell stencil and the
    ghost-projection closures (replaces the O(h^2) linear-interp anchoring);
  * exact face apertures (analytic disk-chord integration) feed set_openness_override ->
    the cut-cell projection's openness is exact.

Reports, per scheme and N: Z&H drag error, and L1/L2/Linf norms of the velocity and pressure
fields against a fine cut-cell-exact reference (default N=192), interpolated to the coarse
sample points by periodic cubic splines (scipy map_coordinates, grid-wrap). Norms over fluid
sample points, relative to the reference field's rms on the same mask; pressure mean-aligned.
Linf additionally reported on the interior fluid (sdf > 2 h_coarse) because the reference
spline carries O(h_ref^2) kink error in the first wall band. Pairwise Richardson orders.

Run:  SDFLOW_BUILD=build_cuda2 python tests/study/staggered_zh_exact_norms.py [--quick]
      (--run-ref to (re)compute the reference; otherwise loaded from zh_exact_ref_N<ref>.npz)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.environ.get("SDFLOW_BUILD", "build_cuda2")))
import numpy as np
from scipy.ndimage import map_coordinates

from peclet import flow

PHI = 0.125
MU = 0.1
F = 1e-3
KREF = 4.2920
HERE = os.path.dirname(os.path.abspath(__file__))


def radius(N):
    return (3 * PHI / (4 * np.pi)) ** (1 / 3) * N


# ---------------------------------------------------------------- exact geometry (grid units)
def point_grid(N, c):
    """Staggered point coordinates for component c (or cell centers for c=None)."""
    g = np.arange(N, dtype=np.float64)
    off = [0.5, 0.5, 0.5]
    if c is not None:
        off[c] = 0.0
    return np.meshgrid(g + off[0], g + off[1], g + off[2], indexing="ij")


def nimg(d, N):
    return d - N * np.round(d / N)


def sdf_at(N, X, Y, Z):
    C = 0.5 * N
    return np.sqrt(nimg(X - C, N) ** 2 + nimg(Y - C, N) ** 2 + nimg(Z - C, N) ** 2) - radius(N)


def exact_crossings(N):
    """t[(c*3+k)*n + i]: exact crossing fraction from component-c staggered point i toward its
    +k neighbour (NaN = no sign change). Line-sphere with nearest image; roots of
    t^2 + 2 t d_k + |d|^2 - R^2 = 0 restricted to the sign-changing segment."""
    R = radius(N)
    C = 0.5 * N
    out = np.full((3, 3, N, N, N), np.nan)
    for c in range(3):
        X, Y, Z = point_grid(N, c)
        d = [nimg(X - C, N), nimg(Y - C, N), nimg(Z - C, N)]
        d2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
        s_here = np.sqrt(d2) - R
        for k in range(3):
            s_nb = np.roll(s_here, -1, axis=k)  # exact sdf at the +k neighbour point (periodic)
            cross = (s_here < 0) != (s_nb < 0)
            disc = d[k] ** 2 - (d2 - R * R)
            ok = cross & (disc > 0)
            sq = np.sqrt(np.where(ok, disc, 1.0))
            t1 = -d[k] - sq
            t2 = -d[k] + sq
            eps = 1e-12
            in1 = ok & (t1 >= -eps) & (t1 <= 1 + eps)
            in2 = ok & (t2 >= -eps) & (t2 <= 1 + eps)
            t = np.where(in1, t1, np.where(in2, t2, np.nan))
            out[c, k] = np.clip(t, 0.0, 1.0)
            miss = cross & ~np.isfinite(out[c, k])
            if miss.any():  # tangent-grazing straddles: fall back to linear interp there
                tl = s_here / (s_here - s_nb)
                out[c, k][miss] = np.clip(tl[miss], 0.0, 1.0)
    # solver expects flat x-fastest blocks, ordered [(c*3 + k)]
    return np.concatenate([out[c, k].ravel(order="F") for c in range(3) for k in range(3)])


def exact_openness(N):
    """Exact fluid area fraction of every -a face (analytic chord integral of the disk cut,
    1024-pt midpoint rule; faces are unit squares in grid units)."""
    R = radius(N)
    C = 0.5 * N
    M = 1024
    res = []
    for a in range(3):
        o = np.ones((N, N, N))
        t1a, t2a = (a + 1) % 3, (a + 2) % 3
        w = nimg(np.arange(N, dtype=np.float64) - C, N)     # face-plane coord along a (x=i etc.)
        rho2 = R * R - w * w
        lo = np.arange(N, dtype=np.float64)                  # square low edges (t1, t2)
        for i in np.nonzero(rho2 > 0)[0]:
            rho = np.sqrt(rho2[i])
            j1 = np.nonzero(np.abs(nimg(lo + 0.5 - C, N)) < rho + 1.0)[0]
            j2 = np.nonzero(np.abs(nimg(lo + 0.5 - C, N)) < rho + 1.0)[0]
            if len(j1) == 0 or len(j2) == 0:
                continue
            J1, J2 = np.meshgrid(j1, j2, indexing="ij")
            lo1 = nimg(lo[J1.ravel()] - C, N)                # displaced to sphere frame
            lo2 = nimg(lo[J2.ravel()] - C, N)
            t = lo1[:, None] + (np.arange(M)[None, :] + 0.5) / M
            half = np.sqrt(np.maximum(rho * rho - t * t, 0.0))
            zlo = np.maximum(-half, lo2[:, None])
            zhi = np.minimum(half, lo2[:, None] + 1.0)
            solid = np.clip(zhi - zlo, 0.0, None).mean(axis=1)
            idx = [None, None, None]
            idx[a] = np.full(len(lo1), i)
            idx[t1a] = J1.ravel()
            idx[t2a] = J2.ravel()
            o[idx[0], idx[1], idx[2]] -= solid
        # axes of the array are (a, t1, t2) by construction order -> reorder to (x, y, z)
        o = np.clip(np.moveaxis(o, [0, 1, 2], [a, t1a, t2a]), 0.0, 1.0)
        # The scheme's flux DOF lives at the face's staggered velocity point and is PINNED to 0
        # when that point is solid (maskVelocity). A nonzero aperture there would count flux the
        # scheme cannot carry (and re-masking after the projection would break the just-projected
        # divergence — measured 1.9e-3 residual). Exact-openness therefore means: exact aperture
        # on faces whose velocity point is fluid, zero otherwise — same structure as the sampled
        # ccFaceOpen, with the retained apertures exact.
        Xf, Yf, Zf = point_grid(N, a)
        o = np.where(sdf_at(N, Xf, Yf, Zf) >= 0.0, o, 0.0)
        res.append(o)
    return res


# ---------------------------------------------------------------- solver run
def run(N, scheme, warm_tol=1e-7, tail=40, max_steps=5000, dt=80.0):
    X, Y, Z = point_grid(N, None)
    sdf = sdf_at(N, X, Y, Z)
    lv = max(2, int(np.log2(N)) - 1)
    s = flow.Solver(N, N, N)
    s.set_rho(1.0)
    s.set_mu(MU)
    s.set_dt(dt)
    s.set_body_force(F, 0, 0)
    s.set_advection(False)
    s.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=lv)
    s.set_pressure_pcg(True, 400, 1e-10)
    s.set_exact_crossings(exact_crossings(N))
    if scheme == "cutcell":
        ox, oy, oz = exact_openness(N)
        s.set_openness_override(ox.ravel(order="F"), oy.ravel(order="F"), oz.ravel(order="F"))
    else:
        mo, ro = {"g22": (2, 2), "g11": (1, 1), "g12": (1, 2)}[scheme]
        s.set_ghost_projection(True, matrix_order=mo, rhs_order=ro)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev, warm, um, t0 = 0.0, None, [], time.time()
    for it in range(max_steps):
        s.step()
        m = float(s.get_u().mean())
        um.append(m)
        if warm is None:
            if it % 10 == 9:
                if it > 10 and abs(m - prev) < warm_tol * (abs(m) + 1e-30):
                    warm = it
                prev = m
        elif it - warm >= tail:
            break
    K = F * N**3 / (6 * np.pi * MU * radius(N) * np.mean(um[-tail:]))
    fields = dict(u=s.get_u(), v=s.get_v(), w=s.get_w(), p=s.get_p())
    return dict(K=K, steps=it + 1, iters=s.last_pressure_iterations(),
                div=s.max_open_divergence(), secs=time.time() - t0, **fields)


# ---------------------------------------------------------------- norms vs reference
def sample_ref(ref, refN, N, comp):
    """Cubic-spline (periodic) sample of a reference field at the N-grid points of comp."""
    r = refN / N
    g = np.arange(N, dtype=np.float64)
    off = [0.5, 0.5, 0.5]
    if comp is not None:
        off[comp] = 0.0
    # reference field index coords: point (grid units, coarse) * r - offset of ref array coords
    co = [None, None, None]
    roff = [0.5, 0.5, 0.5]
    if comp is not None:
        roff[comp] = 0.0
    for a in range(3):
        co[a] = (g + off[a]) * r - roff[a]
    CX, CY, CZ = np.meshgrid(co[0], co[1], co[2], indexing="ij")
    return map_coordinates(ref, [CX, CY, CZ], order=3, mode="grid-wrap", prefilter=True)


def norms(fields, refs, refN, N):
    """Relative L1/L2/Linf of velocity (3 comps pooled) and pressure vs the reference.
    The solver works in grid units (dx = 1), so each N is a RESCALED physical problem:
    u ~ F N^2/mu and p ~ F N. Nondimensionalize by bringing the coarse fields to the
    reference scale (u * (refN/N)^2, p * refN/N) before differencing."""
    su = (refN / N) ** 2
    sp = refN / N
    fields = dict(u=fields["u"] * su, v=fields["v"] * su, w=fields["w"] * su,
                  p=fields["p"] * sp)
    hC = 1.0  # grid units; interior band uses 2 cells of the COARSE grid = 2*refN/N ref units
    out = {}
    # velocity: pool the three components' fluid samples
    num = {1: 0.0, 2: 0.0}
    den = {1: 0.0, 2: 0.0}
    linf = 0.0
    linf_int = 0.0
    npts = 0
    for c, key in enumerate("uvw"):
        X, Y, Z = point_grid(N, c)
        sd = sdf_at(N, X, Y, Z)
        m = sd > 0
        mi = sd > 2.0 * hC
        rf = sample_ref(refs[key], refN, N, c)
        d = np.abs(fields[key] - rf)
        num[1] += d[m].sum()
        num[2] += (d[m] ** 2).sum()
        den[1] += np.abs(rf[m]).sum()
        den[2] += (rf[m] ** 2).sum()
        linf = max(linf, float(d[m].max()))
        if mi.any():
            linf_int = max(linf_int, float(d[mi].max()))
        npts += int(m.sum())
    uref = np.sqrt(den[2] / npts)
    out["u"] = (num[1] / den[1], np.sqrt(num[2] / npts) / uref, linf / uref, linf_int / uref)
    # pressure: cell centers, mean-aligned on the fluid mask
    X, Y, Z = point_grid(N, None)
    sd = sdf_at(N, X, Y, Z)
    m = sd > 0
    mi = sd > 2.0 * hC
    rf = sample_ref(refs["p"], refN, N, None)
    dp = fields["p"] - rf
    dp -= dp[m].mean()
    rf0 = rf - rf[m].mean()
    pref = np.sqrt((rf0[m] ** 2).mean())
    d = np.abs(dp)
    out["p"] = (d[m].sum() / np.abs(rf0[m]).sum(), np.sqrt((d[m] ** 2).mean()) / pref,
                float(d[m].max()) / pref, float(d[mi].max()) / pref if mi.any() else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--run-ref", action="store_true")
    ap.add_argument("--ref-n", type=int, default=None)
    args = ap.parse_args()
    refN = args.ref_n or (96 if args.quick else 192)
    Ns = [16, 24, 32] if args.quick else [24, 32, 48, 64, 96]
    ref_file = os.path.join(HERE, f"zh_exact_ref_N{refN}.npz")

    if args.run_ref or not os.path.exists(ref_file):
        print(f"[ref] running cut-cell-exact reference at N={refN} ...", flush=True)
        r = run(refN, "cutcell")
        np.savez_compressed(ref_file, u=r["u"], v=r["v"], w=r["w"], p=r["p"], K=r["K"])
        print(f"[ref] done: K={r['K']:.4f} err={100 * (r['K'] - KREF) / KREF:+.3f}% "
              f"steps={r['steps']} secs={r['secs']:.0f}", flush=True)
    z = np.load(ref_file)
    refs = {k: z[k] for k in "uvwp"}
    print(f"[ref] N={refN}, K={float(z['K']):.4f} ({100 * (float(z['K']) - KREF) / KREF:+.3f}% "
          f"vs Z&H)", flush=True)

    schemes = ["cutcell", "g22", "g11", "g12"]
    hdr = (f"{'scheme':>8} {'N':>4} | {'K err%':>8} {'ord':>5} | "
           f"{'u L1':>8} {'u L2':>8} {'u Linf':>8} {'Linf>2h':>8} {'ordL2':>5} | "
           f"{'p L1':>8} {'p L2':>8} {'p Linf':>8} {'Linf>2h':>8} {'ordL2':>5} | it  secs")
    print(hdr, flush=True)
    for scheme in schemes:
        prev = None
        for N in Ns:
            r = run(N, scheme)
            nm = norms(r, refs, refN, N)
            eK = 100 * (r["K"] - KREF) / KREF
            oK = np.log(abs(prev["eK"]) / abs(eK)) / np.log(N / prev["N"]) if prev else float("nan")
            oU = (np.log(prev["u2"] / nm["u"][1]) / np.log(N / prev["N"])) if prev else float("nan")
            oP = (np.log(prev["p2"] / nm["p"][1]) / np.log(N / prev["N"])) if prev else float("nan")
            print(f"{scheme:>8} {N:>4} | {eK:>+8.3f} {oK:>5.2f} | "
                  f"{nm['u'][0]:>8.2e} {nm['u'][1]:>8.2e} {nm['u'][2]:>8.2e} {nm['u'][3]:>8.2e} "
                  f"{oU:>5.2f} | "
                  f"{nm['p'][0]:>8.2e} {nm['p'][1]:>8.2e} {nm['p'][2]:>8.2e} {nm['p'][3]:>8.2e} "
                  f"{oP:>5.2f} | {r['iters']:>2d} {r['secs']:>5.0f}", flush=True)
            prev = dict(eK=eK, u2=nm["u"][1], p2=nm["p"][1], N=N)
        print(flush=True)


if __name__ == "__main__":
    main()
