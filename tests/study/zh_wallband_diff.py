#!/usr/bin/env python
"""WHERE does the collocated-vs-staggered difference live, and does its amplitude grow?

The permeability gap does not vanish under refinement, and the constraint operator was exonerated
a-priori (net flux defect ~1e-5). If the surviving difference is localised in a wall band, the
arithmetic is forced: the band's VOLUME fraction shrinks as O(h), so for its contribution to the
volume-averaged velocity to stay constant the AMPLITUDE in the band must grow as O(1/h).

So: run both solvers on identical geometry, compare the alpha-weighted FACE fluxes (the conserved
quantity, and a like-for-like basis -- staggered get_u IS the face field, collocated get_uf is its
projected one), and split the difference by distance to the wall in CELL units.

  amplitude ~ 1/h  => mechanism found: a gauge/scaling error in the near-wall reconstruction
  amplitude ~ O(1) => the band contributes O(h) and the gap must come from somewhere else

    SDFLOW_BUILD=build_ge python tests/study/zh_wallband_diff.py [N ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow  # noqa: E402

PHI0 = 0.125
BED = os.environ.get("BED", "")     # pack_bed.py npz (cubic box) -> use a sphere BED, not Z&H


def bed_sdf(N, npz):
    """Periodic union-of-spheres SDF from a packing, sampled on an N^3 grid (cubic box only)."""
    pk = np.load(npz)
    box = np.asarray(pk["box"], float)
    assert np.allclose(box, box[0]), f"{npz}: box {box} is not cubic"
    Rc = N / box[0]                                   # cells per sphere radius
    c = np.asarray(pk["centers"]) * Rc
    r = np.asarray(pk["scales"]) * Rc
    ax = np.arange(N) + 0.5
    S = np.full((N, N, N), 1e30)
    for sh in np.stack(np.meshgrid(*[[-1., 0., 1.]] * 3, indexing="ij"), -1).reshape(-1, 3):
        cs = c + sh * N
        keep = np.all((cs + (r + 3)[:, None] > 0) & (cs - (r + 3)[:, None] < N), axis=1)
        for (cx, cy, cz), rr in zip(cs[keep], r[keep]):
            i0, i1 = np.searchsorted(ax, [cx - rr - 3, cx + rr + 3])
            j0, j1 = np.searchsorted(ax, [cy - rr - 3, cy + rr + 3])
            k0, k1 = np.searchsorted(ax, [cz - rr - 3, cz + rr + 3])
            if i0 >= i1 or j0 >= j1 or k0 >= k1:
                continue
            d = np.sqrt((ax[i0:i1, None, None] - cx) ** 2 + (ax[None, j0:j1, None] - cy) ** 2
                        + (ax[None, None, k0:k1] - cz) ** 2) - rr
            np.minimum(S[i0:i1, j0:j1, k0:k1], d, out=S[i0:i1, j0:j1, k0:k1])
    return np.asfortranarray(np.clip(S, -1e3, 1e3)), Rc


def solve(N, kind, mu=0.1, F=1e-3, dt=80.0, warm_tol=1e-7, tail=40, max_steps=4000):
    if BED:
        sdf, R = bed_sdf(N, BED)
    else:
        R = (3 * PHI0 / (4 * np.pi)) ** (1 / 3) * N
        g = np.arange(N) + 0.5
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        d = lambda A: A - 0.5 * N - N * np.round((A - 0.5 * N) / N)   # noqa: E731
        sdf = np.asfortranarray(np.sqrt(d(X) ** 2 + d(Y) ** 2 + d(Z) ** 2) - R)
    s = flow.Solver(N, N, N) if kind == "stag" else flow.SolverColocated(N, N, N)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    s.set_body_force(F, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(150)
    s.set_pressure_multigrid(True, max(2, int(np.log2(N)) - 1))
    s.set_pressure_pcg(True, 200, 1e-8)
    if kind != "stag":
        s.set_collocated_scheme(kind)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev, warm = 0.0, None
    for it in range(max_steps):
        s.step()
        um = float(s.get_u().mean())
        if warm is None:
            if it % 10 == 9:
                if it > 10 and abs(um - prev) < warm_tol * (abs(um) + 1e-30):
                    warm = it
                prev = um
        elif it - warm >= tail:
            break
    uf = np.asarray(s.get_u() if kind == "stag" else s.get_uf())
    return uf, np.asarray(s.get_ox()), sdf, R


if __name__ == "__main__":
    Ns = [int(x) for x in (sys.argv[1:] or [32, 64, 128])]
    if BED:
        print(f"geometry: BED {os.path.basename(BED)}")
    print(f"{'N':>5} | {'<dFlux>':>11} {'ord':>6} | {'band<=2h share':>15} | "
          f"{'max|d| band':>12} {'ord':>6} | {'rms|d| band':>12} {'ord':>6}")
    prev = None
    for N in Ns:
        fs, ox, sdf, R = solve(N, "stag")
        fc, _, _, _ = solve(N, "gauge-exact")
        # face-centred sdf along x (the face between cell i-1 and i)
        sf = 0.5 * (sdf + np.roll(sdf, 1, axis=0))
        d = ox * (fc - fs)                      # difference in the conserved face flux
        tot = float(d.mean())
        band = np.abs(sf) <= 2.0                # within 2 CELLS of the wall
        share = float(d[band].sum() / d.sum()) if abs(d.sum()) > 0 else float("nan")
        mx = float(np.abs(d[band]).max())
        rms = float(np.sqrt((d[band] ** 2).mean()))
        o = {}
        if prev:
            lr = np.log(N / prev[0])
            for k, (a, b) in enumerate(zip(("tot", "mx", "rms"), (tot, mx, rms))):
                pass
            o["tot"] = f"{np.log(abs(prev[1] / tot)) / lr:+.2f}"
            o["mx"] = f"{np.log(abs(prev[2] / mx)) / lr:+.2f}"
            o["rms"] = f"{np.log(abs(prev[3] / rms)) / lr:+.2f}"
        print(f"{N:>5} | {tot:>+11.4e} {o.get('tot',''):>6} | {share:>15.3f} | "
              f"{mx:>12.4e} {o.get('mx',''):>6} | {rms:>12.4e} {o.get('rms',''):>6}", flush=True)
        prev = (N, tot, mx, rms)
    print("\nOrders are of the QUANTITY, so a POSITIVE order means it decays under refinement.")
    print("An amplitude order near -1 (growing like 1/h) is the mechanism this test is looking for;")
    print("near 0 or positive means the wall band cannot sustain a constant gap on its own.")
