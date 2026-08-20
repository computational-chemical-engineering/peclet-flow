#!/usr/bin/env python
"""S1 probe (doc/collocated_accuracy_ceiling.md #5): does the ABC cell/face reconciliation carry a
steady-state defect that could hold the ~0.3 % flow-excess plateau?

Structural background (this is what the test checks, not assumes): at the JOINT fixed point of the
mode-9 step -- velocity AND accumulated pressure both stationary -- the rotational update
P += (rho/dt)*phi - mu*div(u*) together with A phi = -div(u*) forces (A + rho/(mu dt)) phi = 0,
i.e. phi = 0 and div(u*) = 0.  Consequences, each measured here at the march's stopping point:

  m1  |uf - halfavg(u)| / <u>      the projected face field vs the plain 1/2-1/2 average of the
                                   cell field.  projectCorrect writes uf = halfavg(u*) - grad_f(phi)
                                   and the cell correction writes u = u* - gpCenterGrad(phi), so a
                                   persistent gap is EXACTLY the un-converged gauge (phi != 0) plus
                                   the gradient-pair mismatch: the S1 channel.  If m1 ~ 0 the ABC
                                   two-field loop closes and S1 is dead.
  m2  alpha-div(halfavg(u)) h/<u>  the constraint residual of the CELL field (the face field's is
                                   pinned by the solve; the cell field's is the approximate
                                   projection's O(h^2) remainder -- it should CONVERGE, not floor).
  m3  |u_cell - cellavg(uf)| / <u> the doc's S1 quantity, split by wall distance (near = |sdf|<=2h).
                                   This is a smoothing difference (h^2 D2u/4 in the bulk); the
                                   interesting part is whether the NEAR-WALL part floors.

Usage:  SDFLOW_BUILD=build_ge BED=<packing.npz> python collocated_s1_reconciliation.py [N ...]
        (defaults N = 96 128 192 -> R = 6, 8, 12 on the 16^3-box beds)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow  # noqa: E402

BED = os.environ.get("BED", "")
if not BED:
    raise SystemExit("BED=<packing.npz> is required (use the phi=0.60 s3 bed)")
MARCH_TOL = float(os.environ.get("MARCH_TOL", "1e-8"))
MAXS = int(os.environ.get("MARCH_MAX", "4000"))
DT = float(os.environ.get("DT", "60.0"))


def bed_sdf(N, npz):
    pk = np.load(npz)
    box = np.asarray(pk["box"], float)
    assert np.allclose(box, box[0]), f"{npz}: box {box} is not cubic"
    Rc = N / box[0]
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


def march(N):
    sdf, R = bed_sdf(N, BED)
    s = flow.SolverColocated(N, N, N)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(DT)
    s.set_body_force(1e-3, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(150)
    s.set_pressure_multigrid(True, max(2, int(np.log2(N)) - 2))
    s.set_pressure_pcg(True, 300, 1e-8)
    if hasattr(s, "set_collocated_scheme"):
        s.set_collocated_scheme("gauge-exact")
    else:
        s.set_face_interp(9)
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    prev = 0.0
    for it in range(MAXS):
        s.step()
        if it % 10 == 9:
            um = float(np.asarray(s.get_u()).mean())
            if it > 10 and abs(um - prev) < MARCH_TOL * (abs(um) + 1e-30):
                break
            prev = um
    return s, sdf, R, it + 1


def stats(name, v, scale):
    v = np.abs(v) / scale
    if v.size == 0:
        print(f"    {name:>34}: (empty)")
        return None
    q = np.quantile(v, [0.5, 0.99])
    print(f"    {name:>34}: max {v.max():.3e}  p99 {q[1]:.3e}  med {q[0]:.3e}  "
          f"rms {np.sqrt((v**2).mean()):.3e}", flush=True)
    return float(np.sqrt((v ** 2).mean()))


prev = {}
for N in [int(x) for x in (sys.argv[1:] or [96, 128, 192])]:
    s, sdf, R, steps = march(N)
    U = [np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w())]
    UF = [np.asarray(s.get_uf()), np.asarray(s.get_vf()), np.asarray(s.get_wf())]
    OX = [np.asarray(s.get_ox()), np.asarray(s.get_oy()), np.asarray(s.get_oz())]
    fluid = sdf >= 0.0
    uscale = float(np.abs(U[0][fluid]).mean())
    print(f"\nN={N} R={R:.1f} steps={steps} <|u|>_fluid={uscale:.4e} "
          f"k/R^2={float(U[0].mean())/1e-3*0.1/R**2:.6e}")
    rms = {}
    for a in range(3):
        half = 0.5 * (U[a] + np.roll(U[a], 1, axis=a))       # halfavg at the low-a face of cell i
        openf = OX[a] > 0.0
        rms[f"m1_ax{a}"] = stats(f"m1 |uf-halfavg(u)| ax{a} (open faces)",
                                 (UF[a] - half)[openf], uscale)
    # m2: alpha-divergence of the CELL field via the halfavg fluxes (x-fastest, low-face layout:
    # div_i = sum_a o_a(i+e_a)*f_a(i+e_a) - o_a(i)*f_a(i))
    div = np.zeros_like(U[0])
    for a in range(3):
        half = 0.5 * (U[a] + np.roll(U[a], 1, axis=a))
        flx = OX[a] * half
        div += np.roll(flx, -1, axis=a) - flx
    stats("m2 alpha-div(halfavg(u)) fluid", div[fluid], uscale)  # per-cell, h=1 units
    # face-field residual for reference (should be at the solve tolerance)
    divf = np.zeros_like(U[0])
    for a in range(3):
        flx = OX[a] * UF[a]
        divf += np.roll(flx, -1, axis=a) - flx
    stats("m2f alpha-div(uf) fluid (solver)", divf[fluid], uscale)
    # m3: cell field vs the cell-average of the face field, split by wall distance
    for a in range(3):
        cavg = 0.5 * (UF[a] + np.roll(UF[a], -1, axis=a))     # avg of the two a-faces of cell i
        d = (U[a] - cavg)
        near = fluid & (np.abs(sdf) <= 2.0)
        bulk = fluid & (np.abs(sdf) > 2.0)
        rms[f"m3n_ax{a}"] = stats(f"m3 |u-cellavg(uf)| ax{a} NEAR wall", d[near], uscale)
        rms[f"m3b_ax{a}"] = stats(f"m3 |u-cellavg(uf)| ax{a} bulk", d[bulk], uscale)
    if prev:
        lr = np.log(N / prev["N"])
        orders = {k: np.log(prev[k] / v) / lr for k, v in rms.items()
                  if k in prev and v and prev[k]}
        print("    orders vs previous N (positive = decays):",
              " ".join(f"{k}:{o:+.2f}" for k, o in orders.items()), flush=True)
    prev = {"N": N, **{k: v for k, v in rms.items() if v}}
