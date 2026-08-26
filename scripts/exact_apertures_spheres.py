#!/usr/bin/env python
"""EXACT analytic face apertures for sphere packings -> peclet.flow set_openness_override.

The in-solver aperture estimators (set_aperture_order 1/2) are limited to the trilinearly
sampled SDF field; for ANALYTIC geometry this module computes machine-precision apertures
(closed-form disk-rectangle overlap, validated to 4e-16) and returns the three staggered
openness fields (ox[i] = -x face of cell i, x-fastest [x,y,z] arrays).

Provenance: the aperture-bias investigation (flow doc/collocated_paper_plan.md row 51) -- the
one-sample linear model carries a signed convexity bias (+0.59%/+0.27% bed permeability at
R=8/12, decaying ~h^2); exact apertures remove it. Multi-sphere faces are handled by clamped
subtraction (a slight over-closing where solid disks of different spheres overlap on one face
-- rare, counted, and conservative). A sub-resolution floor (1e-6) is REQUIRED: alpha ~ 1e-12
rows destroy the pressure-operator conditioning (measured: GPU watchdog-level solve stalls).
For general (non-sphere) analytic SDFs the same interface should be fed by a high-order
implicit quadrature (R. Saye, SIAM J. Sci. Comput. 37(2), 2015) -- not implemented here.

Usage:
    from exact_apertures_spheres import exact_openness
    ox, oy, oz = exact_openness(N, centers_cells, radii_cells)   # cubic N, periodic
    s.set_openness_override(ox.ravel(order="F"), oy.ravel(order="F"), oz.ravel(order="F"))
"""
import numpy as np


def Fquad(x, y, r):
    """Vectorized area of {u<x, v<y, u^2+v^2<r^2}, disk at origin."""
    x = np.clip(x, -r, r)
    yc = np.clip(y, -r, r)

    def G(t):
        t = np.clip(t, -r, r)
        return 0.5 * (t * np.sqrt(np.maximum(r * r - t * t, 0.0)) + r * r * np.arcsin(t / r))

    us = np.sqrt(np.maximum(r * r - yc * yc, 0.0))
    # middle band u in [max(-r,-us), min(x, us)]: contribute y + s(u)
    a_m = np.maximum(-r, -us)
    b_m = np.minimum(x, us)
    mid = np.where(b_m > a_m, yc * (b_m - a_m) + (G(b_m) - G(a_m)), 0.0)
    # outer bands (|u| > us): full 2s where y > 0, zero where y < 0
    a1, b1 = -r + 0.0 * x, np.minimum(x, -us)         # left band
    left = np.where(b1 > a1, 2.0 * (G(b1) - G(a1)), 0.0)
    a2, b2 = us, np.maximum(x, us)                    # right band (only if x > us)
    right = np.where(b2 > a2, 2.0 * (G(b2) - G(a2)), 0.0)
    outer = np.where(y >= 0, left + right, 0.0)
    F = mid + outer
    F = np.where(y >= r, 2.0 * (G(x) - G(-r)), F)
    F = np.where(y <= -r, 0.0, F)
    return F


def disk_cell_areas(cy, cz, rho, j0, j1, k0, k1):
    """Vectorized exact disk-cell overlap areas on cells [j,j+1]x[k,k+1], j in [j0,j1), k in [k0,k1)."""
    j = np.arange(j0, j1)
    k = np.arange(k0, k1)
    X0 = j[:, None] - cy
    X1 = X0 + 1.0
    Y0 = k[None, :] - cz
    Y1 = Y0 + 1.0
    return (Fquad(X1, Y1, rho) - Fquad(X0, Y1, rho) - Fquad(X1, Y0, rho) + Fquad(X0, Y0, rho))


def exact_openness(N, C, Rr):
    """o[a][i,j,k]: openness of the -a face of cell (i,j,k); union over spheres by subtraction
    with clamping (multi-sphere faces are rare; counted and reported)."""
    o = [np.ones((N, N, N)) for _ in range(3)]
    multi = 0
    for sh in np.stack(np.meshgrid(*[[-1., 0., 1.]] * 3, indexing="ij"), -1).reshape(-1, 3):
        cs = C + sh * N
        keep = np.all((cs + (Rr + 2)[:, None] > 0) & (cs - (Rr + 2)[:, None] < N + 1), axis=1)
        for (scx, scy, scz), rr in zip(cs[keep], Rr[keep]):
            cen = (scx, scy, scz)
            for a in range(3):
                t1, t2 = (a + 1) % 3, (a + 2) % 3
                p0 = max(int(np.ceil(cen[a] - rr)), 0)
                p1 = min(int(np.floor(cen[a] + rr)), N - 1)
                for p in range(p0, p1 + 1):
                    d = p - cen[a]
                    rho2 = rr * rr - d * d
                    if rho2 <= 0:
                        continue
                    rho = np.sqrt(rho2)
                    j0 = max(int(np.floor(cen[t1] - rho)), 0)
                    j1 = min(int(np.ceil(cen[t1] + rho)), N)
                    k0 = max(int(np.floor(cen[t2] - rho)), 0)
                    k1 = min(int(np.ceil(cen[t2] + rho)), N)
                    if j0 >= j1 or k0 >= k1:
                        continue
                    A = disk_cell_areas(cen[t1], cen[t2], rho, j0, j1, k0, k1)
                    idx = [slice(None)] * 3
                    idx[a] = p
                    idx[t1] = slice(j0, j1)
                    idx[t2] = slice(k0, k1)
                    view = o[a][tuple(idx)]
                    # view axes = the remaining array axes in ascending order; A is (t1, t2)
                    Ax = A.T if t1 > t2 else A
                    multi += int(((view < 1.0) & (Ax > 1e-12)).sum())
                    np.subtract(view, Ax, out=view)
                    np.clip(view, 0.0, 1.0, out=view)
    print(f"  exact_openness: multi-sphere faces (union approximated): {multi}", flush=True)
    for a in range(3):  # floor: sub-1e-6 apertures cannot carry resolved flux; keep rows sane
        o[a][o[a] < 1e-6] = 0.0
    return o


