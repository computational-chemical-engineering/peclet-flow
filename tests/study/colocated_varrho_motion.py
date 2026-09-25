#!/usr/bin/env python
"""WO-V3 (doc/collocated_varrho_forces.md §8, Q4/Q5) -- an INSTRUMENT that decides the motion
rating of the collocated variable-density path; report only, not a gate.

1. Translating drop, ratio 1000: a heavy drop (R = 6) in a periodic 32^3 box, seeded with the
   uniform velocity U = 0.05 along x, no gravity, sigma = 0.01, mu = 0.01. The exact solution
   translates rigidly with U forever. After the drop has crossed one box length: max|u - U|
   (spurious currents), the colour volume drift, and the L1 shape error against the initial colour
   (the box is periodic, so after one length the exact colour IS the initial one). Staggered
   (enable_vof_momentum, the ratio-1000 reference), collocated V8 with the balanced-force
   projection ON (the default) and OFF.
2. Rayleigh-Taylor, ratio 3 (Atwood 0.5): tests/study/rayleigh_taylor.py's sharp-VoF case on both
   grids, the interface amplitude at t = 40, 80, ..., 240.

Run: OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=<build> python tests/study/colocated_varrho_motion.py
"""
import numpy as np

import peclet.flow as F

N, R, RATIO, SIGMA, MU, U0 = 32, 6.0, 1000.0, 0.01, 0.01, 0.05


def sphere_c(sub=4):
    off = (np.arange(sub) + 0.5) / sub
    c = np.array([N / 2 + 0.13, N / 2 + 0.27, N / 2 + 0.11])
    out = np.zeros((N, N, N))
    for a in off:
        for b in off:
            for d in off:
                X, Y, Z = np.meshgrid(np.arange(N) + a, np.arange(N) + b, np.arange(N) + d,
                                      indexing="ij")
                out += ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 < R * R)
    return np.asfortranarray(out / sub ** 3)


def translating(cls, bfp=None):
    s = cls(N, N, N)
    s.set_rho(RATIO)
    s.set_mu(MU)
    s.set_dt(1.0)
    s.set_advection(True)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, N), 10.0)))
    s.enable_vof()
    c0 = sphere_c()
    s.set_vof(c0)
    s.set_property_model("rho", "linear", "C", [1.0, RATIO - 1.0])
    s.set_surface_tension(SIGMA)
    if cls is F.Solver:
        s.enable_vof_momentum(1.0, RATIO)
    if bfp is not None:
        s.set_balanced_force_projection(bfp)
    dt = min(0.5 * s.capillary_dt(), 0.2 / U0)
    s.set_dt(dt)
    s.set_field("u", np.asfortranarray(np.full((N, N, N), U0)))
    steps = int(round(N / (U0 * dt)))
    v0 = float(np.sum(c0))
    worst = 0.0
    for _ in range(steps):
        s.step()
        du = max(np.max(np.abs(np.asarray(s.get_u()) - U0)), np.max(np.abs(s.get_v())),
                 np.max(np.abs(s.get_w())))
        if not np.isfinite(du):
            return steps, np.inf, np.inf, np.inf, np.inf
        worst = max(worst, du)
    c = np.asarray(s.get_vof())
    return (steps, du / U0, worst / U0, (np.sum(c) - v0) / v0, np.sum(np.abs(c - c0)) / v0)


def rayleigh_taylor_vof(cls, N=48, NZ=96, g=0.005, mu=0.002, steps=240, cfl=0.2, dt0=1.0):
    s = cls(N, 4, NZ)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_dt(dt0)
    s.set_advection(True)
    s.set_domain_bc("-z", "wall", (0, 0, 0))
    s.set_domain_bc("+z", "wall", (0, 0, 0))
    s.set_pressure_geometry(np.asfortranarray(np.full((N, 4, NZ), 10.0)))
    s.enable_vof()
    x, z = np.arange(N), np.arange(NZ)
    zi = NZ / 2 + 1.5 * np.cos(2 * np.pi * x / N)
    c0 = np.clip((z[None, :] + 1.0) - zi[:, None], 0.0, 1.0)
    s.set_vof(np.asfortranarray(np.repeat(c0[:, None, :], 4, axis=1)))
    s.set_property_model("rho", "linear", "C", [1.0, 2.0])
    s.set_property_model("force_z", "linear", "rho", [0.0, -g])

    def amp():
        h = np.asarray(s.get_vof())[:, 1, :].sum(axis=1)
        zc = NZ - h
        return 0.5 * (zc.max() - zc.min())

    t, dt, sample, hist = 0.0, dt0, 40.0, [amp()]
    while t < steps - 1e-9:
        c = s.vof_max_courant()
        if c > 0:
            dt = min(2.0 * dt, dt * cfl / c, dt0)
        dt = min(dt, sample - t)
        s.set_dt(dt)
        s.step()
        t += dt
        if t >= sample - 1e-9:
            hist.append(amp())
            sample += 40.0
    return hist


def main():
    print(f"1. translating drop, ratio {RATIO:g}, R {R:g}, U {U0:g}, one box length")
    for name, cls, bfp in (("staggered (vof momentum)", F.Solver, None),
                           ("collocated V8, BFP ON", F.SolverColocated, None),
                           ("collocated V8, BFP OFF", F.SolverColocated, False)):
        try:
            st, fin, worst, dv, shape = translating(cls, bfp)
            print(f"   {name:26s} {st} steps  max|u-U|/U final {fin:.3e} worst {worst:.3e}  "
                  f"volume drift {dv:.2e}  L1 shape err {shape:.3e}")
        except RuntimeError as exc:
            print(f"   {name:26s} THREW: {exc}")
    print("2. Rayleigh-Taylor ratio 3 (sharp VoF), amplitude at t = 0, 40, ..., 240")
    for name, cls in (("staggered", F.Solver), ("collocated V8 (BFP ON)", F.SolverColocated)):
        try:
            h = rayleigh_taylor_vof(cls)
            print(f"   {name:24s} " + " -> ".join(f"{a:.2f}" for a in h))
        except RuntimeError as exc:
            print(f"   {name:24s} THREW: {exc}")


if __name__ == "__main__":
    main()
