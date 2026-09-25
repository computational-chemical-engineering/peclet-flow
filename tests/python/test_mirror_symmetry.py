#!/usr/bin/env python
"""Wall-normal mirror symmetry of the MAC step, and the placement of the variable viscosity.

A bubble column between two walls (y = 0 and y = NY, x/z periodic) collected its bubbles at the
LOW wall -- and so did the same column with every bubble mirrored to the other side. The cause
was the variable-viscosity momentum operator: it averaged mu across the faces of CELL i, but the
staggered unknown u_c(i) lives on the -c face of cell i, so every component saw mu shifted half a
cell towards +c -- first-order accurate, and in a walled direction a mirror asymmetry (VarFaceProps,
src/face_props.hpp). Two checks:

  1. mirror pair -- solver A and its y-mirror B (cell fields reflected, v negated onto the mirrored
     face), same fixed dt, A(y) against mirror(B)(y) after a few steps. Round-off only (~1e-14);
     the shifted viscosity gave 1e-3..1e-2. Variants: single phase with a transported scalar
     driving rho, the body force and (or not) mu, arithmetic and harmonic viscosity means; the
     structured VoF with CSF; the block container with its per-marker CSF. The blob touches the
     LOW wall (the mirror touches the high one) and straddles the periodic x seam.
  2. placement -- steady variable-viscosity Stokes, u = A (sin kx cos ky, -cos kx sin ky, 0),
     mu = 1 + sin(kx + 0.3) cos(ky + 0.7) / 2, forced by -div(mu grad u_c): second order in h
     (the shifted viscosity was first order: 1.39 then 1.05).

Run: OMP_NUM_THREADS=4 PYTHONPATH=<build> python tests/python/test_mirror_symmetry.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

try:
    import peclet.flow as pf
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

NX, NY, NZ = 24, 20, 12
DT, NSTEPS = 0.05, 3
R = 3.6


def mir_c(f):
    """Mirror of a cell-centred (and x/z-face) field: row j <-> NY-1-j."""
    return np.asfortranarray(f[:, ::-1, :])


def mir_v(v):
    """Mirror of the y-face field: get_v's index j is the face j - 1/2 (index 0 the low wall face),
    so the face j maps to NY - j and changes sign."""
    out = np.zeros_like(v)
    out[:, 1:, :] = -v[:, :0:-1, :]
    return np.asfortranarray(out)


def sphere(c, r, sub=6):
    g = (np.arange(sub) + 0.5) / sub
    frac = np.zeros((NX, NY, NZ))
    for gx in g:
        for gy in g:
            for gz in g:
                X, Y, Z = np.meshgrid(np.arange(NX) + gx, np.arange(NY) + gy, np.arange(NZ) + gz,
                                      indexing="ij")
                dx = X - c[0]
                dx -= NX * np.round(dx / NX)  # periodic in x: the blob straddles the seam
                frac += (dx * dx + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) < r * r
    return np.asfortranarray(frac / sub ** 3)


def build(mode, mirror, variable_mu=True, harmonic=False):
    cy = 2.0 if mode == "scalar" else 3.0  # scalar: straddles the wall; markers: 0.6 cell inside
    c = (0.4, NY - cy if mirror else cy, 6.1)
    s = pf.Solver(NX, NY, NZ)
    s.set_rho(1.0)
    s.set_mu(0.05)
    s.set_domain_bc("-y", "wall", (0, 0, 0))
    s.set_domain_bc("+y", "wall", (0, 0, 0))
    s.set_pressure_geometry(np.full((NX, NY, NZ), 10.0, order="F"))
    if mode == "scalar":
        s.add_scalar("T", 0.0, "koren")
        s.set_field("T", sphere(c, R))
        s.set_property_model("rho", "linear", "T", [1.0, -0.5])
        if variable_mu:
            s.set_property_model("mu", "linear", "T", [0.05, -0.04])
        s.set_property_model("force_x", "linear", "T", [0.0, -0.02])
    else:
        s.enable_vof()
        s.set_vof(sphere(c, R) if mode == "vof" else np.zeros((NX, NY, NZ), order="F"))
        s.set_property_model("rho", "linear", "C", [1.0, -0.9])
        s.set_property_model("mu", "linear", "C", [0.05, -0.04])
        s.set_surface_tension(0.02)
        if mode == "blocks":
            s.enable_vof_blocks([(c[0], c[1], c[2], R)])
            s.enable_vof_block_csf()
        s.set_property_model("force_x", "linear", "C", [0.0, -0.02])
    if harmonic:
        s.diagnostics.set_property_mode("variable", True)
    s.set_pressure_pcg(True, 800, 1e-12)
    s.set_superficial_velocity(True, "x", 0.0)
    rng = np.random.default_rng(1)
    u, v, w = (np.asfortranarray(0.05 * rng.standard_normal((NX, NY, NZ))) for _ in range(3))
    v[:, 0, :] = 0.0  # the low wall face
    if mirror:
        u, v, w = mir_c(u), mir_v(v), mir_c(w)
    for k, a in enumerate((u, v, w)):
        s.set_velocity(k, a)
    return s


def mirror_asymmetry(mode, **kw):
    A, B = build(mode, False, **kw), build(mode, True, **kw)
    for _ in range(NSTEPS):
        for s in (A, B):
            s.set_dt(DT)
            s.step()

    def rel(a, b):
        return float(np.abs(a - b).max() / np.abs(a).max())

    vel = max(rel(A.get_u(), mir_c(B.get_u())), rel(A.get_v(), mir_v(B.get_v())),
              rel(A.get_w(), mir_c(B.get_w())))
    return vel, rel(A.get_p(), mir_c(B.get_p()))


def steady_varmu_error(N, A=1e-3, nz=4):
    k = 2.0 * np.pi / N
    xc = np.arange(N) + 0.5
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    mu = 1 + 0.5 * np.sin(k * X + 0.3) * np.cos(k * Y + 0.7)
    mux = 0.5 * k * np.cos(k * X + 0.3) * np.cos(k * Y + 0.7)
    muy = -0.5 * k * np.sin(k * X + 0.3) * np.sin(k * Y + 0.7)
    ux, uy = A * np.sin(k * X) * np.cos(k * Y), -A * np.cos(k * X) * np.sin(k * Y)
    fx = -(-2 * k * k * mu * ux + A * k * (mux * np.cos(k * X) * np.cos(k * Y)
                                           - muy * np.sin(k * X) * np.sin(k * Y)))
    fy = -(-2 * k * k * mu * uy + A * k * (mux * np.sin(k * X) * np.sin(k * Y)
                                           - muy * np.cos(k * X) * np.cos(k * Y)))
    rep = lambda a: np.asfortranarray(np.repeat(a[:, :, None], nz, axis=2))  # noqa: E731
    s = pf.Solver(N, N, nz)
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(2.0)
    s.set_advection(False)
    s.set_pressure_pcg(True, 400, 1e-13)
    s.diagnostics.set_velocity_solver_params(400, 1e-12, 2)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, nz), 10.0)))
    s.enable_cell_force()
    s.diagnostics.set_property_mode("variable", False)
    # plain incremental pressure: its fixed point is the steady Stokes solution exactly (the
    # rotational correction with a variable mu is not -- see set_variable_rotational)
    s.diagnostics.set_variable_rotational("off", 1.0)
    s.set_field("mu", rep(mu))
    for nm, a in (("force_x", rep(fx)), ("force_y", rep(fy)), ("force_z", rep(0 * fx))):
        s.set_field(nm, a)
        s.diagnostics.exchange_field(nm)
    prev = None
    for _ in range(20000):
        s.step()
        u = np.array(s.get_u())
        if prev is not None and np.max(np.abs(u - prev)) <= 1e-13 * np.max(np.abs(u)):
            break
        prev = u
    u, v = np.array(s.get_u())[:, :, 0], np.array(s.get_v())[:, :, 0]
    i = np.arange(N, dtype=np.float64)
    xu, yc = np.meshgrid(i, i + 0.5, indexing="ij")  # u on the -x face: (x_i, y_c)
    xc2, yv = np.meshgrid(i + 0.5, i, indexing="ij")  # v on the -y face: (x_c, y_j)
    ue = A * np.sin(k * xu) * np.cos(k * yc)
    ve = -A * np.cos(k * xc2) * np.sin(k * yv)
    return max(np.abs(u - ue).max(), np.abs(v - ve).max()) / A


def main():
    ok = True
    cases = [("scalar, constant mu", "scalar", dict(variable_mu=False)),
             ("scalar, mu(T) arithmetic", "scalar", {}),
             ("scalar, mu(T) harmonic", "scalar", dict(harmonic=True)),
             ("structured VoF + CSF", "vof", {}),
             ("block VoF + block CSF", "blocks", {})]
    for name, mode, kw in cases:
        vel, p = mirror_asymmetry(mode, **kw)
        good = vel < 1e-10 and p < 1e-8  # p carries the 1e-12 PCG tolerance, not round-off
        ok &= good
        print(f"mirror  {name:28s} velocity {vel:.2e}  pressure {p:.2e}  "
              f"{'ok' if good else 'FAIL'}")
    e16, e32 = steady_varmu_error(16), steady_varmu_error(32)
    order = np.log2(e16 / e32)
    good = order > 1.8
    ok &= good
    print(f"placement  variable-mu steady Stokes error N=16 {e16:.3e}  N=32 {e32:.3e}  "
          f"order {order:.2f} (> 1.8)  {'ok' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
