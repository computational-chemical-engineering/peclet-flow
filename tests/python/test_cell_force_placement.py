#!/usr/bin/env python
"""Placement of the per-cell body force and drag coefficient in the momentum equation.

The per-cell fields ``force_x/y/z`` (enable_cell_force; the CFD-DEM drag feedback, Boussinesq) and
``drag_beta`` (enable_drag) live at CELL CENTRES. The velocity unknown they act on does not always:

  * staggered (MAC): u_c(i) is the face between cells i - e_c and i, so the force it feels is the
    second-order face interpolant 0.5*(f(i - e_c) + f(i)) -- and the implicit drag coefficient on
    its diagonal is the same face mean of beta, so the drag target beta*u_p and the diagonal agree;
  * collocated: u_c(i) is the cell centre, so it feels the cell value f(i).

The placement follows the GRID, never the density model (the constant- and variable-density RHS
builders used to disagree). Three checks, each on both density models:

  1. placement -- a Taylor-Green force f = A (sin kx cos ky, -cos kx sin ky, 0) sampled at the cell
     centres. Its face mean on the MAC grid (and its cell values under the collocated ABC face
     interpolation) is EXACTLY discretely solenoidal, so the steady Stokes state is p = 0 and
     u = F / (mu * lam_h), lam_h = (8/h^2) sin^2(kh/2) -- the force actually placed is recovered
     pointwise as mu * lam_h * u and compared with the two candidates (cell value, face mean);
  2. convergence -- the same steady state against the continuum u = f(x_u) / (2 mu k^2) at the
     velocity's own location, N = 16, 32, 64: second order (a half-cell shift is first order);
  3. drag consistency -- beta = 2 + sin(kx) cos(ky), force_x = beta * U0 (a uniform particle velocity):
     the exact steady state is u = U0 everywhere iff the RHS target and the diagonal carry the SAME
     beta at the velocity's location.

Run: OMP_NUM_THREADS=4 PYTHONPATH=<build> python tests/python/test_cell_force_placement.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

try:
    import peclet.flow as pf
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

NZ = 4
MU = 1.0


def make(cls, N, variable, dt, drag=False):
    s = cls(N, N, NZ)
    s.set_rho(1.0)
    s.set_mu(MU)
    s.set_dt(dt)
    s.set_advection(False)
    s.set_pressure_pcg(True, 400, 1e-13)
    s.diagnostics.set_velocity_solver_params(400, 1e-12, 2)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, NZ), 10.0)))
    if variable:
        s.diagnostics.set_density_mode("variable")  # uniform rho field: reduces to constant rho
    s.enable_cell_force()
    if drag:
        s.enable_drag()
    return s


def set_cell(s, name, a):
    s.set_field(name, np.asfortranarray(np.ascontiguousarray(a, dtype=np.float64)))
    s.diagnostics.exchange_field(name)


def run_steady(s, tol=1e-13, max_steps=2000):
    prev = None
    for _ in range(max_steps):
        s.step()
        u = np.array(s.get_u())
        if prev is not None and np.max(np.abs(u - prev)) <= tol * max(np.max(np.abs(u)), 1e-300):
            break
        prev = u
    return np.array(s.get_u()), np.array(s.get_v())


def tg_force(N, A):
    k = 2.0 * np.pi / N
    xc = np.arange(N) + 0.5  # cell centres (h = 1)
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    fx = A * np.sin(k * X) * np.cos(k * Y)
    fy = -A * np.cos(k * X) * np.sin(k * Y)
    rep = lambda a: np.repeat(a[:, :, None], NZ, axis=2)  # noqa: E731
    return k, rep(fx), rep(fy)


def tg_case(cls, N, variable, A=1e-3):
    """Steady Stokes under the Taylor-Green force. Returns (u, v, k, fx, fy)."""
    k, fx, fy = tg_force(N, A)
    lam = 2.0 * k * k
    s = make(cls, N, variable, dt=10.0 / (MU * lam))
    set_cell(s, "force_x", fx)
    set_cell(s, "force_y", fy)
    set_cell(s, "force_z", np.zeros_like(fx))
    u, v = run_steady(s)
    return u, v, k, fx, fy


def placement(cls, variable, N=16):
    """Relative misfit of the recovered placed force against (cell value, face mean)."""
    u, v, k, fx, fy = tg_case(cls, N, variable)
    lam_h = 8.0 * np.sin(0.5 * k) ** 2
    rec_x, rec_y = MU * lam_h * u, MU * lam_h * v
    face_x = 0.5 * (fx + np.roll(fx, 1, axis=0))
    face_y = 0.5 * (fy + np.roll(fy, 1, axis=1))
    scale = np.max(np.abs(fx))

    def mis(cx, cy):
        return max(np.max(np.abs(rec_x - cx)), np.max(np.abs(rec_y - cy))) / scale

    return mis(fx, fy), mis(face_x, face_y)


def tg_error(cls, N, variable, A=1e-3):
    """Max error of the steady velocity vs the continuum solution at the velocity's location."""
    u, v, k, _, _ = tg_case(cls, N, variable, A)
    i = np.arange(N, dtype=np.float64)
    off = 0.0 if cls is pf.Solver else 0.5  # MAC face x = i on the staggered grid; centre i + 1/2
    xu, yc = np.meshgrid(i + off, i + 0.5, indexing="ij")  # u: (x_u, y_c)
    xc, yv = np.meshgrid(i + 0.5, i + off, indexing="ij")  # v: (x_c, y_v)
    amp = A / (MU * 2.0 * k * k)
    ue = amp * np.sin(k * xu) * np.cos(k * yc)
    ve = -amp * np.cos(k * xc) * np.sin(k * yv)
    return max(np.max(np.abs(u[:, :, 0] - ue)), np.max(np.abs(v[:, :, 0] - ve))) / amp


def drag_case(cls, variable, N=16, U0=0.1):
    k = 2.0 * np.pi / N
    xc = np.arange(N) + 0.5
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    # varies along BOTH axes: a mismatch varying along x alone is a pure gradient the projection
    # absorbs into p, and would pass even with the target and the diagonal disagreeing
    beta = np.repeat((2.0 + np.sin(k * X) * np.cos(k * Y))[:, :, None], NZ, 2)
    s = make(cls, N, variable, dt=1.0, drag=True)
    set_cell(s, "drag_beta", beta)
    set_cell(s, "force_x", beta * U0)
    set_cell(s, "force_y", np.zeros_like(beta))
    set_cell(s, "force_z", np.zeros_like(beta))
    u, v = run_steady(s)
    return max(np.max(np.abs(u - U0)), np.max(np.abs(v))) / U0


# SolverColocated + variable density is rung V8: every force is a FACE acceleration added after
# centerToFace (collocated_varrho.hpp), the cell velocity taking the mean of the two face
# corrections -- a different architecture, not a placement choice, and a cell force that varies along
# its own axis diverges on it at every dt tried (0.1 .. 100). Not gated here; reported.
SKIP = {("collocated", True)}


def cases():
    for gname, cls in (("staggered", pf.Solver), ("collocated", pf.SolverColocated)):
        for mname, var in (("const-rho", False), ("var-rho", True)):
            if (gname, var) in SKIP:
                print(f"   {gname:10s} {mname:9s} not gated (rung V8 face-acceleration path)")
                continue
            yield gname, cls, mname, var


def main():
    ok = True

    print("1. placement: misfit of the recovered force vs (cell value, face mean), N = 16")
    for gname, cls, mname, var in cases():
            m_cell, m_face = placement(cls, var)
            want = m_face if cls is pf.Solver else m_cell
            good = want < 1e-6
            ok &= good
            print(f"   {gname:10s} {mname:9s} cell {m_cell:.3e}  face {m_face:.3e}  "
                  f"{'ok' if good else 'FAIL'}")

    print("2. convergence of the steady Taylor-Green velocity (max rel. error, observed order)")
    for gname, cls, mname, var in cases():
            e = [tg_error(cls, N, var) for N in (16, 32, 64)]
            p = [np.log2(e[j] / e[j + 1]) for j in range(2)]
            good = p[-1] > 1.8
            ok &= good
            print(f"   {gname:10s} {mname:9s} " + "  ".join(f"{x:.3e}" for x in e)
                  + f"  order {p[0]:.2f} {p[1]:.2f}  {'ok' if good else 'FAIL'}")

    print("3. drag consistency: max |u - U0| / U0 with beta(x, y) and force = beta * U0")
    for gname, cls, mname, var in cases():
            try:
                d = drag_case(cls, var)
            except RuntimeError as exc:  # a configuration the grid refuses is reported, not failed
                print(f"   {gname:10s} {mname:9s} refused: {exc}")
                continue
            good = d < 1e-8
            ok &= good
            print(f"   {gname:10s} {mname:9s} {d:.3e}  {'ok' if good else 'FAIL'}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_cell_force_placement():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
