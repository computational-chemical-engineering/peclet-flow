#!/usr/bin/env python
"""A/B of the COLLOCATED advecting velocity: the projected divergence-free MAC face field
`uf/vf/wf` (the shipped scheme) against the un-projected cell->face average 1/2(u_i+u_j) (the
phase-2 form, kept as `diagnostics.set_uf_advection(False)`).

Why this exists: `doc/flow_colocated_plan.md` §1 step 3 prescribes the projected field ("these u_f
become the advecting velocities for the next step's advection"), `colocated_advection.hpp` said the
swap was pending, and the FOU operator's conservative row-sum identity needs a divergence-free
advecting field (../docs/decisions/flow.md:977).  The swap landed 2026-09-21; this script is the
evidence behind it and the instrument for re-judging it.

Four measurements, each printed for BOTH choices:

  div    -- max |div_h(advecting field)| at the end of a Taylor-Green run.  This is the structural
            claim, not a tuning number: the projection makes `uf` solenoidal to solver tolerance,
            the cell->face average of the *corrected cell* field is not (the residual is intrinsic
            to cell-centred velocity placement -- it is why the coupling is an APPROXIMATE
            projection).  Measured here in Python on the same 7-point stencil `divergOpen` uses.
  tg     -- 2-D Taylor-Green in a triply-periodic box: an EXACT Navier-Stokes solution (the
            nonlinear term is balanced by the pressure gradient), so the L2 error and its observed
            order are absolute, not relative to a reference run.
  shift  -- the same vortex Galilean-shifted by a uniform mean velocity, no body force, periodic:
            the continuum answer is a mean momentum that is exactly constant, so any drift is the
            discrete advection failing to conserve.
  bed    -- the production regime the dt^2 scaling of the two choices' difference points at
            (../../amr/docs/amr_flow_uniform_parity.md §3a): steady large-dt driving through a
            cut-cell sphere bed, advection on, reported as <u_x> (proportional to permeability at
            fixed forcing) with the STAGGERED solver's answer on the same geometry as the
            reference -- the staggered stored velocity IS the projected face velocity, so it has
            no A/B to make and is the closest thing to a known answer here.

Run (host-openmp).  Use PYTHONPATH, not PECLET_FLOW_BUILD: the suite venv carries an INSTALLED
`peclet.flow`, and `_bootstrap`'s first rule (deliberately) keeps whatever is already importable,
so PECLET_FLOW_BUILD is ignored there and the study would silently measure the installed wheel.
Every run prints which module it resolved -- read that line before the numbers.

  OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=$PWD/build_uf \\
      python tests/study/uf_advection_ab.py [div|tg|shift|bed|all]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from _bootstrap import ensure_flow  # noqa: E402

flow = ensure_flow()

NU, DT, RHO = 0.05, 0.5, 1.0


# --------------------------------------------------------------------------------------- fields
def tg_fields(N, nz, U0, amp, shift=0.0):
    """Taylor-Green at decay amplitude `amp`, optionally Galilean-shifted in x by `shift`."""
    k = 2.0 * np.pi / N
    ix = np.arange(N)
    X, Y = np.meshgrid(ix, ix, indexing="ij")
    u2 = U0 * amp * np.sin(k * X) * np.cos(k * Y) + shift
    v2 = -U0 * amp * np.cos(k * X) * np.sin(k * Y)
    u = np.repeat(u2[:, :, None], nz, axis=2)
    v = np.repeat(v2[:, :, None], nz, axis=2)
    w = np.zeros((N, N, nz))
    return np.asfortranarray(u), np.asfortranarray(v), np.asfortranarray(w)


def packing_sdf(N, rfrac=0.18):
    """The eight-sphere periodic bed of tests/regression/state_hash.py (same geometry, so the
    numbers here and the byte gate's talk about the same scene)."""
    R = rfrac * N
    cs = np.array([0.25 * N, 0.75 * N])
    xs = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    best = np.full((N, N, N), 1e30)
    for sx in cs:
        for sy in cs:
            for sz in cs:
                dx = X - sx; dx -= N * np.round(dx / N)
                dy = Y - sy; dy -= N * np.round(dy / N)
                dz = Z - sz; dz -= N * np.round(dz / N)
                best = np.minimum(best, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return np.asfortranarray(best)


# ---------------------------------------------------------------------------------- divergences
def div_periodic_faces(uf, vf, wf):
    """div_h of a MAC face field in flow's LOW-face convention (entry i = the -axis face of cell
    i), on a fully periodic all-fluid block: (uf(i+1) - uf(i)) + ... , spacing 1."""
    return ((np.roll(uf, -1, 0) - uf) + (np.roll(vf, -1, 1) - vf) + (np.roll(wf, -1, 2) - wf))


def div_of_advecting(s, use_uf):
    """max |div_h| of the field the advection operators actually read, on a periodic all-fluid box.

    use_uf  -- the solver's own projected face field (`get_uf`/`get_vf`/`get_wf`, already the low-face
               convention).
    else    -- the cell->face average `cadv::adv_vel` forms from the cell field: the LOW face of
               cell i is 1/2(u(i-1) + u(i)).
    """
    if use_uf:
        f = [np.asarray(g()) for g in (s.get_uf, s.get_vf, s.get_wf)]
    else:
        cell = [np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w())]
        f = [0.5 * (np.roll(cell[c], 1, c) + cell[c]) for c in range(3)]
    return float(np.max(np.abs(div_periodic_faces(*f))))


# ---------------------------------------------------------------------------------------- runs
def run_tg(cls, N, use_uf, nz=4, U0=1.0, steps=40, shift=0.0, dt=DT):
    s = cls(N, N, nz)
    s.set_rho(RHO)
    s.set_mu(RHO * NU)
    s.set_dt(dt)
    s.set_advection(True)
    if use_uf is not None:
        s.diagnostics.set_uf_advection(use_uf)
    s.diagnostics.set_velocity_solver_params(80)
    s.set_solid(np.asfortranarray(np.full((N, N, nz), 1e3)), cutcell_pressure=True)
    u0, v0, w0 = tg_fields(N, nz, U0, 1.0, shift)
    s.set_state(u0, v0, w0)
    for _ in range(steps):
        s.step()
    return s, (u0, v0, w0)


def tg_error(s, u0, N, nz, U0, steps, shift, dt):
    """L2 error against the exact decayed (and shifted) solution, normalized by the initial norm."""
    k = 2.0 * np.pi / N
    amp = np.exp(-2.0 * NU * k * k * dt * steps)
    ue, ve, _ = tg_fields(N, nz, U0, amp, shift)
    uu, vv = np.asarray(s.get_u()), np.asarray(s.get_v())
    num = np.sqrt(np.mean((uu - ue) ** 2 + (vv - ve) ** 2))
    den = np.sqrt(np.mean(np.asarray(u0[0]) ** 2 + np.asarray(u0[1]) ** 2))
    return float(num / den)


def run_bed(cls, N, use_uf, steps, dt, mu=0.02, f0=1e-3):
    """Steady large-dt driving through the cut-cell bed, advection ON and implicit (stable at a
    dt far above the CFL limit -- the regime the two choices differ in)."""
    s = cls(N, N, N)
    s.set_rho(RHO)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force((f0, 0.0, 0.0))
    s.set_advection(True)
    s.set_implicit_advection(True)
    if use_uf is not None:
        s.diagnostics.set_uf_advection(use_uf)
    s.diagnostics.set_velocity_solver_params(80)
    s.set_pressure_multigrid(True, 3)
    s.set_pressure_pcg(True, 400, 1e-10)
    s.set_solid(packing_sdf(N), cutcell_pressure=True)
    for _ in range(steps):
        s.step()
    sdf = np.asarray(packing_sdf(N))
    fluid = sdf > 0.0
    ux = np.asarray(s.get_u())
    return float(np.mean(np.where(fluid, ux, 0.0))), float(s.max_open_divergence())


# ------------------------------------------------------------------------------------- reports
def report_div(Ns=(32, 64)):
    print("=== div: max |div_h(advecting field)| after 40 Taylor-Green steps (periodic, all fluid) ===")
    print(f"{'N':>5} {'projected uf':>14} {'1/2(u_i+u_j)':>14} {'ratio':>9}")
    for N in Ns:
        s, _ = run_tg(flow.SolverColocated, N, True)
        d_uf = div_of_advecting(s, True)
        d_av = div_of_advecting(s, False)
        print(f"{N:5d} {d_uf:14.3e} {d_av:14.3e} {d_av / max(d_uf, 1e-300):9.1e}")


def report_tg(Ns=(32, 64, 128), steps=40):
    print("=== tg: 2-D Taylor-Green, L2 error vs the EXACT solution (nu=0.05, dt=0.5, 40 steps) ===")
    print(f"{'N':>5} {'advecting velocity':>20} {'L2 error':>12} {'order':>7} {'maxdiv(uf)':>12}")
    prev = {}
    for N in Ns:
        for use_uf, name in ((True, "projected uf"), (False, "1/2(u_i+u_j)")):
            s, u0 = run_tg(flow.SolverColocated, N, use_uf, steps=steps)
            l2 = tg_error(s, u0, N, 4, 1.0, steps, 0.0, DT)
            order = ""
            if name in prev:
                order = f"{np.log2(prev[name] / l2):7.2f}"
            prev[name] = l2
            print(f"{N:5d} {name:>20} {l2:12.4e} {order:>7} {s.max_open_divergence():12.2e}")


def report_shift(Ns=(32, 64), steps=120, shift=0.5, dt=0.1):
    # dt=0.1 (CFL ~0.15 on |u|max = U0 + shift): the periodic case has no domain BC, so advection
    # is EXPLICIT and dt=0.5 is past the stability limit -- at DT the average diverges to NaN at
    # both resolutions and the projected field survives only at N=64.  That asymmetry is real but
    # it is a stability observation, not a conservation measurement; this runs both inside the
    # stable range so the drift below is the discretization's and nothing else's.
    print(f"=== shift: Galilean-shifted vortex (mean u = {shift}), momentum drift over {steps} steps (dt={dt}) ===")
    print(f"{'N':>5} {'advecting velocity':>20} {'|<u>-<u>_0|':>13} {'|<v>-<v>_0|':>13}")
    for N in Ns:
        for use_uf, name in ((True, "projected uf"), (False, "1/2(u_i+u_j)")):
            s, u0 = run_tg(flow.SolverColocated, N, use_uf, steps=steps, shift=shift, dt=dt)
            du = abs(float(np.mean(np.asarray(s.get_u()))) - float(np.mean(np.asarray(u0[0]))))
            dv = abs(float(np.mean(np.asarray(s.get_v()))) - float(np.mean(np.asarray(u0[1]))))
            print(f"{N:5d} {name:>20} {du:13.3e} {dv:13.3e}")


def report_bed(N=32, steps=200, dt=20.0):
    print(f"=== bed: cut-cell sphere bed, advection ON + implicit, dt={dt}, {steps} steps ===")
    print(f"{'solver':>28} {'<u_x>_fluid':>13} {'max_open_div':>13}")
    for cls, use_uf, name in (
        (flow.Solver, None, "staggered (reference)"),
        (flow.SolverColocated, True, "collocated, projected uf"),
        (flow.SolverColocated, False, "collocated, 1/2(u_i+u_j)"),
    ):
        ux, div = run_bed(cls, N, use_uf, steps, dt)
        print(f"{name:>28} {ux:13.6e} {div:13.2e}")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("div", "all"):
        report_div()
    if what in ("tg", "all"):
        report_tg()
    if what in ("shift", "all"):
        report_shift()
    if what in ("bed", "all"):
        report_bed()


if __name__ == "__main__":
    main()
