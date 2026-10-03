#!/usr/bin/env python
"""Gate: the HIGH-side domain face of a MAC face field survives the ghost fills (both grids).

Why this test exists. That face is the first GHOST index along its axis, and the ghost fill wraps
every axis periodically, so it came back carrying the LOW boundary's plane -- the inlet on the
outlet (doc/uf_outlet_fix.md). Nothing registered could see it: momentum reads the outlet through
openFaceView, the state hashes record cell fields only, and max_open_divergence reads the
zero-gradient view. The readers that DID see it were the scalar transport, the backflow census
and, at a high-side INFLOW face, collocated momentum.

Checks, in the order they would fail:
  1. census == required: on a channel driven backwards (fluid enters through the +x outflow face),
     outflow_backflow() reads the raw outlet plane; its max_reverse must equal max(-u_out) of the
     flux that closes the last cell's divergence (built from the inner faces), at every step, to
     1e-9 (the reconstruction differences O(1) faces at the pressure tolerance; broken: O(1));
  2. a c == 1 scalar (D = 0, first-order upwind) stays 1 in the OUTLET column after one step;
  3. staggered: max_open_divergence_projected stays at round-off after that scalar step (the
     scalar's fill used to overwrite C[0].u's corrected outlet face in place: 0.785);
  4. a HIGH-side inflow (+x inflow, -x outflow): the collocated inlet-adjacent centreline velocity
     agrees with the staggered one (1.7e-3 fixed; 3.6e-2 when it read the outlet's plane).

Run:  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \
          python tests/python/test_open_face_plane.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

NX, NY, NZ = 24, 12, 4
MU, DT = 0.1, 0.5


def channel(pf, cls, u_in, low="inflow", nx=NX, ny=NY):
    s = cls(nx, ny, NZ)
    s.set_rho(1.0); s.set_mu(MU); s.set_dt(DT)
    s.set_advection(True); s.set_advection_scheme("koren")
    if cls is pf.SolverColocated:
        s.set_collocated_scheme("gauge-exact")
    if low == "inflow":
        s.set_domain_bc("-x", "inflow", (u_in, 0.0, 0.0))
        s.set_domain_bc("+x", "outflow")
    else:
        s.set_domain_bc("-x", "outflow")
        s.set_domain_bc("+x", "inflow", (u_in, 0.0, 0.0))
    s.set_domain_bc("-y", "wall"); s.set_domain_bc("+y", "wall")
    s.set_pressure_geometry(np.asfortranarray(np.full((nx, ny, NZ), 10.0)))
    s.set_pressure_pcg(True, 400, 1e-11)
    return s


def required_outlet(s):
    """The +x outlet flux that closes the last cell's discrete divergence (cell units)."""
    uf, vf, wf = (np.asarray(s.get_uf()), np.asarray(s.get_vf()), np.asarray(s.get_wf()))
    i = NX - 1
    vtop = np.zeros((NY, NZ)); vtop[:-1] = vf[i, 1:, :]  # +y wall face = 0
    wtop = np.roll(wf[i], -1, axis=1)                     # z periodic
    return uf[i] - (vtop - vf[i]) - (wtop - wf[i])


def check_census(pf, cls):
    s = channel(pf, cls, -1.0)
    worst = 0.0
    for _ in range(20):
        s.step()
        ob = s.diagnostics.outflow_backflow()
        req = required_outlet(s)
        worst = max(worst, abs(ob["max_reverse"] - (-req).max()))
        assert ob["reversed_faces"] == ob["outlet_faces"] == NY * NZ, ob
    assert worst <= 1e-9, f"{cls.__name__}: census vs required outlet flux {worst:.3e}"
    return worst


def check_scalar_outlet(pf, cls):
    s = channel(pf, cls, 1.0)
    for _ in range(30):
        s.step()
    s.add_scalar("c", diffusivity=0.0, scheme="fou", iters=1)
    s.set_scalar_bc("c", "-x", "dirichlet", 1.0)
    s.set_scalar_bc("c", "+x", "neumann", 0.0)
    s.set_field("c", np.asfortranarray(np.ones((NX, NY, NZ))))
    s.step()
    c = np.asarray(s.get_field("c"))
    out = float(np.abs(c[NX - 1] - 1.0).max())
    assert out <= 1e-10, f"{cls.__name__}: c = 1 outlet column moved by {out:.3e}"
    div = float(s.max_open_divergence_projected())
    if cls is pf.Solver:
        assert div <= 1e-12, f"staggered max_open_divergence_projected {div:.3e} after a scalar step"
    return out, div


def check_high_inflow(pf):
    nx, ny = 48, 16
    u = {}
    for cls in (pf.SolverColocated, pf.Solver):
        s = channel(pf, cls, -1.0, low="outflow", nx=nx, ny=ny)
        for _ in range(100):
            s.step()
        u[cls] = float(np.asarray(s.get_u())[nx - 1, ny // 2, 0])
    d = abs(u[pf.SolverColocated] - u[pf.Solver])
    assert d < 1e-2, (f"high-side inflow: collocated inlet-cell u {u[pf.SolverColocated]:.5f} vs "
                      f"staggered {u[pf.Solver]:.5f}")
    return d


def main():
    try:
        import peclet.flow as pf
    except ImportError:
        print("SKIP: peclet.flow not importable (set PYTHONPATH to the build tree)")
        return 77
    try:
        for cls in (pf.SolverColocated, pf.Solver):
            w = check_census(pf, cls)
            out, div = check_scalar_outlet(pf, cls)
            print(f"  {cls.__name__:16s} census-required {w:.2e}  outlet |c-1| {out:.2e}  "
                  f"div_projected {div:.2e}")
        d = check_high_inflow(pf)
        print(f"  high-side inflow: |u_colo - u_stag| at the inlet cell {d:.2e}")
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    print("OK: high-side face planes survive the ghost fills")
    return 0


if __name__ == "__main__":
    sys.exit(main())
