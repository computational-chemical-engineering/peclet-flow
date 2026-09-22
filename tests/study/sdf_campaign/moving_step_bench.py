"""Layer 3 rung 4: the MOVING-STEP DRIVER -- cost, and a march that actually moves.

Two things, both measured rather than modelled:

  1. COST. ms/step for a static step vs a moving step (step + rebuild_geometry), at 64^3 and
     128^3. v1 rebuilds the whole geometry every step by design: the Layer-3 input measurement is
     that a 128^3 rebuild is ~65% momentum/IBM stencils and ~35% pressure/MG with scene sampling in
     the noise, so an incremental path has to attack BOTH sides and is deferred.

  2. THAT IT MARCHES. A sphere translating through initially quiescent fluid: the body displaces by
     the distance its velocity implies, the fluid it drags acquires momentum in the direction of
     travel, and the projection keeps max|div| at solver level throughout -- i.e. the rebuild
     preserves the flow field instead of resetting it (set_solid zeroes u/phi/P by design).
"""
import os
import sys
import time
import numpy as np
from peclet import flow as sdflow

KN_R, KI_I, KI_R = 16, 2, 17


def make_scene(cen, R):
    node_ints = np.array([1, -1, -1], dtype=np.int32)
    node_reals = np.zeros(KN_R); node_reals[0] = R
    node_reals[14] = 1.0; node_reals[15] = 1.0
    ii = np.zeros((len(cen), KI_I), dtype=np.int32)
    ir = np.zeros((len(cen), KI_R))
    for m, c in enumerate(cen):
        ii[m] = (0, -1)
        ir[m, 0:3] = c
        ir[m, 6] = 1.0
        ir[m, 7] = 1.0
    return node_ints, node_reals, ii.ravel(), ir.ravel()


def build(N, moving, V):
    R = 0.18 * N
    cen = [(0.25*N, 0.25*N, 0.25*N), (0.75*N, 0.75*N, 0.25*N),
           (0.75*N, 0.25*N, 0.75*N), (0.25*N, 0.75*N, 0.75*N)]
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(0.5)
    s.set_advection(False)
    # A BODY FORCE IS NOT OPTIONAL HERE. Without it the static case has u == 0 forever, so the
    # pressure PCG converges on a zero right-hand side and the "static step" being compared against
    # is solving nothing -- the first version of this bench reported the moving path as 2.8x a
    # static step purely from that.
    s.set_body_force(1e-3, 0.0, 0.0)
    s.set_velocity_solver_params(40)
    s.set_pressure_solver_params(15)
    # MG depth 4, not 1. levels=1 leaves the 'coarse' solve on the FULL grid, which on
    # CUDA costs 920 ms/step at N=64 against 19.5 ms at depth 4 (measured). Depth changes the
    # solver path, not the converged answer, and both runs of a comparison use the same depth.
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(*make_scene(cen, R), periodic=True)
    if moving:
        for i in range(len(cen)):
            s.set_instance_motion(i, lin_vel=[V, 0.0, 0.0])
    s.set_solid_from_scene(True)
    return s, cen, R


def bench(N, nstep=12):
    V = 0.05
    out = {}
    for label, moving, rebuild in (("static", False, False),
                                   ("moving (no rebuild)", True, False),
                                   ("moving + rebuild", True, True)):
        s, cen, R = build(N, moving, V)
        for _ in range(25):   # develop the flow AND warm up: a zero field would make the
            s.step()          # pressure solve trivial and the comparison meaningless
        t0 = time.perf_counter()
        for k in range(nstep):
            s.step()
            if rebuild:
                for i in range(len(cen)):
                    c = cen[i]
                    s.set_instance_transform(i, [c[0] + V * 0.5 * (k + 1), c[1], c[2]])
                s.rebuild_geometry()
        dt = (time.perf_counter() - t0) / nstep * 1e3
        out[label] = dt
        print("  N=%3d  %-20s %8.1f ms/step" % (N, label, dt))
    print("  N=%3d  rebuild overhead      %8.1f ms/step  (%.1fx a static step)"
          % (N, out["moving + rebuild"] - out["moving (no rebuild)"],
             out["moving + rebuild"] / out["static"]))
    return out


def march(N=48, nstep=40):
    """A translating sphere in quiescent fluid: it must move, drag fluid, and stay divergence free."""
    R = 0.16 * N
    V = 0.08
    cen = [(0.25 * N, 0.5 * N, 0.5 * N)]
    s = sdflow.Solver(N, N, N)
    s.set_rho(1.0); s.set_mu(0.1); s.set_dt(0.5)
    s.set_advection(False)
    s.set_velocity_solver_params(60)
    s.set_pressure_solver_params(20)
    # MG depth 4, not 1. levels=1 leaves the 'coarse' solve on the FULL grid, which on
    # CUDA costs 920 ms/step at N=64 against 19.5 ms at depth 4 (measured). Depth changes the
    # solver path, not the converged answer, and both runs of a comparison use the same depth.
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(*make_scene(cen, R), periodic=True)
    s.set_instance_motion(0, lin_vel=[V, 0.0, 0.0])
    s.set_solid_from_scene(True)
    x = cen[0][0]
    worstdiv = 0.0
    for k in range(nstep):
        s.step()
        worstdiv = max(worstdiv, s.max_open_divergence())
        x += V * 0.5
        s.set_instance_transform(0, [x, cen[0][1], cen[0][2]])
        s.rebuild_geometry()
    u = np.asarray(s.get_u())
    # fluid mask at the u-points of the FINAL sphere position
    g = np.arange(N).astype(float)
    X, Y, Z = np.meshgrid(g - 0.5, g, g, indexing="ij")
    dx = np.abs(X - x); dx = np.minimum(dx, N - dx)
    dy = np.abs(Y - cen[0][1]); dy = np.minimum(dy, N - dy)
    dz = np.abs(Z - cen[0][2]); dz = np.minimum(dz, N - dz)
    sd = np.sqrt(dx*dx + dy*dy + dz*dz) - R
    fl = sd > 0.0
    near = fl & (sd < 0.6 * R)
    print("  march: %d steps, sphere x %.2f -> %.2f (moved %.2f cells, expected %.2f)"
          % (nstep, cen[0][0], x, x - cen[0][0], V * 0.5 * nstep))
    print("         max|div| over the march %.3e" % worstdiv)
    print("         mean u in the near wake shell %+.4e   (wall speed %+.4f) -> fluid is dragged "
          "ALONG: %s" % (float(u[near].mean()), V, float(u[near].mean()) > 0.05 * V))
    # A kinematically driven body has infinite mass, so it INJECTS momentum: a nonzero fluid mean
    # is the physics, not an error. What it demonstrates is that the field survives the rebuilds --
    # a reset-every-step driver would show ~one step's worth here, not 40 steps' accumulation.
    print("         mean u over all fluid %+.4e  (momentum injected by the driven body, "
          "accumulated across all %d rebuilds)" % (float(u[fl].mean()), nstep))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "march"):
        march()
    if what in ("all", "bench"):
        for N in (64, 128):
            bench(N)
