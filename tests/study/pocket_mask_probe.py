#!/usr/bin/env python3
"""Option (iv) of the pocket study: what changes if every staggered velocity point whose
PROJECTION aperture is exactly 0 is masked (pinned to zero), so that the momentum mask agrees with
the projection? A study instrument, not a gate.

The masking itself lives in a STUDY-ONLY build: src/flow_ibm_geometry.hpp, setSolidDevice, under
`#ifdef PECLET_FLOW_STUDY_MASK_CLOSED`, never committed to production code. The patch is
tests/study/pocket_mask_closed.patch: `git apply` it in a scratch worktree and configure a separate
tree with -DCMAKE_CXX_FLAGS=-DPECLET_FLOW_STUDY_MASK_CLOSED (it prints the number of points masked). This script runs one case on whichever module is on PYTHONPATH
and saves the outputs; `compare` diffs two such files (baseline build vs masked build).

  run      --case bed  : the A1 bed as a scene (pocket_pressure_probe.set_bed_scene), configured
                         as steady_acceleration_gates.Case.configure, marched to steady.
           --case sphere: one sphere of volume fraction --phi in the periodic unit cube, the
                         tests/python/test_hydro_force_units.py setup ('unit' system, N = 24,
                         nu dt / h^2 = 6), centre on a cell vertex shifted by --shift cells.
  compare  a.npz b.npz : |dK/K|, per-body reaction F / T change (relative to max |F| / |T|),
                         max |du| / max |u| away from the points masked in either run, the
                         velocities the mask removed, and the step counts.

  PYTHONPATH=build_mask python tests/study/pocket_mask_probe.py run --case bed \\
      --arrangement bed.npz --settings tight --out mask_bed_tight.npz
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pocket_pressure_probe as PP  # noqa: E402
import steady_acceleration_gates as SAG  # noqa: E402


def build_sphere(flow, a, st):
    n, L = a.N, 1.0
    s = flow.Solver((n, n, n), extent=(L, L, L))
    h = L / n
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(a.beta * h * h)
    s.set_body_force((1.0, 0.0, 0.0))
    s.set_advection(False)
    s.diagnostics.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=max(2, int(np.log2(n)) - 1))
    s.set_pressure_pcg(True, max_iter=st["pcg"][0], rtol=st["pcg"][1])
    if st["vel_rtol"] is not None:
        s.set_velocity_residual_tolerance(st["vel_rtol"])
    R = (3.0 * a.phi / (4.0 * math.pi)) ** (1.0 / 3.0) * L
    c = [0.5 * L + a.shift[k] * h for k in range(3)]
    ni = np.array([1, -1, -1], dtype=np.int32)
    nr = np.zeros(16)
    nr[0] = R
    nr[14] = 1.0
    nr[15] = 1.0
    ii = np.array([0, -1], dtype=np.int32)
    ir = np.zeros(17)
    ir[0:3] = c
    ir[6] = 1.0
    ir[7] = 1.0
    s.set_scene(ni, nr, ii, ir, periodic=True)
    s.set_solid_from_scene(cutcell_pressure=True)

    def K(u):  # Zick & Homsy normalisation
        return L ** 3 / (6.0 * math.pi * R * u)
    return s, K


def cmd_run(flow, a):
    st = SAG.settings_of(a.settings)
    if a.max_steps:
        st["max_steps"] = a.max_steps
    if a.case == "bed":
        case = SAG.Case(PP.case_args(a))
        s = PP.build(flow, case, "staggered", st)
        K = case.K
    else:
        s, K = build_sphere(flow, a, st)
    res = flow.march_to_steady(s, lambda: float(s.get_u().mean()), rtol=st["rtol"],
                               max_steps=st["max_steps"], accelerate=(a.window > 0),
                               window=max(a.window, 1))
    r = np.array(s.hydro_force_torque_reaction())
    U = [np.array(g()) for g in (s.get_u, s.get_v, s.get_w)]
    O = [np.array(g()) for g in (s.get_ox_proj, s.get_oy_proj, s.get_oz_proj)]
    print(f"{a.case} {a.settings} m={a.window}: converged={res.converged} {res.reason} "
          f"steps={res.steps} acc={res.accelerated_steps} K={K(res.monitor):.12f} "
          f"sum Fx={r[0][:, 0].sum():.12e}", flush=True)
    np.savez(a.out, K=K(res.monitor), monitor=res.monitor, steps=res.steps,
             accelerated=res.accelerated_steps, converged=res.converged, F=r[0], T=r[1],
             u=U[0], v=U[1], w=U[2], ox=O[0], oy=O[1], oz=O[2])


def cmd_compare(a):
    A, B = np.load(a.a), np.load(a.b)
    dK = float(B["K"] / A["K"] - 1.0)
    dF = float(np.abs(B["F"] - A["F"]).max() / np.abs(A["F"]).max())
    dT = float(np.abs(B["T"] - A["T"]).max() / np.abs(A["T"]).max())
    umax = max(float(np.abs(A[f]).max()) for f in "uvw")
    du, rem, nmask = 0.0, 0.0, 0
    for f, o in zip("uvw", ("ox", "oy", "oz")):
        # a point either run pinned to zero (solid, or closed in the masked build)
        pinned = (A[f] == 0.0) | (B[f] == 0.0)
        closed = (A[o] == 0.0) & (A[f] != 0.0)  # live in A although its aperture is 0
        nmask += int(closed.sum())
        rem = max(rem, float(np.abs(A[f][closed]).max()) if closed.any() else 0.0)
        du = max(du, float(np.abs(B[f] - A[f])[~pinned].max()))
    print(f"{os.path.basename(a.a)} -> {os.path.basename(a.b)}: |dK/K| {abs(dK):.3e} "
          f"(K {float(A['K']):.12f} -> {float(B['K']):.12f}); reaction F {dF:.3e}, T {dT:.3e} "
          f"(per body, / max); max|du| / max|u| off the pinned points {du / umax:.3e}; "
          f"{nmask} closed-aperture points live in the first run, max |u| there / max|u| "
          f"{rem / umax:.3e}; steps {int(A['steps'])} -> {int(B['steps'])} "
          f"(accelerated {int(A['accelerated'])} -> {int(B['accelerated'])}), converged "
          f"{bool(A['converged'])} -> {bool(B['converged'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "compare"))
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--case", choices=("bed", "sphere"), default="bed")
    ap.add_argument("--arrangement", default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--phi", type=float, default=0.216)
    ap.add_argument("--shift", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--settings", choices=("production", "tight"), default="production")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "compare":
        cmd_compare(a)
        return
    if a.N is None:
        a.N = 64 if a.case == "bed" else 24
    sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
    from _bootstrap import ensure_flow
    cmd_run(ensure_flow(), a)


if __name__ == "__main__":
    main()
