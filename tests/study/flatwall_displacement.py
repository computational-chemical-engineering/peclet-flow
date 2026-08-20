#!/usr/bin/env python
"""Flat-wall displacement sweep (Frank's isolation ladder, 2026-08-20): the collocated ceiling
hunted in the simplest IBM geometry -- a plane channel whose walls sit at a FRACTIONAL grid offset
s, so every cut cell has the same theta and the closure error is a coherent function of one
parameter instead of an average over random incidences.

Steady-state structure (doc/collocated_accuracy_ceiling.md + the joint-fixed-point argument): the
collocated steady state satisfies  nu*L_ibm(u) + F = G_gp(P)  with  D_alpha(avg_half(u)) = 0.
In a wall-aligned channel these decouple, giving three isolating experiments:

  E1   uniform Fx           -> parabola.  u = u(y) x-only => div == 0 discretely, P == 0: the
                               pressure machinery NEVER ENGAGES.  Any error is the momentum IBM
                               closure L_ibm alone (suspect S4).  (Caveat: 2nd-order closures can
                               be exact on parabolas -- a null here does not clear curved walls.)
  E1b  Fx(y)=F0 cos(k(y-yc)), k=pi/W -> u = F0/(mu k^2) cos(k(y-yc)), zero at both walls,
                               non-polynomial: the order of L_ibm is readable.
  E2   E1's Fx  PLUS  Fy(y)=A sin(2pi(y-w_lo)/W) in the fluid -> exact answer: u IDENTICAL to E1,
                               v == 0, p = integral(Fy).  The y-force must be absorbed ENTIRELY by
                               the discrete pressure.  Any (u - u_E1, v != 0) response is a defect
                               of the (G_gp vs masked-half-average constraint) pair -- the
                               non-adjointness channel, with L_ibm differenced out.

Sweep s in [0,1) x N (channel width in cells) x solver in {stag, gauge-exact (mode 9), plain
(mode 0)}.  Errors are reported on the conserved face flux (primary; mean(alpha_x*uf)) and the
cell mean, both relative to the exact discharge.

  SDFLOW_BUILD=build_omp3 python tests/study/flatwall_displacement.py            # default sweep
  ... flatwall_displacement.py --N 8,16,32,64 --s 0.1,0.3,0.5,0.7,0.9 --exp E1,E1b,E2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow  # noqa: E402

MU, F0 = 0.1, 1e-3
NX = NZ = 8          # periodic; the solution is x/z-invariant
SLAB = 8             # total solid thickness in y (walls wrap periodically through it)
TOL = float(os.environ.get("MARCH_TOL", "1e-9"))
CAP = int(os.environ.get("MARCH_MAX", "8000"))
DT = float(os.environ.get("DT", "60.0"))


def channel(N, s):
    """SDF (F-order (nx,ny,nz)) + wall positions for a width-N channel displaced by s cells."""
    ny = N + SLAB
    w_lo = SLAB / 2 + s
    w_hi = w_lo + N
    yc = np.arange(ny) + 0.5
    sd = np.minimum(yc - w_lo, w_hi - yc)         # planar, exact
    sdf = np.asfortranarray(np.broadcast_to(sd[None, :, None], (NX, ny, NZ)).copy())
    return sdf, ny, w_lo, w_hi


def exact_profile(exp, y, w_lo, w_hi):
    W = w_hi - w_lo
    yc = 0.5 * (w_lo + w_hi)
    if exp == "E1b":
        k = np.pi / W
        u = F0 / (MU * k * k) * np.cos(k * (y - yc))
    else:
        u = F0 / (2 * MU) * ((W / 2) ** 2 - (y - yc) ** 2)
    return np.where((y > w_lo) & (y < w_hi), u, 0.0)


def exact_discharge(exp, w_lo, w_hi):
    W = w_hi - w_lo
    if exp == "E1b":
        k = np.pi / W
        return F0 / (MU * k * k) * (2.0 / k)      # integral of cos over [-pi/2..pi/2]/k
    return F0 * W ** 3 / (12 * MU)


def make_solver(kind, N, s):
    sdf, ny, w_lo, w_hi = channel(N, s)
    sv = flow.Solver if kind == "stag" else flow.SolverColocated
    sol = sv(NX, ny, NZ)
    sol.set_rho(1.0); sol.set_mu(MU); sol.set_dt(DT)
    sol.set_advection(False)
    sol.set_velocity_solver_params(200)
    sol.set_pressure_multigrid(True, 3)
    sol.set_pressure_pcg(True, 200, 1e-10)
    if kind != "stag":
        if hasattr(sol, "set_collocated_scheme"):
            sol.set_collocated_scheme(kind)
        else:
            sol.set_face_interp({"gauge-exact": 9, "plain": 0}[kind])
    sol.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    return sol, sdf, ny, w_lo, w_hi


def set_forces(sol, exp, ny, w_lo, w_hi):
    y = np.arange(ny) + 0.5
    W = w_hi - w_lo
    yc = 0.5 * (w_lo + w_hi)
    fluid = (y > w_lo) & (y < w_hi)
    if exp == "E1":
        sol.set_body_force(F0, 0.0, 0.0)
        return
    sol.enable_cell_force()
    if exp == "E1b":
        k = np.pi / W
        fx = np.where(fluid, F0 * np.cos(k * (y - yc)), 0.0)
        fy = np.zeros(ny)
    else:  # E2: E1's uniform Fx + a fluid-only sinusoidal Fy (exactly hydrostatic)
        fx = np.where(fluid, F0, 0.0)
        fy = np.where(fluid, 10 * F0 * np.sin(2 * np.pi * (y - w_lo) / W), 0.0)
    zero = np.zeros((NX, ny, NZ), order="F")
    sol.set_field("force_x", np.asfortranarray(np.broadcast_to(fx[None, :, None],
                                                               (NX, ny, NZ)).copy()))
    sol.set_field("force_y", np.asfortranarray(np.broadcast_to(fy[None, :, None],
                                                               (NX, ny, NZ)).copy()))
    sol.set_field("force_z", zero)


def march(sol):
    prev = 0.0
    for it in range(CAP):
        sol.step()
        if it % 10 == 9:
            um = float(np.asarray(sol.get_u()).mean())
            if it > 20 and abs(um - prev) < TOL * (abs(um) + 1e-300):
                return it + 1
            prev = um
    return CAP


def run(kind, exp, N, s):
    sol, sdf, ny, w_lo, w_hi = make_solver(kind, N, s)
    set_forces(sol, exp, ny, w_lo, w_hi)
    steps = march(sol)
    U = np.asarray(sol.get_u())
    V = np.asarray(sol.get_v())
    if kind == "stag":
        UF, ax = U, np.asarray(sol.get_ox())
    else:
        UF, ax = np.asarray(sol.get_uf()), np.asarray(sol.get_ox())
    Qex = exact_discharge(exp, w_lo, w_hi)
    q_flux = float((ax * UF).mean()) * ny         # conserved-face-flux discharge
    q_cell = float(U.mean()) * ny
    y = np.arange(ny) + 0.5
    uex = exact_profile(exp, y, w_lo, w_hi)
    prof = U.mean(axis=(0, 2))
    inner = (y > w_lo + 1.5) & (y < w_hi - 1.5)   # interior cells: no cut-cell weighting question
    l2 = float(np.sqrt(np.mean((prof[inner] - uex[inner]) ** 2)) / np.sqrt(np.mean(uex[inner]**2)))
    return dict(steps=steps, e_flux=q_flux / Qex - 1, e_cell=q_cell / Qex - 1, l2=l2,
                U=U, V=V, uscale=float(np.abs(uex).max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default="8,16,32")
    ap.add_argument("--s", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--exp", default="E1,E1b,E2")
    ap.add_argument("--solvers", default="stag,gauge-exact,plain")
    a = ap.parse_args()
    Ns = [int(x) for x in a.N.split(",")]
    Ss = [float(x) for x in a.s.split(",")]
    exps = a.exp.split(",")
    kinds = a.solvers.split(",")
    for exp in exps:
        print(f"\n================ {exp} ================")
        hdr = f"{'N':>4} {'s':>5} {'solver':>12} {'steps':>6} {'e_flux':>11} {'e_cell':>11} {'L2prof':>10}"
        if exp == "E2":
            hdr += f" {'sp_du':>10} {'sp_v':>10}"
        print(hdr, flush=True)
        for N in Ns:
            for s in Ss:
                base = {}
                for kind in kinds:
                    r = run(kind, exp, N, s)
                    row = (f"{N:>4} {s:>5.2f} {kind:>12} {r['steps']:>6} "
                           f"{r['e_flux']:>+11.3e} {r['e_cell']:>+11.3e} {r['l2']:>10.3e}")
                    if exp == "E2":
                        # spurious response vs this solver's own E1 (hydrostatic leakage)
                        r1 = base.get(kind)
                        if r1 is None:
                            r1 = run(kind, "E1", N, s)
                        du = float(np.abs(r['U'] - r1['U']).max()) / r['uscale']
                        dv = float(np.abs(r['V']).max()) / r['uscale']
                        row += f" {du:>10.3e} {dv:>10.3e}"
                    print(row, flush=True)
