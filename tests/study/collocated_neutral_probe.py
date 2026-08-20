#!/usr/bin/env python
"""Neutral-mode probe: is the collocated march's stalled state an ATTRACTOR or a FROZEN defect?

The steady Stokes step map is linear with a unique true fixed point (phi = 0).  The measured
stalled states (|uf-halfavg(u)| ~ 4e-2 <u>, k off by the plateau) can only be exactly stationary
if the map has (near-)neutral modes; then the reached state depends on the initial condition and
on any perturbation applied along the way.  Test: march from IC-A (zero velocity, the protocol
IC), then (1) perturb the converged state with a small random solenoidal-ish kick and re-march,
(2) march independently from IC-B (a bulk plug profile).  Attractor => all k agree to march
noise.  Neutral modes => they differ at the plateau scale.

  SDFLOW_BUILD=build_omp3 BED=...npz N=96 python collocated_neutral_probe.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow  # noqa: E402

BED = os.environ["BED"]
N = int(os.environ.get("N", "96"))
TOL = float(os.environ.get("MARCH_TOL", "1e-8"))
CAP = int(os.environ.get("MARCH_MAX", "6000"))
DT = float(os.environ.get("DT", "60.0"))
KIND = os.environ.get("KIND", "gauge-exact")
MU, F0 = 0.1, 1e-3


def bed_sdf(N, npz):
    pk = np.load(npz)
    box = np.asarray(pk["box"], float)
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


sdf, R = bed_sdf(N, BED)
fluid = sdf >= 0.0


def make():
    s = (flow.Solver if KIND == "stag" else flow.SolverColocated)(N, N, N)
    s.set_rho(1.0); s.set_mu(MU); s.set_dt(DT)
    s.set_body_force(F0, 0, 0); s.set_advection(False)
    s.set_velocity_solver_params(150)
    s.set_pressure_multigrid(True, max(2, int(np.log2(N)) - 2))
    s.set_pressure_pcg(True, 300, 1e-8)
    if KIND != "stag":
        if hasattr(s, "set_collocated_scheme"):
            s.set_collocated_scheme(KIND)
        else:
            s.set_face_interp({"gauge-exact": 9, "plain": 0}[KIND])
    s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
    return s


def march(s, cap=CAP):
    prev = 0.0
    for it in range(cap):
        s.step()
        if it % 10 == 9:
            um = float(np.asarray(s.get_u()).mean())
            if it > 20 and abs(um - prev) < TOL * (abs(um) + 1e-300):
                return it + 1
            prev = um
    return cap


def diag(s, tag, steps):
    U = [np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w())]
    us = float(np.abs(U[0][fluid]).mean()) + 1e-300
    kc = float(U[0].mean()) * MU / F0 / R ** 2
    m1 = 0.0
    if KIND != "stag":
        UF = [np.asarray(s.get_uf()), np.asarray(s.get_vf()), np.asarray(s.get_wf())]
        OX = [np.asarray(s.get_ox()), np.asarray(s.get_oy()), np.asarray(s.get_oz())]
        m1sq = cnt = 0.0
        for a in range(3):
            half = 0.5 * (U[a] + np.roll(U[a], 1, axis=a))
            op = OX[a] > 0
            m1sq += float(((UF[a] - half)[op] ** 2).sum()); cnt += int(op.sum())
        m1 = np.sqrt(m1sq / cnt) / us
    print(f"[{tag}] steps={steps}  k_cell/R2={kc:.7e}  m1={m1:.3e}", flush=True)
    return kc, U


print(f"# bed {os.path.basename(BED)} N={N} R={R:.1f} kind={KIND} dt={DT} tol={TOL}", flush=True)

# A: protocol IC (zero velocity)
sA = make()
stA = march(sA)
kA, UA = diag(sA, "A  zero-IC       ", stA)

# A': perturb the converged state and re-march
rng = np.random.default_rng(7)
pert = [np.asfortranarray(np.where(fluid, 0.1 * np.abs(UA[0][fluid]).mean()
                                   * rng.standard_normal(UA[a].shape), 0.0))
        for a in range(3)]
sA.set_state(np.asfortranarray(UA[0] + pert[0]), np.asfortranarray(UA[1] + pert[1]),
             np.asfortranarray(UA[2] + pert[2]))
stP = march(sA)
kP, _ = diag(sA, "A' perturbed     ", stP)

# B: plug-profile IC (mean-flow magnitude, x only, fluid cells)
sB = make()
plug = np.asfortranarray(np.where(fluid, float(np.abs(UA[0][fluid]).mean()), 0.0))
z = np.zeros_like(plug)
sB.set_state(plug, z.copy(), z.copy())
stB = march(sB)
kB, _ = diag(sB, "B  plug-IC       ", stB)

print(f"\nrel spread: (A'-A)/A = {(kP - kA) / kA:+.3e}   (B-A)/A = {(kB - kA) / kA:+.3e}",
      flush=True)
print("attractor => spreads ~ march noise (<1e-6); neutral/frozen modes => plateau-scale (1e-3)")
