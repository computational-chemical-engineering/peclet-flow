#!/usr/bin/env python3
"""Cross-validation: the distributed solver (`dcfd`, one GPU) vs the production `pnm_backend` on the
SAME physical problems. The two are independent implementations (production: float, fused-delta Picard;
distributed: double, backward-Euler + projection + multigrid), so we expect agreement to discretisation
accuracy, NOT bit-for-bit. Both run in identical grid units (spacing = 1, rho = 1, mu = nu) to steady
state. Cases:
  (1) Poiseuille channel  -- quantitative, against the analytic parabola too.
  (2) flow around a sphere -- full 3D cut-cell IBM + pressure coupling, fields compared directly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build_mpi")))
import pnm_backend  # noqa: E402
import dcfd  # noqa: E402

NU = 0.1


# dcfd uses arrays indexed [x,y,z] (shape (nx,ny,nz)); pnm_backend uses [z,y,x] (shape (nz,ny,nx)).
# These helpers convert between the two so both end up as u[x,y,z].
def to_pnm(field_xyz):
    return np.ascontiguousarray(np.transpose(field_xyz, (2, 1, 0)))   # [x,y,z] -> (nz,ny,nx)


def from_pnm(field_zyx):
    return np.transpose(np.asarray(field_zyx, np.float64), (2, 1, 0))  # (nz,ny,nx) -> [x,y,z]


def run_pnm(res, sdf_xyz, fx, advection_irrelevant, steps, dt, pmg=False):
    s = pnm_backend.CFDSolver(list(res), [1.0, 1.0, 1.0])
    s.initialize(to_pnm(sdf_xyz).astype(np.float32), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    s.set_rho(1.0)
    s.set_mu(NU)
    s.set_body_force(pnm_backend.float3(fx, 0, 0))
    s.set_theta_(1.0)
    s.set_pressure_solver_params(200)
    s.set_velocity_solver_params(80)
    if pmg:
        s.set_pressure_multigrid_enabled(True)
    prev = 0.0
    for it in range(steps):
        s.step(dt)
        if it % 20 == 0:
            um = float(from_pnm(s.get_u()).mean())
            if it > 40 and abs(um - prev) < 1e-6 * (abs(um) + 1e-12):
                break
            prev = um
    u = from_pnm(s.get_u())  # -> u[x,y,z]
    return float(u.max()), float(u.mean()), u


def run_dcfd(res, sdf_xyz, fx, advection, steps, dt, cutcell=False):
    nx, ny, nz = res
    s = dcfd.Solver(nx, ny, nz, NU, dt)
    s.set_body_force(fx, 0.0, 0.0)
    s.set_advection(advection)
    if advection:
        s.set_outer_iterations(3)  # Picard coupling for the nonlinear advection
    s.set_ibm_solid(sdf_xyz)
    if cutcell:
        s.set_cutcell_pressure_operator(sdf_xyz, galerkin=True)
        s.set_pressure_pcg(True, max_iter=120, rtol=1e-9)
    s.set_velocity_multigrid(True, levels=3, v_cycles=16)
    prev = 0.0
    for it in range(steps):
        s.step(n_diff=0, n_pois=8)
        if s.rank() != 0:
            return None
        um = float(s.get_u().mean())
        if it > 5 and abs(um - prev) < 1e-7 * (abs(um) + 1e-12):
            break
        prev = um
    u = s.get_u()  # already u[x,y,z]
    return float(u.max()), float(u.mean()), u


def compare(label, res, sdf, fx, *, analytic=None, advection=False, steps_pnm=800,
            steps_dcfd=400, dt_pnm=20.0, dt_dcfd=50.0, cutcell=False):
    print(f"--- {label}  (res={res[0]}x{res[1]}x{res[2]}, nu={NU}, fx={fx}) ---")
    umax_p, mean_p, up = run_pnm(res, sdf, fx, advection, steps_pnm, dt_pnm, pmg=cutcell)
    out = run_dcfd(res, sdf, fx, advection, steps_dcfd, dt_dcfd, cutcell=cutcell)
    if out is None:
        return None
    umax_d, mean_d, ud = out
    fluid = np.abs(ud) > 1e-9
    l2 = np.sqrt(np.mean((up[fluid] - ud[fluid]) ** 2)) / (np.max(np.abs(ud)) + 1e-30)
    cl = abs(umax_d - umax_p) / (abs(umax_p) + 1e-30)
    if analytic is not None:
        print(f"  analytic U_max = {analytic:.5f}")
    print(f"  pnm_backend : U_max={umax_p:.5f}  mean={mean_p:.5e}")
    print(f"  dcfd        : U_max={umax_d:.5f}  mean={mean_d:.5e}")
    print(f"  field agreement (relative L2) = {l2*100:.2f}%   centreline diff = {cl*100:.2f}%")
    return l2, cl, umax_p, umax_d


def channel_sdf(res, ylo, yhi):  # -> sdf[x,y,z]
    nx, ny, nz = res
    gy = np.arange(ny, dtype=np.float64)
    sdf = np.empty((nx, ny, nz))
    sdf[:, :, :] = np.minimum(gy - ylo, yhi - gy)[None, :, None]
    return sdf


def sphere_sdf(res, rfrac=0.3):  # -> sdf[x,y,z], negative inside the sphere
    nx, ny, nz = res
    R = nx * rfrac
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return np.sqrt((X - nx / 2.0) ** 2 + (Y - ny / 2.0) ** 2 + (Z - nz / 2.0) ** 2) - R


def main():
    print("=== cross-validation: dcfd (distributed) vs pnm_backend (production) ===")
    results = []
    # (1) Poiseuille channel: walls at non-grid positions -> cut cells
    N = 32
    ylo, yhi = round(0.30 * N) + 0.5, round(0.70 * N) + 0.5
    H = yhi - ylo
    r1 = compare("Poiseuille channel", (N, N, N), channel_sdf((N, N, N), ylo, yhi), 0.01,
                 analytic=(0.01 / (2 * NU)) * (H / 2) ** 2)
    results.append(("channel", r1))
    # (2) flow around a sphere: full 3D Navier-Stokes + cut-cell IBM + pressure coupling, both with
    # nonlinear advection at a moderate steady Reynolds number (~30). Both use Koren TVD advection; the
    # residual difference is float vs double, the production's implicit-FOU deferred correction vs the
    # distributed solver's explicit (Picard-lagged) advection, and the different pressure solvers.
    r2 = compare("flow around a sphere (NS)", (32, 32, 32), sphere_sdf((32, 32, 32)), 2e-4,
                 advection=True, cutcell=True, steps_pnm=800, steps_dcfd=400, dt_pnm=5.0, dt_dcfd=10.0)
    results.append(("sphere", r2))

    ok = True
    print("--- summary ---")
    thr = {"channel": 0.03, "sphere": 0.10}  # exact (advection-free) vs ~few-% (different advection)
    for name, r in results:
        if r is None:
            return
        l2, cl, up, ud = r
        good = l2 < thr[name] and cl < thr[name]
        ok = ok and good
        print(f"  {name:18s}: field L2 {l2*100:5.2f}%  centreline {cl*100:5.2f}%  -> {'OK' if good else 'DIFFER'}")
    print(f"  result: {'PASS' if ok else 'FAIL'}  (distributed solver matches production to discretisation)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
