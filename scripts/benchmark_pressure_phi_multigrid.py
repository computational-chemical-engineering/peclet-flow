import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import pnm_backend


def make_sphere_sdf(nx, ny, nz, radius):
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    sdf_xyz = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) - radius
    return np.transpose(sdf_xyz.astype(np.float32), (2, 1, 0))


def run_case(enable_mg, steps, pressure_iter, res_n, radius):
    shape = [res_n, res_n, res_n]
    spacing = [1.0, 1.0, 1.0]
    solver = pnm_backend.CFDSolver(shape, spacing)
    solver.initialize(make_sphere_sdf(res_n, res_n, res_n, radius),
                      [0.0, 0.0, 0.0], spacing)
    solver.set_body_force(pnm_backend.float3(0.001, 0.0, 0.0))
    solver.set_rho(1.0)
    solver.set_mu(0.5)
    solver.set_pressure_solver_params(iter=pressure_iter)
    solver.set_pressure_multigrid_enabled(enable_mg)
    if enable_mg:
        solver.set_pressure_multigrid_params(4, 2, 2, 32, 2)

    start = time.perf_counter()
    for _ in range(steps):
        solver.step(0.01)
    elapsed = time.perf_counter() - start

    u_mean = float(np.mean(solver.get_u()))
    p_mean = float(np.mean(solver.get_p()))
    return elapsed, u_mean, p_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--pressure-iter", type=int, default=1000)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--radius", type=float, default=10.0)
    args = parser.parse_args()

    base_elapsed, base_u, base_p = run_case(False, args.steps,
                                            args.pressure_iter, args.res,
                                            args.radius)
    mg_elapsed, mg_u, mg_p = run_case(True, args.steps, args.pressure_iter,
                                      args.res, args.radius)

    print(
        f"baseline steps={args.steps} res={args.res} pressure_iter={args.pressure_iter} "
        f"elapsed_s={base_elapsed:.6f} per_step_s={base_elapsed/args.steps:.6f} "
        f"u_mean={base_u:.6e} p_mean={base_p:.6e}"
    )
    print(
        f"multigrid steps={args.steps} res={args.res} pressure_iter={args.pressure_iter} "
        f"elapsed_s={mg_elapsed:.6f} per_step_s={mg_elapsed/args.steps:.6f} "
        f"u_mean={mg_u:.6e} p_mean={mg_p:.6e}"
    )
    print(f"speedup={base_elapsed / mg_elapsed:.3f}x")


if __name__ == "__main__":
    main()
