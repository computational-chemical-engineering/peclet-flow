import argparse
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import pnm_backend


ZICK_HOMSY_PHIS = np.array(
    [0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236],
    dtype=np.float64,
)
ZICK_HOMSY_K = np.array(
    [1.096, 1.212, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1],
    dtype=np.float64,
)


def generate_sc_sdf_zyx(phi, res_n, length=1.0):
    dx = length / res_n
    x = np.linspace(0.0, length, res_n, endpoint=False) + 0.5 * dx
    y = np.linspace(0.0, length, res_n, endpoint=False) + 0.5 * dx
    z = np.linspace(0.0, length, res_n, endpoint=False) + 0.5 * dx

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    radius = (phi * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * length

    xc = 0.5 * length
    yc = 0.5 * length
    zc = 0.5 * length
    sdf_xyz = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2) - radius

    # The Python binding expects a dense array shaped (nz, ny, nx).
    return np.transpose(sdf_xyz.astype(np.float32), (2, 1, 0)), radius, [dx, dx, dx]


def get_reference_k(phi):
    if phi < ZICK_HOMSY_PHIS[0]:
        return 1.0
    return float(np.interp(phi, ZICK_HOMSY_PHIS, ZICK_HOMSY_K))


def build_solver(phi, res_n, enable_pressure_mg, pressure_mg_params,
                 enable_velocity_mg, velocity_mg_params, pressure_iter,
                 velocity_iter, outer_iterations, outer_tolerance,
                 outer_mode):
    sdf_zyx, radius, spacing = generate_sc_sdf_zyx(phi, res_n)
    solver = pnm_backend.CFDSolver([res_n, res_n, res_n], spacing)
    solver.initialize(sdf_zyx, [0.0, 0.0, 0.0], spacing)

    solver.set_body_force(pnm_backend.float3(1.0, 0.0, 0.0))
    solver.set_rho(0.0)
    solver.set_mu(1.0)
    solver.set_pressure_solver_params(iter=pressure_iter)
    solver.set_velocity_solver_params(iter=velocity_iter)
    solver.set_outer_iterations(outer_iterations)
    solver.set_outer_tolerance(outer_tolerance)
    solver.set_outer_convergence_mode(outer_mode)

    solver.set_pressure_multigrid_enabled(enable_pressure_mg)
    if enable_pressure_mg:
        solver.set_pressure_multigrid_params(*pressure_mg_params)
    solver.set_velocity_multigrid_enabled(enable_velocity_mg)
    if enable_velocity_mg:
        solver.set_velocity_multigrid_params(*velocity_mg_params)

    return solver, radius


def warm_up_cuda():
    solver, _ = build_solver(
        phi=0.001,
        res_n=16,
        enable_pressure_mg=False,
        pressure_mg_params=(2, 1, 1, 4, 1),
        enable_velocity_mg=False,
        velocity_mg_params=(2, 1, 1, 4, 1),
        pressure_iter=2,
        velocity_iter=1,
        outer_iterations=2,
        outer_tolerance=0.0,
        outer_mode=0,
    )
    solver.step(0.1)
    np.array(solver.get_u(), copy=False).mean()


def compute_drag_factor(radius, u_superficial, force=1.0, mu=1.0):
    return force / (6.0 * math.pi * mu * radius * u_superficial)


def run_case(phi, res_n, enable_pressure_mg, pressure_mg_params,
             enable_velocity_mg, velocity_mg_params, pressure_iter,
             velocity_iter, outer_iterations, outer_tolerance, outer_mode,
             dt, max_steps,
             convergence_stride, convergence_rel_tol):
    solver, radius = build_solver(
        phi,
        res_n,
        enable_pressure_mg,
        pressure_mg_params,
        enable_velocity_mg,
        velocity_mg_params,
        pressure_iter,
        velocity_iter,
        outer_iterations,
        outer_tolerance,
        outer_mode,
    )

    u_mean_history = []
    outer_iterations_used = []
    steps_taken = 0
    start = time.perf_counter()

    for step_idx in range(max_steps):
        solver.step(dt)
        steps_taken = step_idx + 1
        outer_iterations_used.append(int(solver.get_last_outer_iterations()))

        if step_idx % convergence_stride != 0:
            continue

        u_mean = float(np.mean(np.array(solver.get_u(), copy=False)))
        p_mean = float(np.mean(np.array(solver.get_p(), copy=False)))
        if not np.isfinite(u_mean) or not np.isfinite(p_mean):
            raise RuntimeError(
                f"non-finite state detected at step {steps_taken}: "
                f"u_mean={u_mean}, p_mean={p_mean}"
            )

        u_mean_history.append(u_mean)
        if len(u_mean_history) < 2:
            continue
        if abs(u_mean_history[-1]) <= 1e-14:
            continue

        rel_change = abs(u_mean_history[-1] - u_mean_history[-2]) / abs(u_mean_history[-1])
        if rel_change < convergence_rel_tol:
            break

    elapsed = time.perf_counter() - start
    u_superficial = float(np.mean(np.array(solver.get_u(), copy=False)))
    if not np.isfinite(u_superficial) or abs(u_superficial) <= 1e-14:
        raise RuntimeError(f"invalid superficial velocity: {u_superficial}")

    k_sim = compute_drag_factor(radius, u_superficial)
    k_ref = get_reference_k(phi)
    ref_rel_error = abs(k_sim - k_ref) / k_ref

    return {
        "phi": phi,
        "resolution": res_n,
        "method": (
            "pressure+velocity-multigrid"
            if (enable_pressure_mg and enable_velocity_mg)
            else "pressure-multigrid"
            if enable_pressure_mg
            else "rbgs"
        ),
        "steps_taken": steps_taken,
        "elapsed_s": elapsed,
        "per_step_s": elapsed / max(steps_taken, 1),
        "u_superficial": u_superficial,
        "k_sim": k_sim,
        "k_ref": k_ref,
        "ref_rel_error": ref_rel_error,
        "outer_iterations_mean": float(np.mean(outer_iterations_used)),
        "outer_iterations_max": int(np.max(outer_iterations_used)),
    }


def validate_pair(phi, args):
    rbgs = run_case(
        phi,
        args.res,
        False,
        args.pressure_mg_params,
        False,
        args.velocity_mg_params,
        args.pressure_iter,
        args.velocity_iter,
        args.outer_iterations,
        args.outer_tol,
        args.outer_mode,
        args.dt,
        args.max_steps,
        args.convergence_stride,
        args.convergence_rel_tol,
    )
    mg = run_case(
        phi,
        args.res,
        True,
        args.pressure_mg_params,
        args.enable_velocity_mg,
        args.velocity_mg_params,
        args.pressure_iter,
        args.velocity_iter,
        args.outer_iterations,
        args.outer_tol,
        args.outer_mode,
        args.dt,
        args.max_steps,
        args.convergence_stride,
        args.convergence_rel_tol,
    )

    reproduce_rel_error = abs(mg["k_sim"] - rbgs["k_sim"]) / abs(rbgs["k_sim"])
    return {
        "phi": phi,
        "k_ref": rbgs["k_ref"],
        "rbgs": rbgs,
        "mg": mg,
        "reproduce_rel_error": reproduce_rel_error,
        "reference_pass_rbgs": rbgs["ref_rel_error"] <= args.reference_rel_tol,
        "reference_pass_mg": mg["ref_rel_error"] <= args.reference_rel_tol,
        "reproduce_pass": reproduce_rel_error <= args.reproduce_rel_tol,
        "speedup": rbgs["elapsed_s"] / mg["elapsed_s"],
    }


def print_summary(results, args):
    print(
        f"Validation setup: N={args.res}, dt={args.dt}, max_steps={args.max_steps}, "
        f"pressure_iter={args.pressure_iter}, velocity_iter={args.velocity_iter}, "
        f"outer_iterations={args.outer_iterations}, outer_tol={args.outer_tol}, "
        f"outer_mode={args.outer_mode}"
    )
    print(
        f"Pressure MG params: levels={args.mg_levels}, pre={args.mg_pre}, "
        f"post={args.mg_post}, bottom={args.mg_bottom}, cycles={args.mg_cycles}"
    )
    print(
        f"Velocity MG: enabled={args.enable_velocity_mg}, levels={args.velocity_mg_levels}, "
        f"pre={args.velocity_mg_pre}, post={args.velocity_mg_post}, "
        f"bottom={args.velocity_mg_bottom}, cycles={args.velocity_mg_cycles}"
    )
    print(
        f"Tolerances: reference_rel_tol={args.reference_rel_tol:.4f}, "
        f"reproduce_rel_tol={args.reproduce_rel_tol:.4f}"
    )
    print()
    print(
        f"{'phi':>8} {'K_ref':>10} {'K_rbgs':>10} {'K_mg':>10} "
        f"{'err_rbgs%':>10} {'err_mg%':>10} {'mg-rbgs%':>10} "
        f"{'t_rbgs(s)':>10} {'t_mg(s)':>10} {'speedup':>8} "
        f"{'outer_rbgs':>10} {'outer_mg':>10}"
    )
    print("-" * 110)

    for res in results:
        print(
            f"{res['phi']:8.4f} {res['k_ref']:10.4f} "
            f"{res['rbgs']['k_sim']:10.4f} {res['mg']['k_sim']:10.4f} "
            f"{100.0 * res['rbgs']['ref_rel_error']:10.2f} "
            f"{100.0 * res['mg']['ref_rel_error']:10.2f} "
            f"{100.0 * res['reproduce_rel_error']:10.3f} "
            f"{res['rbgs']['elapsed_s']:10.3f} {res['mg']['elapsed_s']:10.3f} "
            f"{res['speedup']:8.3f} "
            f"{res['rbgs']['outer_iterations_mean']:10.2f} "
            f"{res['mg']['outer_iterations_mean']:10.2f}"
        )

    print()
    print(f"RBGS within reference tolerance: {all(r['reference_pass_rbgs'] for r in results)}")
    print(f"MG within reference tolerance:   {all(r['reference_pass_mg'] for r in results)}")
    print(f"MG reproduces RBGS:              {all(r['reproduce_pass'] for r in results)}")


def write_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "phi",
            "resolution",
            "k_ref",
            "k_rbgs",
            "k_mg",
            "rbgs_ref_rel_error",
            "mg_ref_rel_error",
            "mg_vs_rbgs_rel_error",
            "rbgs_elapsed_s",
            "mg_elapsed_s",
            "speedup",
            "rbgs_steps",
            "mg_steps",
            "rbgs_outer_iterations_mean",
            "mg_outer_iterations_mean",
            "rbgs_outer_iterations_max",
            "mg_outer_iterations_max",
        ])
        for res in results:
            writer.writerow([
                res["phi"],
                res["rbgs"]["resolution"],
                res["k_ref"],
                res["rbgs"]["k_sim"],
                res["mg"]["k_sim"],
                res["rbgs"]["ref_rel_error"],
                res["mg"]["ref_rel_error"],
                res["reproduce_rel_error"],
                res["rbgs"]["elapsed_s"],
                res["mg"]["elapsed_s"],
                res["speedup"],
                res["rbgs"]["steps_taken"],
                res["mg"]["steps_taken"],
                res["rbgs"]["outer_iterations_mean"],
                res["mg"]["outer_iterations_mean"],
                res["rbgs"]["outer_iterations_max"],
                res["mg"]["outer_iterations_max"],
            ])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the N=128 periodic-sphere drag case with RBGS and pressure multigrid."
    )
    parser.add_argument(
        "--phis",
        type=float,
        nargs="+",
        default=[0.001],
        help="Solid fractions from the Zick and Homsy table. Default keeps the known stable N=128 case.",
    )
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--convergence-stride", type=int, default=10)
    parser.add_argument("--convergence-rel-tol", type=float, default=1e-8)
    parser.add_argument("--pressure-iter", type=int, default=50)
    parser.add_argument("--velocity-iter", type=int, default=2)
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--outer-tol", type=float, default=0.0)
    parser.add_argument(
        "--outer-mode",
        type=int,
        default=0,
        help="0 = absolute max correction over all cells, 1 = RMS correction over active velocity DOFs",
    )
    parser.add_argument("--reference-rel-tol", type=float, default=0.02)
    parser.add_argument("--reproduce-rel-tol", type=float, default=0.01)
    parser.add_argument("--mg-levels", type=int, default=4)
    parser.add_argument("--mg-pre", type=int, default=2)
    parser.add_argument("--mg-post", type=int, default=2)
    parser.add_argument("--mg-bottom", type=int, default=32)
    parser.add_argument("--mg-cycles", type=int, default=2)
    parser.add_argument("--enable-velocity-mg", action="store_true")
    parser.add_argument("--velocity-mg-levels", type=int, default=4)
    parser.add_argument("--velocity-mg-pre", type=int, default=1)
    parser.add_argument("--velocity-mg-post", type=int, default=1)
    parser.add_argument("--velocity-mg-bottom", type=int, default=16)
    parser.add_argument("--velocity-mg-cycles", type=int, default=1)
    parser.add_argument(
        "--output-csv",
        default="output/periodic_spheres_n128_rbgs_vs_mg.csv",
        help="CSV path for the summary table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.pressure_mg_params = (
        args.mg_levels,
        args.mg_pre,
        args.mg_post,
        args.mg_bottom,
        args.mg_cycles,
    )
    args.velocity_mg_params = (
        args.velocity_mg_levels,
        args.velocity_mg_pre,
        args.velocity_mg_post,
        args.velocity_mg_bottom,
        args.velocity_mg_cycles,
    )

    warm_up_cuda()
    results = [validate_pair(phi, args) for phi in args.phis]
    print_summary(results, args)
    write_csv(results, args.output_csv)
    print(f"\nWrote summary CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
