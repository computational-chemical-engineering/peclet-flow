import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import pnm_backend


METHODS = {
    "rbgs": {
        "enable_pressure_mg": False,
        "enable_velocity_mg": False,
    },
    "pmg": {
        "enable_pressure_mg": True,
        "enable_velocity_mg": False,
    },
    "fullmg": {
        "enable_pressure_mg": True,
        "enable_velocity_mg": True,
    },
}


def resolve_default_sdf():
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../ring_packing_sdf.vti")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/packing_ring.vti")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No default packing SDF found. Looked for ring_packing_sdf.vti and data/packing_ring.vti."
    )


def load_sdf(sdf_path):
    sdf_zyx, origin, spacing = pnm_backend.SDFReader.read_vti(sdf_path)
    return np.asarray(sdf_zyx, dtype=np.float32), list(origin), list(spacing)


def build_solver(args, sdf_zyx, origin, spacing, method_name):
    solver = pnm_backend.CFDSolver(list(sdf_zyx.shape), spacing)
    solver.initialize(sdf_zyx, origin, spacing)
    solver.set_body_force(pnm_backend.float3(args.force_x, args.force_y, args.force_z))
    solver.set_rho(args.rho)
    solver.set_mu(args.mu)
    solver.set_ibm_scheme(args.ibm_scheme)
    solver.set_pressure_solver_params(iter=args.pressure_iter)
    solver.set_velocity_solver_params(iter=args.velocity_iter)
    solver.set_outer_iterations(args.outer_iterations)
    solver.set_outer_tolerance(args.outer_tol)
    solver.set_outer_convergence_mode(args.outer_mode)

    method = METHODS[method_name]
    solver.set_pressure_multigrid_enabled(method["enable_pressure_mg"])
    if method["enable_pressure_mg"]:
        solver.set_pressure_multigrid_params(
            args.pressure_mg_levels,
            args.pressure_mg_pre,
            args.pressure_mg_post,
            args.pressure_mg_bottom,
            args.pressure_mg_cycles,
        )
    solver.set_velocity_multigrid_enabled(method["enable_velocity_mg"])
    if method["enable_velocity_mg"]:
        solver.set_velocity_multigrid_params(
            args.velocity_mg_levels,
            args.velocity_mg_pre,
            args.velocity_mg_post,
            args.velocity_mg_bottom,
            args.velocity_mg_cycles,
        )
    return solver


def run_case(args, sdf_zyx, origin, spacing, method_name):
    solver = build_solver(args, sdf_zyx, origin, spacing, method_name)
    u_mean_history = []
    outer_iterations_used = []
    steps_taken = 0

    start = time.perf_counter()
    for step_idx in range(args.max_steps):
        solver.step(args.dt)
        steps_taken = step_idx + 1
        outer_iterations_used.append(int(solver.get_last_outer_iterations()))

        if step_idx % args.convergence_stride != 0:
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
        denom = abs(u_mean_history[-1]) + 1e-15
        rel_change = abs(u_mean_history[-1] - u_mean_history[-2]) / denom
        if rel_change < args.convergence_rel_tol:
            break

    elapsed = time.perf_counter() - start
    u = np.array(solver.get_u(), copy=False)
    v = np.array(solver.get_v(), copy=False)
    w = np.array(solver.get_w(), copy=False)
    p = np.array(solver.get_p(), copy=False)

    return {
        "method": method_name,
        "elapsed_s": elapsed,
        "per_step_s": elapsed / max(steps_taken, 1),
        "steps_taken": steps_taken,
        "outer_iterations_mean": float(np.mean(outer_iterations_used)),
        "outer_iterations_max": int(np.max(outer_iterations_used)),
        "u_mean": float(np.mean(u)),
        "v_mean": float(np.mean(v)),
        "w_mean": float(np.mean(w)),
        "p_mean": float(np.mean(p)),
        "u_abs_max": float(np.max(np.abs(u))),
        "p_abs_max": float(np.max(np.abs(p))),
    }


def warm_up_cuda():
    shape = [16, 16, 16]
    spacing = [1.0, 1.0, 1.0]
    x = np.arange(shape[2], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    z = np.arange(shape[0], dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    sdf_xyz = np.sqrt((X - 8.0) ** 2 + (Y - 8.0) ** 2 + (Z - 8.0) ** 2) - 4.0
    sdf_zyx = np.transpose(sdf_xyz, (2, 1, 0)).astype(np.float32)

    solver = pnm_backend.CFDSolver(shape, spacing)
    solver.initialize(sdf_zyx, [0.0, 0.0, 0.0], spacing)
    solver.set_body_force(pnm_backend.float3(1.0, 0.0, 0.0))
    solver.set_rho(0.0)
    solver.set_mu(1.0)
    solver.set_pressure_solver_params(iter=2)
    solver.set_velocity_solver_params(iter=1)
    solver.set_outer_iterations(1)
    solver.set_outer_tolerance(0.0)
    solver.step(0.1)
    np.mean(np.array(solver.get_u(), copy=False))


def write_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "method",
        "status",
        "elapsed_s",
        "per_step_s",
        "steps_taken",
        "outer_iterations_mean",
        "outer_iterations_max",
        "u_mean",
        "v_mean",
        "w_mean",
        "p_mean",
        "u_abs_max",
        "p_abs_max",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def print_summary(rows, args):
    print(f"Packing SDF: {args.sdf}")
    print(
        f"dt={args.dt}, max_steps={args.max_steps}, convergence_stride={args.convergence_stride}, "
        f"convergence_rel_tol={args.convergence_rel_tol}"
    )
    print(
        f"pressure_iter={args.pressure_iter}, velocity_iter={args.velocity_iter}, "
        f"outer_iterations={args.outer_iterations}, outer_tol={args.outer_tol}, outer_mode={args.outer_mode}"
    )
    print(
        f"pressure_mg=({args.pressure_mg_levels}, {args.pressure_mg_pre}, {args.pressure_mg_post}, "
        f"{args.pressure_mg_bottom}, {args.pressure_mg_cycles})"
    )
    print(
        f"velocity_mg=({args.velocity_mg_levels}, {args.velocity_mg_pre}, {args.velocity_mg_post}, "
        f"{args.velocity_mg_bottom}, {args.velocity_mg_cycles})"
    )
    print()
    print(
        f"{'method':<10} {'time(s)':>10} {'steps':>8} {'outer_mean':>12} {'outer_max':>10} "
        f"{'u_mean':>12} {'|u|max':>12} {'|p|max':>12} {'status':<16}"
    )
    print("-" * 110)
    for row in rows:
        print(
            f"{row['method']:<10} {row['elapsed_s']:10.3f} {row['steps_taken']:8d} "
            f"{row['outer_iterations_mean']:12.2f} {row['outer_iterations_max']:10d} "
            f"{row['u_mean']:12.4e} {row['u_abs_max']:12.4e} {row['p_abs_max']:12.4e} "
            f"{row['status']:<16}"
        )

    rbgs = next((row for row in rows if row["method"] == "rbgs" and row["status"] == "ok"), None)
    if rbgs is not None:
        for row in rows:
            if row["method"] == "rbgs" or row["status"] != "ok":
                continue
            print(f"speedup_{row['method']}={rbgs['elapsed_s'] / row['elapsed_s']:.3f}x")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark RBGS and multigrid performance on a real SDF packing geometry."
    )
    parser.add_argument("--sdf", default=resolve_default_sdf())
    parser.add_argument("--methods", nargs="+", default=["rbgs", "pmg", "fullmg"])
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--convergence-stride", type=int, default=10)
    parser.add_argument("--convergence-rel-tol", type=float, default=1e-8)
    parser.add_argument("--force-x", type=float, default=1.0)
    parser.add_argument("--force-y", type=float, default=0.0)
    parser.add_argument("--force-z", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--ibm-scheme", type=int, default=0)
    parser.add_argument("--pressure-iter", type=int, default=50)
    parser.add_argument("--velocity-iter", type=int, default=2)
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--outer-tol", type=float, default=1e-6)
    parser.add_argument("--outer-mode", type=int, default=1)
    parser.add_argument("--pressure-mg-levels", type=int, default=4)
    parser.add_argument("--pressure-mg-pre", type=int, default=1)
    parser.add_argument("--pressure-mg-post", type=int, default=1)
    parser.add_argument("--pressure-mg-bottom", type=int, default=16)
    parser.add_argument("--pressure-mg-cycles", type=int, default=1)
    parser.add_argument("--velocity-mg-levels", type=int, default=4)
    parser.add_argument("--velocity-mg-pre", type=int, default=2)
    parser.add_argument("--velocity-mg-post", type=int, default=2)
    parser.add_argument("--velocity-mg-bottom", type=int, default=32)
    parser.add_argument("--velocity-mg-cycles", type=int, default=1)
    parser.add_argument(
        "--output-csv",
        default="output/packing_multigrid_benchmark.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for method in args.methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Expected one of {sorted(METHODS)}.")

    sdf_zyx, origin, spacing = load_sdf(args.sdf)
    warm_up_cuda()

    rows = []
    for method in args.methods:
        print(f"Running method={method}")
        try:
            row = run_case(args, sdf_zyx, origin, spacing, method)
            row["status"] = "ok"
        except Exception as exc:
            row = {
                "method": method,
                "status": f"failed: {exc}",
                "elapsed_s": float("nan"),
                "per_step_s": float("nan"),
                "steps_taken": 0,
                "outer_iterations_mean": float("nan"),
                "outer_iterations_max": 0,
                "u_mean": float("nan"),
                "v_mean": float("nan"),
                "w_mean": float("nan"),
                "p_mean": float("nan"),
                "u_abs_max": float("nan"),
                "p_abs_max": float("nan"),
            }
        rows.append(row)

    write_csv(rows, args.output_csv)
    print_summary(rows, args)
    print(f"Wrote benchmark CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
