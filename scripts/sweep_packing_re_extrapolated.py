"""Run an extrapolated Reynolds-number sweep on the 128^3 packing case.

The sweep reuses the last converged state, scales it on the GPU, and then
continues the solve at the next predicted body force. Each successful point is
saved as a compressed NumPy archive so it can later seed a 256^3 run.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import pnm_backend

from state_initialization import (
    extract_solver_state,
    fit_force_law,
    load_solver_state,
    scale_solver_state,
)


DEFAULT_RE_TARGETS = [
    1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100,
    125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000,
]


def resolve_default_sdf():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/packing_128.vti"))


def load_sdf(sdf_path):
    sdf_zyx, origin, spacing = pnm_backend.SDFReader.read_vti(sdf_path)
    return np.asarray(sdf_zyx, dtype=np.float32), list(origin), list(spacing)


def build_solver(args, sdf_zyx, origin, spacing):
    """Create and configure a solver for the packed-bed case."""
    resolution = [int(sdf_zyx.shape[2]), int(sdf_zyx.shape[1]), int(sdf_zyx.shape[0])]
    solver = pnm_backend.CFDSolver(resolution, spacing)
    solver.initialize(sdf_zyx, origin, spacing)
    solver.set_body_force(pnm_backend.float3(args.seed_force, 0.0, 0.0))
    solver.set_rho(args.rho)
    solver.set_mu(args.mu)
    solver.set_ibm_scheme(args.ibm_scheme)
    solver.set_pressure_solver_params(iter=args.pressure_iter)
    solver.set_velocity_solver_params(iter=args.velocity_iter)
    solver.set_outer_iterations(args.outer_iterations)
    solver.set_outer_tolerance(args.outer_tol)
    solver.set_outer_convergence_mode(args.outer_mode)
    solver.set_pressure_multigrid_enabled(True)
    solver.set_pressure_multigrid_params(
        args.pressure_mg_levels,
        args.pressure_mg_pre,
        args.pressure_mg_post,
        args.pressure_mg_bottom,
        args.pressure_mg_cycles,
    )
    solver.set_velocity_multigrid_enabled(False)
    return solver


def warm_up_cuda():
    """Warm up CUDA and JIT compilation before timing the real sweep."""
    shape = [16, 16, 16]
    spacing = [1.0, 1.0, 1.0]
    x = np.arange(shape[2], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    z = np.arange(shape[0], dtype=np.float32)
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
    sdf_xyz = np.sqrt((xg - 8.0) ** 2 + (yg - 8.0) ** 2 + (zg - 8.0) ** 2) - 4.0
    sdf_zyx = np.transpose(sdf_xyz, (2, 1, 0)).astype(np.float32)

    solver = pnm_backend.CFDSolver(shape, spacing)
    solver.initialize(sdf_zyx, [0.0, 0.0, 0.0], spacing)
    solver.set_body_force(pnm_backend.float3(1.0, 0.0, 0.0))
    solver.set_rho(1.0)
    solver.set_mu(1.0)
    solver.set_pressure_solver_params(iter=2)
    solver.set_velocity_solver_params(iter=1)
    solver.set_outer_iterations(1)
    solver.set_outer_tolerance(0.0)
    solver.step(0.1)
    np.mean(np.asarray(solver.get_u(), dtype=np.float64))


def mean_u(solver):
    return float(np.mean(np.asarray(solver.get_u(), dtype=np.float64)))


def compute_reynolds(u_mean, rho, mu, outer_diameter):
    return rho * u_mean * outer_diameter / mu


def choose_dt(force, args):
    return max(args.min_dt, min(args.max_dt, args.force_dt_product / max(abs(force), 1e-14)))


def target_hit(target_re, achieved_re, args):
    tol = max(args.re_target_tol_abs, args.re_target_tol_rel * max(target_re, 1.0))
    return abs(achieved_re - target_re) <= tol


def predict_force(successes, target_u_mean, args):
    """Predict the next body force from recent converged states."""
    if len(successes) == 1:
        last = successes[-1]
        return max(args.min_force_growth * last["force"],
                   last["force"] * target_u_mean / max(last["u_mean"], 1e-14))

    history = successes[-min(len(successes), args.fit_window):]
    u_samples = [row["u_mean"] for row in history]
    force_samples = [row["force"] for row in history]
    linear_coeff, quadratic_coeff = fit_force_law(u_samples, force_samples)
    predicted = linear_coeff * target_u_mean + quadratic_coeff * target_u_mean ** 2
    predicted = max(predicted, args.min_force_growth * successes[-1]["force"])
    return predicted


def corrected_force_guess(records, target_re):
    """Update the force guess using the newest local continuation samples."""
    below = [row for row in records if row["re"] <= target_re]
    above = [row for row in records if row["re"] >= target_re]
    if below and above:
        low = max(below, key=lambda row: row["re"])
        high = min(above, key=lambda row: row["re"])
        if high["re"] > low["re"]:
            alpha = (target_re - low["re"]) / (high["re"] - low["re"])
            return (1.0 - alpha) * low["force"] + alpha * high["force"]

    latest = records[-1]
    return latest["force"] * target_re / max(latest["re"], 1e-12)


def steady_solve(solver, force, dt, args):
    """March the solver until the mean velocity stops changing materially."""
    solver.set_body_force(pnm_backend.float3(force, 0.0, 0.0))
    start = time.perf_counter()
    u_samples = []
    stable_checks = 0
    steps_taken = 0
    outer_iterations_used = []

    for step_idx in range(args.max_steps):
        solver.step(dt)
        steps_taken = step_idx + 1
        outer_iterations_used.append(int(solver.get_last_outer_iterations()))

        if step_idx % args.convergence_stride != 0:
            continue

        u_mean = mean_u(solver)
        p_mean = float(np.mean(np.asarray(solver.get_p(), dtype=np.float64)))
        if not np.isfinite(u_mean) or not np.isfinite(p_mean):
            raise RuntimeError(
                f"non-finite state detected at step {steps_taken}: "
                f"u_mean={u_mean}, p_mean={p_mean}"
            )

        u_samples.append(u_mean)
        if len(u_samples) < 2:
            continue

        rel_change = abs(u_samples[-1] - u_samples[-2]) / max(abs(u_samples[-1]), 1e-14)
        if rel_change < args.convergence_rel_tol:
            stable_checks += 1
            if stable_checks >= args.convergence_stable_checks:
                break
        else:
            stable_checks = 0

    elapsed = time.perf_counter() - start
    state = extract_solver_state(solver)
    u_mean = float(np.mean(state["u"]))
    p_mean = float(np.mean(state["p"]))
    return {
        "elapsed_s": elapsed,
        "steps_taken": steps_taken,
        "outer_iterations_mean": float(np.mean(outer_iterations_used)),
        "outer_iterations_max": int(np.max(outer_iterations_used)),
        "u_mean": u_mean,
        "p_mean": p_mean,
        "u_abs_max": float(np.max(np.abs(state["u"]))),
        "p_abs_max": float(np.max(np.abs(state["p"]))),
        "state": state,
    }


def save_state(output_dir, index, record, origin, spacing, resolution):
    """Save a converged state and its metadata for later 256^3 interpolation."""
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{index:03d}_re_{record['re']:.3f}"
    state_path = os.path.join(output_dir, f"{stem}.npz")
    np.savez_compressed(
        state_path,
        u=record["state"]["u"],
        v=record["state"]["v"],
        w=record["state"]["w"],
        p=record["state"]["p"],
        origin=np.asarray(origin, dtype=np.float64),
        spacing=np.asarray(spacing, dtype=np.float64),
        resolution=np.asarray(resolution, dtype=np.int32),
        target_re=np.asarray(record["target_re"], dtype=np.float64),
        achieved_re=np.asarray(record["re"], dtype=np.float64),
        force=np.asarray(record["force"], dtype=np.float64),
        dt=np.asarray(record["dt"], dtype=np.float64),
    )
    return state_path


def write_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "index",
        "target_re",
        "re",
        "force",
        "dt",
        "status",
        "elapsed_s",
        "steps_taken",
        "outer_iterations_mean",
        "outer_iterations_max",
        "u_mean",
        "p_mean",
        "u_abs_max",
        "p_abs_max",
        "state_file",
        "notes",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def summarize(rows):
    print(
        f"{'idx':>3} {'target_Re':>10} {'Re':>10} {'force':>12} {'dt':>10} "
        f"{'steps':>8} {'time(s)':>10} {'status':<12} {'state':<40}"
    )
    print("-" * 120)
    for row in rows:
        print(
            f"{row['index']:3d} {row['target_re']:10.3f} {row.get('re', float('nan')):10.3f} "
            f"{row.get('force', float('nan')):12.4f} {row.get('dt', float('nan')):10.4e} "
            f"{row.get('steps_taken', 0):8d} {row.get('elapsed_s', float('nan')):10.3f} "
            f"{row['status']:<12} {os.path.basename(row.get('state_file', '')):<40}"
        )


def save_run_metadata(output_dir, args, rows):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "sdf": args.sdf,
        "outer_diameter": args.outer_diameter,
        "rho": args.rho,
        "mu": args.mu,
        "pressure_mg": [
            args.pressure_mg_levels,
            args.pressure_mg_pre,
            args.pressure_mg_post,
            args.pressure_mg_bottom,
            args.pressure_mg_cycles,
        ],
        "pressure_iter": args.pressure_iter,
        "velocity_iter": args.velocity_iter,
        "outer_iterations": args.outer_iterations,
        "outer_tol": args.outer_tol,
        "outer_mode": args.outer_mode,
        "force_dt_product": args.force_dt_product,
        "targets": args.targets,
        "successful_re_max": max((row["re"] for row in rows if row["status"] == "ok"), default=None),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def attempt_target(target_re, last_success, successes, args, sdf_zyx, origin, spacing):
    """Attempt one target using the latest converged state as the continuation base."""
    target_u_mean = target_re * args.mu / (args.rho * args.outer_diameter)
    base_record = last_success
    local_records = [last_success]
    predicted_force = predict_force(successes, target_u_mean, args)

    for _ in range(args.max_force_corrections):
        dt = choose_dt(predicted_force, args)

        solver = build_solver(args, sdf_zyx, origin, spacing)
        load_solver_state(solver, base_record["state"])
        scale_solver_state(
            solver,
            base_record["u_mean"],
            target_u_mean,
            base_record["force"],
            predicted_force,
        )

        result = steady_solve(solver, predicted_force, dt, args)
        achieved_re = compute_reynolds(result["u_mean"], args.rho, args.mu, args.outer_diameter)
        record = {
            "target_re": target_re,
            "re": achieved_re,
            "force": predicted_force,
            "dt": dt,
            **result,
        }
        if target_hit(target_re, achieved_re, args):
            return record

        local_records.append(record)
        base_record = record
        predicted_force = corrected_force_guess(local_records, target_re)

    return record


def advance_target(target_re, last_success, successes, args, sdf_zyx, origin, spacing):
    """Try to reach a target Reynolds number, bisecting if the jump fails."""
    low_re = last_success["re"]
    high_re = target_re
    best = None
    notes = []
    tried_target = False

    while True:
        trial_re = high_re if not tried_target else 0.5 * (low_re + high_re)
        tried_target = True
        try:
            result = attempt_target(trial_re, last_success, successes, args, sdf_zyx, origin, spacing)
            best = result
            low_re = result["re"]
            notes.append(f"success@{trial_re:.3f}")
            if abs(trial_re - target_re) <= 1e-9 and target_hit(target_re, result["re"], args):
                return best, "; ".join(notes)
        except Exception as exc:
            high_re = trial_re
            notes.append(f"fail@{trial_re:.3f}:{exc}")
            if best is None and high_re - low_re <= args.re_refine_tol:
                raise

        if high_re - low_re <= args.re_refine_tol:
            break

    if best is None:
        raise RuntimeError(f"failed to advance beyond Re={last_success['re']:.3f}")
    return best, "; ".join(notes)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrapolated Re sweep on packing_128 with saved final states."
    )
    parser.add_argument("--sdf", default=resolve_default_sdf())
    parser.add_argument("--output-dir", default="output/packing_128_re_sweep")
    parser.add_argument("--output-csv", default="output/packing_128_re_sweep/summary.csv")
    parser.add_argument("--targets", type=float, nargs="+", default=DEFAULT_RE_TARGETS)
    parser.add_argument("--seed-force", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--outer-diameter", type=float, default=0.9498)
    parser.add_argument("--ibm-scheme", type=int, default=0)
    parser.add_argument("--pressure-iter", type=int, default=50)
    parser.add_argument("--velocity-iter", type=int, default=2)
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--outer-tol", type=float, default=1e-4)
    parser.add_argument("--outer-mode", type=int, default=1)
    parser.add_argument("--pressure-mg-levels", type=int, default=3)
    parser.add_argument("--pressure-mg-pre", type=int, default=1)
    parser.add_argument("--pressure-mg-post", type=int, default=1)
    parser.add_argument("--pressure-mg-bottom", type=int, default=16)
    parser.add_argument("--pressure-mg-cycles", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--convergence-stride", type=int, default=5)
    parser.add_argument("--convergence-rel-tol", type=float, default=1e-3)
    parser.add_argument("--convergence-stable-checks", type=int, default=3)
    parser.add_argument("--force-dt-product", type=float, default=50.0)
    parser.add_argument("--max-dt", type=float, default=64.0)
    parser.add_argument("--min-dt", type=float, default=1.0e-4)
    parser.add_argument("--fit-window", type=int, default=6)
    parser.add_argument("--min-force-growth", type=float, default=1.05)
    parser.add_argument("--re-refine-tol", type=float, default=0.5)
    parser.add_argument("--re-target-tol-abs", type=float, default=0.25)
    parser.add_argument("--re-target-tol-rel", type=float, default=0.05)
    parser.add_argument("--max-force-corrections", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rho <= 0.0 or args.mu <= 0.0:
        raise ValueError("rho and mu must be positive to define Reynolds number")

    warm_up_cuda()
    sdf_zyx, origin, spacing = load_sdf(args.sdf)
    resolution = [int(sdf_zyx.shape[2]), int(sdf_zyx.shape[1]), int(sdf_zyx.shape[0])]
    rows = []

    seed_solver = build_solver(args, sdf_zyx, origin, spacing)
    seed_dt = choose_dt(args.seed_force, args)
    seed_result = steady_solve(seed_solver, args.seed_force, seed_dt, args)
    seed_re = compute_reynolds(seed_result["u_mean"], args.rho, args.mu, args.outer_diameter)
    seed_record = {
        "index": 0,
        "target_re": seed_re,
        "re": seed_re,
        "force": args.seed_force,
        "dt": seed_dt,
        "status": "ok",
        "notes": "seed",
        **seed_result,
    }
    seed_record["state_file"] = save_state(args.output_dir, 0, seed_record, origin, spacing, resolution)
    rows.append(seed_record)
    write_csv(rows, args.output_csv)

    successes = [seed_record]
    last_success = seed_record

    for target_re in args.targets:
        if target_re <= last_success["re"] + 1e-9:
            continue
        row = {"index": len(rows), "target_re": target_re}
        try:
            result, notes = advance_target(target_re, last_success, successes, args, sdf_zyx, origin, spacing)
            row.update(result)
            row["status"] = "ok"
            row["notes"] = notes
            row["state_file"] = save_state(args.output_dir, row["index"], row, origin, spacing, resolution)
            rows.append(row)
            write_csv(rows, args.output_csv)
            successes.append(row)
            last_success = row
            print(
                f"Reached target Re={target_re:.3f} with achieved Re={row['re']:.3f}, "
                f"force={row['force']:.4f}, dt={row['dt']:.4e}"
            )
        except Exception as exc:
            row["status"] = "failed"
            row["notes"] = str(exc)
            row["state_file"] = ""
            rows.append(row)
            write_csv(rows, args.output_csv)
            print(f"Failed to reach target Re={target_re:.3f}: {exc}")
            break

    summarize(rows)
    save_run_metadata(args.output_dir, args, rows)
    print(f"Wrote sweep summary to {args.output_csv}")


if __name__ == "__main__":
    main()
