"""Compare plain and extrapolated continuation on the 128^3 packing case."""

import csv
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import pnm_backend

from state_initialization import fit_force_law, predict_u_mean, scale_solver_state


FORCES = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
OUTER_DIAMETER = 0.9498


def load_sdf():
    sdf_zyx, origin, spacing = pnm_backend.SDFReader.read_vti("data/packing_128.vti")
    return np.asarray(sdf_zyx, dtype=np.float32), list(origin), list(spacing)


def build_solver(sdf_zyx, origin, spacing):
    solver = pnm_backend.CFDSolver([128, 128, 128], spacing)
    solver.initialize(sdf_zyx, origin, spacing)
    solver.set_rho(1.0)
    solver.set_mu(1.0)
    solver.set_ibm_scheme(0)
    solver.set_pressure_solver_params(iter=50)
    solver.set_velocity_solver_params(iter=2)
    solver.set_outer_iterations(20)
    solver.set_outer_tolerance(1e-4)
    solver.set_outer_convergence_mode(1)
    solver.set_pressure_multigrid_enabled(True)
    solver.set_pressure_multigrid_params(3, 1, 1, 16, 1)
    solver.set_velocity_multigrid_enabled(False)
    return solver


def warm_up_cuda():
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


def current_u_mean(solver):
    return float(np.mean(np.asarray(solver.get_u(), dtype=np.float64)))


def current_p_mean(solver):
    return float(np.mean(np.asarray(solver.get_p(), dtype=np.float64)))


def run_to_convergence(solver, force, dt):
    solver.set_body_force(pnm_backend.float3(force, 0.0, 0.0))
    u_hist = []
    stable = 0
    outer_hist = []
    start = time.perf_counter()
    steps = 0
    status = "ok"
    note = ""

    for step_idx in range(400):
        solver.step(dt)
        steps = step_idx + 1
        outer_hist.append(int(solver.get_last_outer_iterations()))
        if step_idx % 5 != 0:
            continue
        u_mean = current_u_mean(solver)
        p_mean = current_p_mean(solver)
        if not np.isfinite(u_mean) or not np.isfinite(p_mean):
            status = "failed"
            note = f"non-finite at step {steps}"
            break
        u_hist.append(u_mean)
        if len(u_hist) < 2:
            continue
        rel = abs(u_hist[-1] - u_hist[-2]) / max(abs(u_hist[-1]), 1e-14)
        if rel < 1e-3:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0

    elapsed = time.perf_counter() - start
    u = np.asarray(solver.get_u(), dtype=np.float64)
    p = np.asarray(solver.get_p(), dtype=np.float64)
    return {
        "status": status,
        "note": note,
        "steps": steps,
        "elapsed_s": elapsed,
        "u_mean": float(np.mean(u)),
        "p_mean": float(np.mean(p)),
        "re": float(np.mean(u)) * OUTER_DIAMETER,
        "outer_mean": float(np.mean(outer_hist)) if outer_hist else float("nan"),
        "outer_max": int(np.max(outer_hist)) if outer_hist else 0,
        "outer_total": int(np.sum(outer_hist)) if outer_hist else 0,
    }


def run_plain_continuation(sdf_zyx, origin, spacing):
    solver = build_solver(sdf_zyx, origin, spacing)
    rows = []
    for force in FORCES:
        dt = 50.0 / force
        result = run_to_convergence(solver, force, dt)
        rows.append({"force": force, "dt": dt, **result})
        print(f"plain force={force:g} steps={result['steps']} outer_total={result['outer_total']} status={result['status']}")
        if result["status"] != "ok":
            break
    return rows


def run_extrapolated_continuation(sdf_zyx, origin, spacing):
    solver = build_solver(sdf_zyx, origin, spacing)
    rows = []
    for index, force in enumerate(FORCES):
        dt = 50.0 / force
        predicted_u_mean = None
        model = "seed"
        prediction_error = float("nan")

        if index > 0:
            previous = rows[-1]
            if index == 1:
                linear_coeff = previous["force"] / previous["u_mean"]
                quadratic_coeff = 0.0
                predicted_u_mean = force / linear_coeff
                model = f"linear(a={linear_coeff:.6g})"
            else:
                linear_coeff, quadratic_coeff = fit_force_law(
                    [row["u_mean"] for row in rows],
                    [row["force"] for row in rows],
                )
                predicted_u_mean = predict_u_mean(force, linear_coeff, quadratic_coeff)
                model = f"quadratic(a={linear_coeff:.6g},b={quadratic_coeff:.6g})"

            scale_solver_state(
                solver,
                previous["u_mean"],
                predicted_u_mean,
                previous["force"],
                force,
            )

        initial_u_mean = current_u_mean(solver)
        result = run_to_convergence(solver, force, dt)
        if predicted_u_mean is not None and result["status"] == "ok":
            prediction_error = abs(predicted_u_mean - result["u_mean"]) / max(abs(result["u_mean"]), 1e-14)

        rows.append(
            {
                "force": force,
                "dt": dt,
                "predicted_u_mean": predicted_u_mean,
                "initial_u_mean": initial_u_mean,
                "prediction_rel_error": prediction_error,
                "model": model,
                **result,
            }
        )
        print(
            f"extrap force={force:g} steps={result['steps']} outer_total={result['outer_total']} "
            f"status={result['status']}"
        )
        if result["status"] != "ok":
            break
    return rows


def write_detailed_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(plain_rows, extrap_rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "force",
        "dt",
        "plain_steps",
        "extrap_steps",
        "step_delta",
        "plain_outer_total",
        "extrap_outer_total",
        "outer_total_delta",
        "plain_elapsed_s",
        "extrap_elapsed_s",
        "elapsed_delta_s",
        "plain_u_mean",
        "extrap_u_mean",
        "predicted_u_mean",
        "initial_u_mean",
        "prediction_rel_error",
        "model",
        "plain_status",
        "extrap_status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for plain, extrap in zip(plain_rows, extrap_rows):
            writer.writerow(
                {
                    "force": plain["force"],
                    "dt": plain["dt"],
                    "plain_steps": plain["steps"],
                    "extrap_steps": extrap["steps"],
                    "step_delta": extrap["steps"] - plain["steps"],
                    "plain_outer_total": plain["outer_total"],
                    "extrap_outer_total": extrap["outer_total"],
                    "outer_total_delta": extrap["outer_total"] - plain["outer_total"],
                    "plain_elapsed_s": plain["elapsed_s"],
                    "extrap_elapsed_s": extrap["elapsed_s"],
                    "elapsed_delta_s": extrap["elapsed_s"] - plain["elapsed_s"],
                    "plain_u_mean": plain["u_mean"],
                    "extrap_u_mean": extrap["u_mean"],
                    "predicted_u_mean": extrap.get("predicted_u_mean"),
                    "initial_u_mean": extrap.get("initial_u_mean"),
                    "prediction_rel_error": extrap.get("prediction_rel_error"),
                    "model": extrap.get("model"),
                    "plain_status": plain["status"],
                    "extrap_status": extrap["status"],
                }
            )


def print_summary(plain_rows, extrap_rows):
    print(
        f"{'force':>8} {'plain_steps':>12} {'extra_steps':>12} {'d_steps':>8} "
        f"{'plain_outer':>12} {'extra_outer':>12} {'d_outer':>10} {'pred_err%':>10}"
    )
    print("-" * 96)
    for plain, extrap in zip(plain_rows, extrap_rows):
        pred_err = extrap.get("prediction_rel_error", float("nan"))
        print(
            f"{plain['force']:8g} {plain['steps']:12d} {extrap['steps']:12d} "
            f"{extrap['steps'] - plain['steps']:8d} {plain['outer_total']:12d} "
            f"{extrap['outer_total']:12d} {extrap['outer_total'] - plain['outer_total']:10d} "
            f"{100.0 * pred_err if pred_err == pred_err else float('nan'):10.2f}"
        )

    total_plain_steps = sum(row["steps"] for row in plain_rows if row["status"] == "ok")
    total_extrap_steps = sum(row["steps"] for row in extrap_rows if row["status"] == "ok")
    total_plain_outer = sum(row["outer_total"] for row in plain_rows if row["status"] == "ok")
    total_extrap_outer = sum(row["outer_total"] for row in extrap_rows if row["status"] == "ok")
    total_plain_time = sum(row["elapsed_s"] for row in plain_rows if row["status"] == "ok")
    total_extrap_time = sum(row["elapsed_s"] for row in extrap_rows if row["status"] == "ok")
    print()
    print(
        f"totals: plain_steps={total_plain_steps}, extrap_steps={total_extrap_steps}, "
        f"plain_outer={total_plain_outer}, extrap_outer={total_extrap_outer}, "
        f"plain_time={total_plain_time:.3f}s, extrap_time={total_extrap_time:.3f}s"
    )


def main():
    warm_up_cuda()
    sdf_zyx, origin, spacing = load_sdf()
    plain_rows = run_plain_continuation(sdf_zyx, origin, spacing)
    extrap_rows = run_extrapolated_continuation(sdf_zyx, origin, spacing)
    write_detailed_csv(plain_rows, "output/packing_128_plain_continuation.csv")
    write_detailed_csv(extrap_rows, "output/packing_128_extrapolated_continuation.csv")
    write_comparison_csv(
        plain_rows,
        extrap_rows,
        "output/packing_128_continuation_comparison.csv",
    )
    print_summary(plain_rows, extrap_rows)


if __name__ == "__main__":
    main()
