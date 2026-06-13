"""Compare per-cell field extrapolation against continuation baselines."""

import csv
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import pnm_backend

from state_initialization import fit_force_law, predict_u_mean


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


def extract_state(solver):
    return {
        "u": np.asarray(solver.get_u(), dtype=np.float64),
        "v": np.asarray(solver.get_v(), dtype=np.float64),
        "w": np.asarray(solver.get_w(), dtype=np.float64),
        "p": np.asarray(solver.get_p(), dtype=np.float64),
    }


def load_state(solver, state):
    solver.set_state(state["u"], state["v"], state["w"], state["p"])


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
        "state": extract_state(solver),
    }


def lagrange_extrapolate(target, xs, ys):
    """Lagrange extrapolation for scalar or array-valued data."""
    if len(xs) == 1:
        return ys[0]
    result = np.zeros_like(ys[0], dtype=np.float64)
    for i, xi in enumerate(xs):
        basis = 1.0
        for j, xj in enumerate(xs):
            if i == j:
                continue
            basis *= (target - xj) / (xi - xj)
        result += basis * ys[i]
    return result


def build_fieldwise_initial_state(history, predicted_u_mean, target_force):
    """Extrapolate each field using the latest available states."""
    samples = history[-3:]
    u_params = [row["u_mean"] for row in samples]
    f_params = [row["force"] for row in samples]

    state = {}
    for name in ("u", "v", "w"):
        fields = [row["state"][name] for row in samples]
        state[name] = lagrange_extrapolate(predicted_u_mean, u_params, fields)
    p_fields = [row["state"]["p"] for row in samples]
    state["p"] = lagrange_extrapolate(target_force, f_params, p_fields)
    return state


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    warm_up_cuda()
    sdf_zyx, origin, spacing = load_sdf()
    zero_template = np.zeros((128, 128, 128), dtype=np.float64)
    history = [{
        "force": 0.0,
        "u_mean": 0.0,
        "state": {
            "u": zero_template.copy(),
            "v": zero_template.copy(),
            "w": zero_template.copy(),
            "p": zero_template.copy(),
        },
    }]
    rows = []
    solver = build_solver(sdf_zyx, origin, spacing)

    for index, force in enumerate(FORCES):
        dt = 50.0 / force
        model = "seed"
        predicted_u_mean = None
        initial_u_mean = current_u_mean(solver)

        if index == 0:
            pass
        elif index == 1:
            a = history[-1]["force"] / history[-1]["u_mean"]
            predicted_u_mean = force / a
            model = f"linear(a={a:.6g})"
            predicted_state = build_fieldwise_initial_state(history, predicted_u_mean, force)
            load_state(solver, predicted_state)
            initial_u_mean = current_u_mean(solver)
        else:
            real_history = history[1:]
            a, b = fit_force_law(
                [row["u_mean"] for row in real_history],
                [row["force"] for row in real_history],
            )
            predicted_u_mean = predict_u_mean(force, a, b)
            model = f"quadratic(a={a:.6g},b={b:.6g})"
            predicted_state = build_fieldwise_initial_state(history, predicted_u_mean, force)
            load_state(solver, predicted_state)
            initial_u_mean = current_u_mean(solver)

        result = run_to_convergence(solver, force, dt)
        prediction_error = float("nan")
        if predicted_u_mean is not None and result["status"] == "ok":
            prediction_error = abs(predicted_u_mean - result["u_mean"]) / max(abs(result["u_mean"]), 1e-14)

        row = {
            "force": force,
            "dt": dt,
            "predicted_u_mean": predicted_u_mean,
            "initial_u_mean": initial_u_mean,
            "prediction_rel_error": prediction_error,
            "model": model,
            "status": result["status"],
            "note": result["note"],
            "steps": result["steps"],
            "elapsed_s": result["elapsed_s"],
            "u_mean": result["u_mean"],
            "p_mean": result["p_mean"],
            "re": result["re"],
            "outer_mean": result["outer_mean"],
            "outer_max": result["outer_max"],
            "outer_total": result["outer_total"],
        }
        rows.append(row)
        print(
            f"fieldwise force={force:g} steps={row['steps']} outer_total={row['outer_total']} "
            f"status={row['status']}"
        )
        if result["status"] != "ok":
            break
        history.append({
            "force": force,
            "u_mean": result["u_mean"],
            "state": result["state"],
        })

    write_csv(rows, "output/packing_128_fieldwise_extrapolated_continuation.csv")

    with open("output/packing_128_plain_continuation.csv", newline="", encoding="utf-8") as handle:
        plain_rows = list(csv.DictReader(handle))
    with open("output/packing_128_extrapolated_continuation.csv", newline="", encoding="utf-8") as handle:
        scalar_rows = list(csv.DictReader(handle))

    comparison_rows = []
    for fieldwise, plain, scalar in zip(rows, plain_rows, scalar_rows):
        comparison_rows.append({
            "force": fieldwise["force"],
            "dt": fieldwise["dt"],
            "plain_outer_total": int(plain["outer_total"]),
            "scalar_outer_total": int(scalar["outer_total"]),
            "fieldwise_outer_total": int(fieldwise["outer_total"]),
            "fieldwise_minus_plain": int(fieldwise["outer_total"]) - int(plain["outer_total"]),
            "fieldwise_minus_scalar": int(fieldwise["outer_total"]) - int(scalar["outer_total"]),
            "plain_steps": int(plain["steps"]),
            "scalar_steps": int(scalar["steps"]),
            "fieldwise_steps": int(fieldwise["steps"]),
            "fieldwise_status": fieldwise["status"],
        })
    write_csv(comparison_rows, "output/packing_128_fieldwise_extrapolation_comparison.csv")

    total_plain = sum(int(row["outer_total"]) for row in plain_rows[:len(rows)])
    total_scalar = sum(int(row["outer_total"]) for row in scalar_rows[:len(rows)])
    total_fieldwise = sum(int(row["outer_total"]) for row in rows)
    print(
        f"total_outer plain={total_plain} scalar={total_scalar} fieldwise={total_fieldwise}"
    )


if __name__ == "__main__":
    main()
