"""Run a multi-resolution packed-bed target-Re study with saved states."""

import csv
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "../build")))
sys.path.append(SCRIPT_DIR)
import pnm_backend

from state_initialization import resample_field_linear


TARGET_RES = [1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
GRID_SIZES = [64, 128, 256]
OUTER_DIAMETER = 0.9498
MAX_STEPS = 500
TARGET_RE_TOL = 0.015
MAX_FORCE_ITERS = 6
OUTPUT_DIR = Path("output/packing_multires_re_targets")
STATE_DIR = OUTPUT_DIR / "states"


def read_vti(path):
    sdf_zyx, origin, spacing = pnm_backend.SDFReader.read_vti(str(path))
    return np.asarray(sdf_zyx, dtype=np.float32), list(origin), list(spacing)


def build_sdf_catalog():
    sdf_256, origin_256, spacing_256 = read_vti("data/packing_256.vti")
    sdf_128, origin_128, spacing_128 = read_vti("data/packing_128.vti")
    sdf_64 = resample_field_linear(sdf_256, (64, 64, 64)).astype(np.float32)
    spacing_64 = [4.0 * spacing_256[0], 4.0 * spacing_256[1], 4.0 * spacing_256[2]]
    return {
        64: {"sdf": sdf_64, "origin": origin_256, "spacing": spacing_64},
        128: {"sdf": sdf_128, "origin": origin_128, "spacing": spacing_128},
        256: {"sdf": sdf_256, "origin": origin_256, "spacing": spacing_256},
    }


def build_solver(resolution, sdf_zyx, origin, spacing):
    solver = pnm_backend.CFDSolver([resolution, resolution, resolution], spacing)
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


def reshape_field(flat_field, resolution):
    return np.asarray(flat_field, dtype=np.float64).reshape(
        (resolution, resolution, resolution)
    )


def flatten_field(field):
    return np.asarray(field, dtype=np.float64).reshape(-1)


def extract_state_3d(solver, resolution):
    return {
        "u": reshape_field(solver.get_u(), resolution),
        "v": reshape_field(solver.get_v(), resolution),
        "w": reshape_field(solver.get_w(), resolution),
        "p": reshape_field(solver.get_p(), resolution),
    }


def load_state_3d(solver, state):
    solver.set_state(
        flatten_field(state["u"]),
        flatten_field(state["v"]),
        flatten_field(state["w"]),
        flatten_field(state["p"]),
    )


def resample_state_3d(state, target_resolution):
    target_shape = (target_resolution, target_resolution, target_resolution)
    return {
        name: resample_field_linear(field, target_shape) for name, field in state.items()
    }


def save_state(path, state, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        u=state["u"],
        v=state["v"],
        w=state["w"],
        p=state["p"],
        **meta,
    )


def current_u_mean(solver):
    return float(np.mean(np.asarray(solver.get_u(), dtype=np.float64)))


def current_p_mean(solver):
    return float(np.mean(np.asarray(solver.get_p(), dtype=np.float64)))


def run_to_convergence(solver, force, dt, resolution):
    solver.set_body_force(pnm_backend.float3(force, 0.0, 0.0))
    u_hist = []
    stable = 0
    outer_hist = []
    status = "ok"
    note = ""
    steps = 0
    start = time.perf_counter()

    for step_idx in range(MAX_STEPS):
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
    u_mean = current_u_mean(solver)
    p_mean = current_p_mean(solver)
    return {
        "status": status,
        "note": note,
        "steps": steps,
        "elapsed_s": elapsed,
        "u_mean": u_mean,
        "p_mean": p_mean,
        "re": u_mean * OUTER_DIAMETER,
        "outer_mean": float(np.mean(outer_hist)) if outer_hist else float("nan"),
        "outer_max": int(np.max(outer_hist)) if outer_hist else 0,
        "outer_total": int(np.sum(outer_hist)) if outer_hist else 0,
        "state": extract_state_3d(solver, resolution),
    }


def update_force_guess(target_re, attempts):
    if len(attempts) == 1:
        re_value = attempts[-1]["re"]
        if re_value <= 1e-12:
            return attempts[-1]["force"] * 2.0
        return attempts[-1]["force"] * target_re / re_value

    sorted_attempts = sorted(attempts, key=lambda row: row["re"])
    below = None
    above = None
    for attempt in sorted_attempts:
        if attempt["re"] <= target_re:
            below = attempt
        if attempt["re"] >= target_re and above is None:
            above = attempt
    if below is not None and above is not None and above["re"] != below["re"]:
        slope = (above["force"] - below["force"]) / (above["re"] - below["re"])
        return below["force"] + slope * (target_re - below["re"])

    a = sorted_attempts[-2]
    b = sorted_attempts[-1]
    if b["re"] == a["re"]:
        return b["force"] * target_re / max(b["re"], 1e-12)
    slope = (b["force"] - a["force"]) / (b["re"] - a["re"])
    return b["force"] + slope * (target_re - b["re"])


def solve_target_re(
    resolution,
    target_re,
    sdf_catalog,
    initial_state=None,
    initial_force=None,
):
    config = sdf_catalog[resolution]
    solver = build_solver(
        resolution, config["sdf"], config["origin"], config["spacing"]
    )
    if initial_state is not None:
        load_state_3d(solver, initial_state)

    target_u_mean = target_re / OUTER_DIAMETER
    force_guess = float(initial_force if initial_force is not None else 700.0 * target_re)
    attempts = []
    best = None

    for force_iter in range(1, MAX_FORCE_ITERS + 1):
        dt = 50.0 / force_guess
        result = run_to_convergence(solver, force_guess, dt, resolution)
        rel_re_error = abs(result["re"] - target_re) / max(target_re, 1e-12)
        attempt = {
            "force_iter": force_iter,
            "force": force_guess,
            "dt": dt,
            "target_re": target_re,
            "rel_re_error": rel_re_error,
            **{k: v for k, v in result.items() if k != "state"},
        }
        attempts.append(attempt)
        if best is None or rel_re_error < best["rel_re_error"]:
            best = {**attempt, "state": result["state"]}
        print(
            f"res={resolution} target_re={target_re:g} iter={force_iter} "
            f"force={force_guess:.6g} re={result['re']:.6g} "
            f"rel_err={rel_re_error:.3e} status={result['status']}",
            flush=True,
        )

        if result["status"] != "ok":
            break
        if rel_re_error <= TARGET_RE_TOL:
            return best, attempts

        next_force = max(update_force_guess(target_re, attempts), 1e-8)
        solver.scale_state(
            target_u_mean / max(result["u_mean"], 1e-12),
            next_force / force_guess,
        )
        force_guess = next_force

    return best, attempts


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_res = {}
    for row in summary_rows:
        by_res.setdefault(int(row["resolution"]), []).append(row)
    for rows in by_res.values():
        rows.sort(key=lambda row: row["target_re"])

    plt.figure(figsize=(7, 5))
    for resolution, rows in sorted(by_res.items()):
        plt.plot(
            [row["target_re"] for row in rows],
            [row["force"] for row in rows],
            marker="o",
            label=f"{resolution}^3",
        )
    plt.xlabel("Target Re_Do")
    plt.ylabel("Required body force")
    plt.title("Force required to maintain the same Re")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "force_vs_re_by_resolution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for resolution, rows in sorted(by_res.items()):
        plt.plot(
            [row["target_re"] for row in rows],
            [row["outer_total"] for row in rows],
            marker="o",
            label=f"{resolution}^3",
        )
    plt.xlabel("Target Re_Do")
    plt.ylabel("outer_total")
    plt.title("Outer work at each target Re")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outer_total_vs_re_by_resolution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for resolution, rows in sorted(by_res.items()):
        plt.plot(
            [row["target_re"] for row in rows],
            [row["steps"] for row in rows],
            marker="o",
            label=f"{resolution}^3",
        )
    plt.xlabel("Target Re_Do")
    plt.ylabel("Time steps to convergence")
    plt.title("Pseudo-transient steps at each target Re")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "steps_vs_re_by_resolution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    ref = by_res[max(by_res)]
    ref_map = {row["target_re"]: row["force"] for row in ref}
    for resolution, rows in sorted(by_res.items()):
        if resolution == max(by_res):
            continue
        errs = []
        targets = []
        for row in rows:
            ref_force = ref_map[row["target_re"]]
            errs.append(abs(row["force"] - ref_force) / abs(ref_force))
            targets.append(row["target_re"])
        plt.semilogy(targets, errs, marker="o", label=f"{resolution}^3 vs 256^3")
    plt.xlabel("Target Re_Do")
    plt.ylabel("Relative force error")
    plt.title("Resolution convergence relative to 256^3")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "force_relative_error_vs_256.png", dpi=180)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    sdf_catalog = build_sdf_catalog()

    summary_rows = []
    attempt_rows = []
    converged_states = {64: {}, 128: {}, 256: {}}

    for target_re in TARGET_RES:
        state_64 = None
        force_64 = None
        if converged_states[64]:
            prev_re = max(converged_states[64].keys())
            prev_state = converged_states[64][prev_re]
            state_64 = prev_state["state"]
            force_64 = prev_state["force"] * target_re / prev_re

        best_64, attempts_64 = solve_target_re(
            64,
            target_re,
            sdf_catalog,
            initial_state=state_64,
            initial_force=force_64,
        )
        for row in attempts_64:
            attempt_rows.append({"resolution": 64, **row})
        if best_64["status"] != "ok":
            raise RuntimeError(f"64^3 failed at target Re={target_re:g}")
        summary_rows.append({"resolution": 64, **{k: v for k, v in best_64.items() if k != "state"}})
        converged_states[64][target_re] = {"state": best_64["state"], "force": best_64["force"]}
        save_state(
            STATE_DIR / f"packing_state_res64_re{target_re:05.1f}.npz",
            best_64["state"],
            {"target_re": target_re, "resolution": 64, "force": best_64["force"]},
        )

        state_128 = resample_state_3d(best_64["state"], 128)
        best_128, attempts_128 = solve_target_re(
            128,
            target_re,
            sdf_catalog,
            initial_state=state_128,
            initial_force=best_64["force"],
        )
        for row in attempts_128:
            attempt_rows.append({"resolution": 128, **row})
        if best_128["status"] != "ok":
            raise RuntimeError(f"128^3 failed at target Re={target_re:g}")
        summary_rows.append({"resolution": 128, **{k: v for k, v in best_128.items() if k != "state"}})
        converged_states[128][target_re] = {"state": best_128["state"], "force": best_128["force"]}
        save_state(
            STATE_DIR / f"packing_state_res128_re{target_re:05.1f}.npz",
            best_128["state"],
            {"target_re": target_re, "resolution": 128, "force": best_128["force"]},
        )

        state_256 = resample_state_3d(best_128["state"], 256)
        best_256, attempts_256 = solve_target_re(
            256,
            target_re,
            sdf_catalog,
            initial_state=state_256,
            initial_force=best_128["force"],
        )
        for row in attempts_256:
            attempt_rows.append({"resolution": 256, **row})
        if best_256["status"] != "ok":
            raise RuntimeError(f"256^3 failed at target Re={target_re:g}")
        summary_rows.append({"resolution": 256, **{k: v for k, v in best_256.items() if k != "state"}})
        converged_states[256][target_re] = {"state": best_256["state"], "force": best_256["force"]}
        save_state(
            STATE_DIR / f"packing_state_res256_re{target_re:05.1f}.npz",
            best_256["state"],
            {"target_re": target_re, "resolution": 256, "force": best_256["force"]},
        )

    write_csv(OUTPUT_DIR / "multires_re_targets_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "multires_re_targets_attempts.csv", attempt_rows)
    plot_results(summary_rows)
    print("WROTE", OUTPUT_DIR / "multires_re_targets_summary.csv")
    print("WROTE", OUTPUT_DIR / "multires_re_targets_attempts.csv")


if __name__ == "__main__":
    main()
