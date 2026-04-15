"""Sweep pressure/velocity multigrid settings for the periodic-sphere benchmark."""

import argparse
import csv
import os
import sys
from copy import deepcopy

THIS_DIR = os.path.dirname(__file__)
sys.path.append(THIS_DIR)

from validate_periodic_spheres_n128 import get_reference_k, run_case, warm_up_cuda


def make_config(
    name,
    enable_pressure_mg,
    pressure_mg_params,
    enable_velocity_mg,
    velocity_mg_params,
    pressure_iter,
    velocity_iter,
):
    """Build one tuning configuration record."""
    return {
        "name": name,
        "enable_pressure_mg": enable_pressure_mg,
        "pressure_mg_params": pressure_mg_params,
        "enable_velocity_mg": enable_velocity_mg,
        "velocity_mg_params": velocity_mg_params,
        "pressure_iter": pressure_iter,
        "velocity_iter": velocity_iter,
    }


CONFIGS = [
    make_config("rbgs_p50_v2", False, (4, 2, 2, 32, 2), False, (4, 1, 1, 16, 1), 50, 2),
    make_config("rbgs_p20_v2", False, (4, 2, 2, 32, 2), False, (4, 1, 1, 16, 1), 20, 2),
    make_config("rbgs_p10_v2", False, (4, 2, 2, 32, 2), False, (4, 1, 1, 16, 1), 10, 2),
    make_config("rbgs_p20_v4", False, (4, 2, 2, 32, 2), False, (4, 1, 1, 16, 1), 20, 4),
    make_config("pmg_default_v2", True, (4, 2, 2, 32, 2), False, (4, 1, 1, 16, 1), 50, 2),
    make_config("pmg_light_v2", True, (4, 1, 1, 16, 1), False, (4, 1, 1, 16, 1), 50, 2),
    make_config("pmg_light_v4", True, (4, 1, 1, 16, 1), False, (4, 1, 1, 16, 1), 50, 4),
    make_config("pmg_deep_v1", True, (4, 2, 2, 32, 2), False, (4, 1, 1, 16, 1), 50, 1),
    make_config("pmg_light_l3_v2", True, (3, 1, 1, 16, 1), False, (3, 1, 1, 16, 1), 50, 2),
    make_config("pmg_light_l3_v1", True, (3, 1, 1, 16, 1), False, (3, 1, 1, 16, 1), 50, 1),
    make_config("pmg_light_l3_v4", True, (3, 1, 1, 16, 1), False, (3, 1, 1, 16, 1), 50, 4),
    make_config("pmg_light_l5_v2", True, (5, 1, 1, 16, 1), False, (5, 1, 1, 16, 1), 50, 2),
    make_config("pmg_default_l5_v2", True, (5, 2, 2, 32, 2), False, (5, 1, 1, 16, 1), 50, 2),
    make_config("pmg_mid_l5_v2", True, (5, 1, 1, 16, 2), False, (5, 1, 1, 16, 1), 50, 2),
    make_config("pmg_light_l5_p20_v2", True, (5, 1, 1, 16, 1), False, (5, 1, 1, 16, 1), 20, 2),
    make_config("fullmg_fast", True, (4, 1, 1, 16, 1), True, (4, 1, 1, 16, 1), 50, 2),
    make_config("fullmg_balanced", True, (4, 1, 1, 16, 1), True, (4, 2, 2, 32, 1), 50, 2),
    make_config("fullmg_inner4", True, (4, 4, 4, 32, 1), True, (4, 4, 4, 32, 1), 50, 2),
    make_config("fullmg_inner6", True, (4, 6, 6, 32, 1), True, (4, 6, 6, 32, 1), 50, 2),
    make_config("fullmg_vcycle2", True, (4, 1, 1, 16, 1), True, (4, 1, 1, 16, 2), 50, 2),
    make_config("fullmg_robust", True, (4, 2, 2, 32, 2), True, (4, 2, 2, 32, 1), 50, 2),
    make_config("fullmg_balanced_l3", True, (3, 1, 1, 16, 1), True, (3, 2, 2, 32, 1), 50, 2),
    make_config("fullmg_balanced_l5", True, (5, 1, 1, 16, 1), True, (5, 2, 2, 32, 1), 50, 2),
    make_config("fullmg_balanced_l5_v4", True, (5, 1, 1, 16, 1), True, (5, 2, 2, 32, 1), 50, 4),
    make_config("fullmg_robust_l5", True, (5, 2, 2, 32, 2), True, (5, 2, 2, 32, 1), 50, 2),
]


def run_config(phi, res, config, args):
    """Execute one tuning point and capture failures in the output row."""
    row = deepcopy(config)
    row["phi"] = phi
    row["resolution"] = res
    row["k_ref"] = get_reference_k(phi)
    row["status"] = "ok"
    try:
        result = run_case(
            phi,
            res,
            config["enable_pressure_mg"],
            config["pressure_mg_params"],
            config["enable_velocity_mg"],
            config["velocity_mg_params"],
            config["pressure_iter"],
            config["velocity_iter"],
            args.outer_iterations,
            args.outer_tol,
            args.outer_mode,
            args.dt,
            args.max_steps,
            args.convergence_stride,
            args.convergence_rel_tol,
        )
        row.update(result)
        row["ref_rel_error"] = abs(row["k_sim"] - row["k_ref"]) / row["k_ref"]
        row["within_reference_tol"] = row["ref_rel_error"] <= args.reference_rel_tol
    except Exception as exc:
        row["status"] = f"failed: {exc}"
        row["elapsed_s"] = float("nan")
        row["per_step_s"] = float("nan")
        row["steps_taken"] = 0
        row["outer_iterations_mean"] = float("nan")
        row["outer_iterations_max"] = 0
        row["k_sim"] = float("nan")
        row["ref_rel_error"] = float("nan")
        row["within_reference_tol"] = False
    return row


def write_csv(rows, path):
    """Persist the sweep results for later analysis."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "phi",
        "resolution",
        "name",
        "enable_pressure_mg",
        "enable_velocity_mg",
        "pressure_iter",
        "velocity_iter",
        "pressure_mg_params",
        "velocity_mg_params",
        "elapsed_s",
        "per_step_s",
        "steps_taken",
        "outer_iterations_mean",
        "outer_iterations_max",
        "k_sim",
        "k_ref",
        "ref_rel_error",
        "within_reference_tol",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            writer.writerow(out)


def print_summary(rows, args):
    """Print the fastest acceptable configurations for each case."""
    print(
        f"Tuning setup: resolutions={args.resolutions}, phis={args.phis}, "
        f"outer_iterations={args.outer_iterations}, outer_tol={args.outer_tol}, "
        f"outer_mode={args.outer_mode}, "
        f"max_steps={args.max_steps}"
    )
    print()
    grouped = {}
    for row in rows:
        grouped.setdefault((row["resolution"], row["phi"]), []).append(row)

    for (res, phi), group_rows in sorted(grouped.items()):
        print(f"Case: N={res}, phi={phi}")
        print(
            f"{'config':<20} {'time(s)':>10} {'steps':>8} {'outer_mean':>12} "
            f"{'outer_max':>10} {'K':>10} {'err%':>8} {'ok':>6} {'status':<24}"
        )
        best = sorted(
            group_rows,
            key=lambda row: (
                row["status"] != "ok",
                not row["within_reference_tol"],
                row["elapsed_s"] if row["elapsed_s"] == row["elapsed_s"] else float("inf"),
            ),
        )
        for row in best:
            print(
                f"{row['name']:<20} {row['elapsed_s']:10.3f} {row['steps_taken']:8d} "
                f"{row['outer_iterations_mean']:12.2f} {row['outer_iterations_max']:10d} "
                f"{row['k_sim']:10.4f} {100.0 * row['ref_rel_error']:8.2f} "
                f"{str(row['within_reference_tol']):>6} {row['status']:<24}"
            )
        print()


def parse_args():
    """Parse CLI arguments for the multigrid parameter sweep."""
    parser = argparse.ArgumentParser(
        description="Tune RBGS and multigrid settings for periodic spheres at lower resolutions."
    )
    parser.add_argument("--phis", type=float, nargs="+", default=[0.001, 0.5236])
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--convergence-stride", type=int, default=5)
    parser.add_argument("--convergence-rel-tol", type=float, default=1e-8)
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--outer-tol", type=float, default=1e-6)
    parser.add_argument("--outer-mode", type=int, default=0)
    parser.add_argument("--reference-rel-tol", type=float, default=0.02)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional list of config names to run.",
    )
    parser.add_argument(
        "--output-csv",
        default="output/periodic_spheres_multigrid_tuning.csv",
    )
    return parser.parse_args()


def main():
    """Run the configured parameter sweep and emit a CSV summary."""
    args = parse_args()
    warm_up_cuda()
    configs = CONFIGS
    if args.configs is not None:
        wanted = set(args.configs)
        configs = [config for config in CONFIGS if config["name"] in wanted]
        missing = sorted(wanted - {config["name"] for config in configs})
        if missing:
            raise ValueError(f"unknown configs requested: {missing}")
    rows = []
    for res in args.resolutions:
        for phi in args.phis:
            for config in configs:
                print(
                    f"Running config={config['name']} N={res} phi={phi} "
                    f"outer_tol={args.outer_tol}"
                )
                row = run_config(phi, res, config, args)
                rows.append(row)
    write_csv(rows, args.output_csv)
    print_summary(rows, args)
    print(f"Wrote tuning CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
