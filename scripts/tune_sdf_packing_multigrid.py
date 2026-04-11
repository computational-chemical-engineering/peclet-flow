import argparse
import csv
import os
import sys
from copy import deepcopy

THIS_DIR = os.path.dirname(__file__)
sys.path.append(THIS_DIR)

from benchmark_sdf_packing_multigrid import load_sdf, run_case, warm_up_cuda


def make_config(
    name,
    method,
    pressure_iter=50,
    velocity_iter=2,
    pressure_mg=(4, 1, 1, 16, 1),
    velocity_mg=(4, 2, 2, 32, 1),
):
    return {
        "name": name,
        "method": method,
        "pressure_iter": pressure_iter,
        "velocity_iter": velocity_iter,
        "pressure_mg": pressure_mg,
        "velocity_mg": velocity_mg,
    }


CONFIGS = [
    make_config("rbgs_p50_v2", "rbgs", pressure_iter=50, velocity_iter=2),
    make_config("rbgs_p20_v2", "rbgs", pressure_iter=20, velocity_iter=2),
    make_config("rbgs_p10_v2", "rbgs", pressure_iter=10, velocity_iter=2),
    make_config("rbgs_p20_v4", "rbgs", pressure_iter=20, velocity_iter=4),
    make_config("pmg_l2_light", "pmg", pressure_mg=(2, 1, 1, 16, 1)),
    make_config("pmg_l3_light", "pmg", pressure_mg=(3, 1, 1, 16, 1)),
    make_config("pmg_l3_light_v1", "pmg", velocity_iter=1, pressure_mg=(3, 1, 1, 16, 1)),
    make_config("pmg_l3_light_v4", "pmg", velocity_iter=4, pressure_mg=(3, 1, 1, 16, 1)),
    make_config("pmg_l3_light_b8", "pmg", pressure_mg=(3, 1, 1, 8, 1)),
    make_config("pmg_l3_light_b32", "pmg", pressure_mg=(3, 1, 1, 32, 1)),
    make_config("pmg_l3_mid", "pmg", pressure_mg=(3, 1, 1, 16, 2)),
    make_config("pmg_l3_default", "pmg", pressure_mg=(3, 2, 2, 32, 2)),
    make_config("pmg_l4_light", "pmg", pressure_mg=(4, 1, 1, 16, 1)),
    make_config("fullmg_l2_balanced", "fullmg", pressure_mg=(2, 1, 1, 16, 1), velocity_mg=(2, 2, 2, 32, 1)),
    make_config("fullmg_l3_balanced", "fullmg", pressure_mg=(3, 1, 1, 16, 1), velocity_mg=(3, 2, 2, 32, 1)),
    make_config("fullmg_l3_inner4", "fullmg", pressure_mg=(3, 1, 1, 16, 1), velocity_mg=(3, 4, 4, 32, 1)),
    make_config("fullmg_l3_robust", "fullmg", pressure_mg=(3, 2, 2, 32, 1), velocity_mg=(3, 4, 4, 64, 1)),
]


def apply_config(args, config):
    cfg = deepcopy(args)
    cfg.pressure_iter = config["pressure_iter"]
    cfg.velocity_iter = config["velocity_iter"]
    (
        cfg.pressure_mg_levels,
        cfg.pressure_mg_pre,
        cfg.pressure_mg_post,
        cfg.pressure_mg_bottom,
        cfg.pressure_mg_cycles,
    ) = config["pressure_mg"]
    (
        cfg.velocity_mg_levels,
        cfg.velocity_mg_pre,
        cfg.velocity_mg_post,
        cfg.velocity_mg_bottom,
        cfg.velocity_mg_cycles,
    ) = config["velocity_mg"]
    return cfg


def run_config(base_args, sdf_zyx, origin, spacing, config):
    args = apply_config(base_args, config)
    row = {
        "name": config["name"],
        "method": config["method"],
        "pressure_iter": config["pressure_iter"],
        "velocity_iter": config["velocity_iter"],
        "pressure_mg": config["pressure_mg"],
        "velocity_mg": config["velocity_mg"],
        "status": "ok",
    }
    try:
        result = run_case(args, sdf_zyx, origin, spacing, config["method"])
        row.update(result)
    except Exception as exc:
        row.update(
            {
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
        )
    return row


def write_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "name",
        "method",
        "pressure_iter",
        "velocity_iter",
        "pressure_mg",
        "velocity_mg",
        "elapsed_s",
        "per_step_s",
        "steps_taken",
        "outer_iterations_mean",
        "outer_iterations_max",
        "u_mean",
        "u_abs_max",
        "p_abs_max",
        "status",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def print_summary(rows, args):
    print(f"Packing SDF: {args.sdf}")
    print(
        f"dt={args.dt}, max_steps={args.max_steps}, convergence_stride={args.convergence_stride}, "
        f"convergence_rel_tol={args.convergence_rel_tol}, outer_tol={args.outer_tol}, outer_mode={args.outer_mode}"
    )
    print()
    print(
        f"{'config':<20} {'method':<8} {'time(s)':>10} {'steps':>8} {'outer_mean':>12} "
        f"{'outer_max':>10} {'u_mean':>12} {'status':<24}"
    )
    print("-" * 110)
    best = sorted(
        rows,
        key=lambda row: (
            row["status"] != "ok",
            row["elapsed_s"] if row["elapsed_s"] == row["elapsed_s"] else float("inf"),
        ),
    )
    for row in best:
        print(
            f"{row['name']:<20} {row['method']:<8} {row['elapsed_s']:10.3f} "
            f"{row['steps_taken']:8d} {row['outer_iterations_mean']:12.2f} "
            f"{row['outer_iterations_max']:10d} {row['u_mean']:12.4e} {row['status']:<24}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune RBGS and multigrid settings on a real SDF packing geometry."
    )
    parser.add_argument(
        "--sdf",
        default=os.path.abspath(os.path.join(THIS_DIR, "../ring_packing_sdf.vti")),
    )
    parser.add_argument("--configs", nargs="+", default=None)
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
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--outer-tol", type=float, default=1e-6)
    parser.add_argument("--outer-mode", type=int, default=1)
    parser.add_argument(
        "--output-csv",
        default="output/packing_multigrid_tuning.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.configs is None:
        configs = CONFIGS
    else:
        wanted = set(args.configs)
        configs = [config for config in CONFIGS if config["name"] in wanted]
        missing = sorted(wanted - {config["name"] for config in configs})
        if missing:
            raise ValueError(f"unknown configs requested: {missing}")

    sdf_zyx, origin, spacing = load_sdf(args.sdf)
    warm_up_cuda()

    rows = []
    for config in configs:
        print(f"Running config={config['name']}")
        rows.append(run_config(args, sdf_zyx, origin, spacing, config))

    write_csv(rows, args.output_csv)
    print_summary(rows, args)
    print(f"Wrote tuning CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
