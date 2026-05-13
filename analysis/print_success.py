import numpy as np
import os
import pickle
import re
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot import CONFIGS, LABELS, lookup_optimal_dir, get_model_directories, load_eval, pad_curve_to_length


def process_last_success(directories, method_type="curriculum"):
    last_values = []

    for path in directories:
        if method_type in ("curriculum", "no_refine"):
            base_directories = lookup_optimal_dir(path)
        else:
            base_directories = [path]

        run_curve = []
        for base_dir in base_directories:
            model_dirs = get_model_directories(base_dir)
            for model_path in model_dirs:
                eval_results = load_eval(model_path)
                if eval_results is None:
                    continue
                total = eval_results.get("total_runs", 0)
                success = eval_results.get("success_runs", 0)
                run_curve.append(success / total * 100 if total > 0 else 0)

        if run_curve:
            last_values.append(run_curve[-1])

    return last_values


def main():
    parser = argparse.ArgumentParser(description="Print last success rates")
    parser.add_argument("--task", type=str, default="go2gate",
                        choices=["go2gate", "go2seesaw"], help="Task to evaluate")
    args = parser.parse_args()

    config = CONFIGS[args.task]
    methods = {
        "curriculum": config["curriculum_directories"],
        "no_refine":  config["no_refine_directories"],
        "scratch":    config["scratch_directories"],
        "example":    config["example_directories"],
        "mqe":        config["mqe_directories"],
    }

    print(f"\nLast success rates for task: {args.task}\n")
    print(f"{'Method':<20} {'Mean':>8}  {'Std':>8}  {'Values'}")
    print("-" * 70)

    for method_key, dirs in methods.items():
        values = process_last_success(dirs, method_key)
        if values:
            mean = np.mean(values)
            std = np.std(values)
            label = LABELS[method_key]
            vals_str = "  ".join(f"{v:.1f}" for v in values)
            print(f"{label:<20} {mean:>7.2f}%  {std:>7.2f}%  [{vals_str}]")
        else:
            print(f"{LABELS[method_key]:<20}  no data")


if __name__ == "__main__":
    main()
