import numpy as np
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot import CONFIGS, LABELS, load_eval, load_eureka_best_eval
from craft_eval import resolve_best_sample, get_final_checkpoint


def curriculum_final_checkpoint(run_dir):
    """Return final checkpoint path for the best policy in a curriculum run dir."""
    sample_dir, _ = resolve_best_sample(run_dir)
    if sample_dir is None:
        return None
    model_dir = os.path.join(sample_dir, "model")
    if not os.path.isdir(model_dir):
        return None
    return get_final_checkpoint(model_dir)


def get_final_success_rates(directories, method_type):
    values = []
    for path in directories:
        if method_type in ("curriculum", "no_refine"):
            checkpoint = curriculum_final_checkpoint(path)
        else:
            checkpoint = get_final_checkpoint(path)

        if checkpoint is None:
            print(f"  Warning: no checkpoint found for {path}")
            continue

        eval_results = load_eval(checkpoint)
        if eval_results is None:
            print(f"  Warning: no eval_results.pkl in {checkpoint}")
            continue

        total = eval_results.get("total_runs", 0)
        success = eval_results.get("success_runs", 0)
        values.append(success / total * 100 if total > 0 else 0)

    return values


def main():
    parser = argparse.ArgumentParser(description="Print final checkpoint success rates")
    parser.add_argument("--task", type=str, default="go2gate",
                        choices=["go2gate", "go2seesaw"], help="Task to evaluate")
    args = parser.parse_args()

    config = CONFIGS[args.task]
    methods = {
        "curriculum": (config["curriculum_directories"], "curriculum"),
        "no_refine":  (config["no_refine_directories"],  "no_refine"),
        "scratch":    (config["scratch_directories"],     "scratch"),
        "example":    (config["example_directories"],     "example"),
        "mqe":        (config["mqe_directories"],         "mqe"),
    }

    # Modality ablation arms share the CRAFT run layout, so they resolve their final
    # checkpoint the same way. Skipped entirely until the arm has runs configured.
    for arm in ("state_only", "vision_only"):
        arm_dirs = config.get(f"{arm}_directories", [])
        if arm_dirs:
            methods[arm] = (arm_dirs, "curriculum")

    print(f"\nFinal checkpoint success rates for task: {args.task}\n")
    print(f"{'Method':<20} {'Mean':>8}  {'Std':>8}  {'Values'}")
    print("-" * 70)

    for method_key, (dirs, method_type) in methods.items():
        values = get_final_success_rates(dirs, method_type)
        label = LABELS[method_key]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            vals_str = "  ".join(f"{v:.1f}" for v in values)
            print(f"{label:<20} {mean:>7.2f}%  {std:>7.2f}%  [{vals_str}]")
        else:
            print(f"{label:<20}  no data")

    eureka_root = config.get("eureka_root")
    if eureka_root:
        eureka_results = load_eureka_best_eval(eureka_root, args.task)
        label = LABELS["eureka"]
        if eureka_results:
            values = [
                er["success_runs"] / er["total_runs"] * 100 if er["total_runs"] > 0 else 0
                for er in eureka_results
            ]
            mean = np.mean(values)
            std = np.std(values)
            vals_str = "  ".join(f"{v:.1f}" for v in values)
            print(f"{label:<20} {mean:>7.2f}%  {std:>7.2f}%  [{vals_str}]")
        else:
            print(f"{label:<20}  no data")


if __name__ == "__main__":
    main()
