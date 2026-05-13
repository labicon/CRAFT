import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot import get_model_directories, load_eval, lookup_optimal_dir, pad_curve_to_length

CRAFT_RUNS_ROOT = "CRAFT_runs"


def discover_runs(task, include_no_refine=False):
    """Return sorted list of run directories under CRAFT_runs/{task}/."""
    task_dir = os.path.join(CRAFT_RUNS_ROOT, task)
    if not os.path.isdir(task_dir):
        print(f"Warning: {task_dir} does not exist")
        return []

    timestamp_re = re.compile(r"^\d{2}-\d{2}_\d{2}-\d{2}")
    runs = []
    for name in sorted(os.listdir(task_dir)):
        path = os.path.join(task_dir, name)
        if not os.path.isdir(path):
            continue
        if not timestamp_re.match(name):
            continue
        is_no_refine = "no_refine" in name
        if is_no_refine and not include_no_refine:
            continue
        runs.append(path)
    return runs


def load_success_curve(run_dir, metric="total"):
    """Return (steps, success_pct) arrays for the optimal model path in a run."""
    base_dirs = lookup_optimal_dir(run_dir)
    points = []
    for base_dir in base_dirs:
        model_dirs = get_model_directories(base_dir)
        for model_path in model_dirs:
            ev = load_eval(model_path)
            if ev is None:
                continue
            step_match = re.search(r"rl_model_(\d+)_steps", model_path)
            step = int(step_match.group(1)) if step_match else len(points)
            total = ev.get("total_runs", 0)
            if total == 0:
                continue
            if metric == "total":
                pct = ev["success_runs"] / total * 100
            else:
                pct = ev["partial_success_runs"] / total * 100
            points.append((step, pct))

    if not points:
        return np.array([]), np.array([])
    points.sort(key=lambda x: x[0])
    steps = np.array([p[0] for p in points])
    pcts = np.array([p[1] for p in points])
    return steps, pcts


def print_summary(task, runs, metric="total"):
    label = "Total" if metric == "total" else "Partial"
    print(f"\n{label} success rate — task: {task}\n")
    print(f"  {'Run':<30}  {'Final':>7}  {'Max':>7}")
    print("  " + "-" * 50)

    finals = []
    for run_dir in runs:
        _, pcts = load_success_curve(run_dir, metric)
        name = os.path.basename(run_dir)
        if len(pcts) == 0:
            print(f"  {name:<30}  {'no data':>7}")
            continue
        final, maxval = pcts[-1], pcts.max()
        finals.append(final)
        print(f"  {name:<30}  {final:>6.1f}%  {maxval:>6.1f}%")

    if finals:
        print("  " + "-" * 50)
        print(f"  {'Mean ± Std':<30}  {np.mean(finals):>6.1f}%  (σ={np.std(finals):.1f}%)")


def plot_success(task, runs, metric="total", output_dir="figure"):
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(10, 6))

    all_curves = []
    for run_dir in runs:
        steps, pcts = load_success_curve(run_dir, metric)
        if len(pcts) == 0:
            continue
        all_curves.append((steps, pcts))
        ax.plot(steps, pcts, color="#1f77b4", alpha=0.3, linewidth=1.5)

    if all_curves:
        max_len = max(len(p) for _, p in all_curves)
        padded = np.array([pad_curve_to_length(list(p), max_len) for _, p in all_curves])
        ref_steps = all_curves[0][0]
        if len(ref_steps) < max_len:
            step = ref_steps[-1] - ref_steps[-2] if len(ref_steps) > 1 else 2_500_000
            extra = np.arange(1, max_len - len(ref_steps) + 1) * step + ref_steps[-1]
            ref_steps = np.concatenate([ref_steps, extra])
        mean_curve = padded.mean(axis=0)
        std_curve = padded.std(axis=0)
        ax.plot(ref_steps, mean_curve, color="#1f77b4", linewidth=3, label="CRAFT (mean)")
        ax.fill_between(ref_steps, mean_curve - std_curve, mean_curve + std_curve,
                        color="#1f77b4", alpha=0.15)

    label = "Total" if metric == "total" else "Partial"
    ax.set_title(f"{label} Success Rate — {task}", fontsize=24)
    ax.set_xlabel("Training Steps", fontsize=18)
    ax.set_ylabel("Success Rate (%)", fontsize=18)
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, fontsize=14)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.xaxis.offsetText.set_fontsize(13)

    suffix = "total" if metric == "total" else "partial"
    path = os.path.join(output_dir, f"{task}_craft_{suffix}_success.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot/print CRAFT success rates")
    parser.add_argument("--task", type=str, default="go2gate",
                        choices=["go2gate", "go2seesaw"], help="Task to evaluate")
    parser.add_argument("--no_refine", action="store_true",
                        help="Include no_refine runs")
    parser.add_argument("--partial", action="store_true",
                        help="Also show partial success rate")
    parser.add_argument("--output_dir", type=str, default="figure",
                        help="Directory for saved figures")
    parser.add_argument("--no_plot", action="store_true",
                        help="Print only, skip plotting")
    args = parser.parse_args()

    runs = discover_runs(args.task, include_no_refine=args.no_refine)
    if not runs:
        print(f"No runs found under {CRAFT_RUNS_ROOT}/{args.task}/")
        return

    print(f"Found {len(runs)} run(s) in CRAFT_runs/{args.task}/")

    print_summary(args.task, runs, metric="total")
    if args.partial:
        print_summary(args.task, runs, metric="partial")

    if not args.no_plot:
        plot_success(args.task, runs, metric="total", output_dir=args.output_dir)
        if args.partial:
            plot_success(args.task, runs, metric="partial", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
