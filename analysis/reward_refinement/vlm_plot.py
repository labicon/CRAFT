"""
Plot success rate progression across VLM refinement samples.

Reads eval_results.pkl from each sample's checkpoint directory and generates:
  1. Success rate per refinement sample (individual runs + mean ± std)
  2. Cumulative best success rate vs. sample number (mean ± std)

Run vlm_eval.py first to populate eval_results.pkl in each checkpoint.

Usage:
    python -m analysis.reward_refinement.vlm_plot \
        --dir analysis/reward_refinement/logs \
        [--output path/to/figure.pdf] [--show]
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ──────────────────────────────────────────────────────────────────────────────
# Discovery helpers (shared with vlm_eval.py)
# ──────────────────────────────────────────────────────────────────────────────

def find_vlm_runs(root):
    """Return sorted list of vlm_* directories under root, or root itself if it is one."""
    if os.path.isdir(os.path.join(root, "models")) and os.path.exists(
        os.path.join(root, "state.pkl")
    ):
        return [root]
    runs = []
    for item in sorted(os.listdir(root)):
        if item.startswith("vlm_"):
            item_path = os.path.join(root, item)
            if os.path.isdir(item_path):
                runs.append(item_path)
    return runs


def get_sample_checkpoint(log_dir, sample_idx):
    """Return path to the highest-step rl_model_N_steps dir for sample_idx, or None."""
    model_dir = os.path.join(log_dir, "models", f"sample_{sample_idx}")
    if not os.path.isdir(model_dir):
        return None
    pattern = re.compile(r"^rl_model_(\d+)_steps$")
    dirs = []
    for item in os.listdir(model_dir):
        m = pattern.match(item)
        if m:
            dirs.append((int(m.group(1)), os.path.join(model_dir, item)))
    if not dirs:
        return None
    dirs.sort(key=lambda x: x[0])
    return dirs[-1][1]


def count_samples(log_dir):
    """Count sample_* directories under models/."""
    models_dir = os.path.join(log_dir, "models")
    if not os.path.isdir(models_dir):
        return 0
    return sum(
        1 for item in os.listdir(models_dir)
        if re.match(r"^sample_\d+$", item) and os.path.isdir(os.path.join(models_dir, item))
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def collect_run_results(log_dir):
    """
    Return list of (success_rate, partial_success_rate) per sample.
    Entry is None when checkpoint or eval_results.pkl is missing.
    """
    n_samples = count_samples(log_dir)
    results = []
    for i in range(n_samples):
        checkpoint = get_sample_checkpoint(log_dir, i)
        if checkpoint is None:
            results.append(None)
            continue
        eval_path = os.path.join(checkpoint, "eval_results.pkl")
        if not os.path.exists(eval_path):
            results.append(None)
            continue
        with open(eval_path, "rb") as f:
            er = pickle.load(f)
        sr = er["success_runs"] / er["total_runs"] * 100
        psr = er["partial_success_runs"] / er["total_runs"] * 100
        results.append((sr, psr))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_runs(all_run_results):
    """
    Build (success_matrix, partial_matrix) of shape (n_runs, max_samples).
    Missing entries are NaN.
    """
    max_samples = max(len(r) for r in all_run_results) if all_run_results else 0
    success_rows, partial_rows = [], []
    for run in all_run_results:
        sr_row, psr_row = [], []
        for entry in run:
            if entry is None:
                sr_row.append(np.nan)
                psr_row.append(np.nan)
            else:
                sr_row.append(entry[0])
                psr_row.append(entry[1])
        # Pad shorter runs with NaN
        pad = max_samples - len(sr_row)
        sr_row += [np.nan] * pad
        psr_row += [np.nan] * pad
        success_rows.append(sr_row)
        partial_rows.append(psr_row)
    return np.array(success_rows, dtype=float), np.array(partial_rows, dtype=float)


def cumulative_best(row):
    """Cumulative maximum of a 1-D array, treating NaN as missing."""
    out = np.full_like(row, np.nan)
    best = np.nan
    for i, v in enumerate(row):
        if not np.isnan(v):
            best = v if np.isnan(best) else max(best, v)
        out[i] = best
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_progression(success_matrix, partial_matrix, output_path, show):
    import matplotlib.pyplot as plt

    n_runs, n_samples = success_matrix.shape
    x = np.arange(n_samples)
    colors = plt.cm.tab10.colors

    mean_sr = np.nanmean(success_matrix, axis=0)
    std_sr = np.nanstd(success_matrix, axis=0)
    mean_psr = np.nanmean(partial_matrix, axis=0)
    std_psr = np.nanstd(partial_matrix, axis=0)

    cum_best_matrix = np.array([cumulative_best(success_matrix[i]) for i in range(n_runs)])
    mean_cum = np.nanmean(cum_best_matrix, axis=0)
    std_cum = np.nanstd(cum_best_matrix, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: per-sample success rate ────────────────────────────────────────
    ax = axes[0]
    for i in range(n_runs):
        ax.plot(x, success_matrix[i], color=colors[i % len(colors)], alpha=0.30, linewidth=1.2)
    ax.plot(x, mean_sr, color="tab:blue", linewidth=2.5, label="Success (mean)")
    ax.fill_between(x, mean_sr - std_sr, mean_sr + std_sr, color="tab:blue", alpha=0.15)
    ax.plot(x, mean_psr, color="tab:orange", linewidth=2.0, linestyle="--",
            label="Partial success (mean)")
    ax.fill_between(x, mean_psr - std_psr, mean_psr + std_psr, color="tab:orange", alpha=0.12)
    ax.set_xlabel("Refinement sample", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_title("Success Rate per VLM Refinement Sample", fontsize=13)
    ax.set_xticks(x)
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Right: cumulative best ────────────────────────────────────────────────
    ax = axes[1]
    for i in range(n_runs):
        ax.plot(x, cum_best_matrix[i], color=colors[i % len(colors)], alpha=0.30, linewidth=1.2)
    ax.plot(x, mean_cum, color="tab:green", linewidth=2.5, label="Cumulative best (mean)")
    ax.fill_between(x, mean_cum - std_cum, mean_cum + std_cum, color="tab:green", alpha=0.15)
    ax.set_xlabel("Refinement sample", fontsize=13)
    ax.set_ylabel("Best success rate so far (%)", fontsize=13)
    ax.set_title("Cumulative Best Success Rate", fontsize=13)
    ax.set_xticks(x)
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot VLM refinement success rate progression across samples."
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Root directory with vlm_* subdirs, or a single vlm_* run directory.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path. Defaults to <logdir>/vlm_refinement_progression.pdf",
    )
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    dir_arg = os.path.abspath(args.dir)
    output_path = args.output or os.path.join(dir_arg, "vlm_refinement_progression.pdf")
    show = args.show

    if not os.path.isdir(dir_arg):
        raise FileNotFoundError(f"Directory not found: {dir_arg}")

    # Set matplotlib backend before importing pyplot.
    import matplotlib
    if not show:
        matplotlib.use("Agg")

    run_dirs = find_vlm_runs(dir_arg)
    if not run_dirs:
        print(f"No VLM refinement run directories found under {dir_arg}")
        return

    print(f"Found {len(run_dirs)} VLM refinement run(s)")

    all_run_results = []
    for log_dir in run_dirs:
        results = collect_run_results(log_dir)
        evaluated = sum(1 for r in results if r is not None)
        missing = sum(1 for r in results if r is None)
        print(f"  {os.path.basename(log_dir)}: {len(results)} samples, "
              f"{evaluated} evaluated, {missing} missing")
        all_run_results.append(results)

    any_data = any(r is not None for run in all_run_results for r in run)
    if not any_data:
        print("\nNo eval_results.pkl found. Run vlm_eval.py first.")
        return

    success_matrix, partial_matrix = aggregate_runs(all_run_results)
    n_runs, n_samples = success_matrix.shape
    print(f"\nAggregated: {n_runs} runs × {n_samples} samples")

    # Summary table
    print(f"\n{'Sample':<8} {'Mean SR':>8} {'± Std':>7} {'Mean PSR':>9} {'N eval':>7}")
    for col in range(n_samples):
        col_sr = success_matrix[:, col]
        col_psr = partial_matrix[:, col]
        n_eval = int(np.sum(~np.isnan(col_sr)))
        mean_sr = float(np.nanmean(col_sr))
        std_sr = float(np.nanstd(col_sr))
        mean_psr = float(np.nanmean(col_psr))
        print(f"{col:<8} {mean_sr:>7.1f}% {std_sr:>6.1f}% {mean_psr:>8.1f}%  {n_eval:>5}")

    plot_progression(success_matrix, partial_matrix, output_path, show)
    print("\nDone.")


if __name__ == "__main__":
    main()
