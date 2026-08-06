"""
Compare VLM reward refinement vs Eureka across reward samples.

For a fair sample-budget comparison this script aligns both methods on a
shared x-axis of "reward sample index".  Eureka iter_0 (candidates 0-2)
occupies indices 0-2 and iter_1 (candidates 0-2) occupies indices 3-5,
giving 6 samples total — the same budget used for VLM refinement.

Prerequisites:
  Run vlm_eval.py and eureka_eval.py first so that eval_results.pkl files
  exist inside each checkpoint directory.

Usage:
    python -m analysis.reward_refinement.vlm_eureka_compare \
        --vlm-dir  analysis/reward_refinement/logs \
        --eureka-dir logs/Eureka_long/go2gate \
        [--max-samples 6] [--output path/to/figure.pdf] [--show]
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Re-use helpers from vlm_plot to avoid duplication
from analysis.reward_refinement.vlm_plot import (
    find_vlm_runs,
    collect_run_results,
    aggregate_runs,
    cumulative_best,
)


# ──────────────────────────────────────────────────────────────────────────────
# Eureka data loading
# ──────────────────────────────────────────────────────────────────────────────

def _get_model_dirs(model_dir):
    """Return checkpoint dirs sorted by step count (ascending)."""
    pattern = re.compile(r"^rl_model_(\d+)_steps$")
    dirs = []
    if not os.path.isdir(model_dir):
        return dirs
    for item in os.listdir(model_dir):
        m = pattern.match(item)
        if m:
            dirs.append((int(m.group(1)), os.path.join(model_dir, item)))
    dirs.sort(key=lambda x: x[0])
    return [p for _, p in dirs]


def _load_eval(checkpoint_dir):
    """Load eval_results.pkl from a checkpoint dir; return None if missing."""
    path = os.path.join(checkpoint_dir, "eval_results.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def find_eureka_runs(eureka_root):
    """Return list of timestamped run directories that contain an eureka/ subdir."""
    runs = []
    if not os.path.isdir(eureka_root):
        return runs
    for item in sorted(os.listdir(eureka_root)):
        item_path = os.path.join(eureka_root, item)
        if os.path.isdir(item_path) and os.path.isdir(os.path.join(item_path, "eureka")):
            runs.append(item_path)
    return runs


def _count_candidates(eureka_dir, iteration):
    """Return the number of candidate_N dirs present for a given iteration."""
    it_dir = os.path.join(eureka_dir, f"iteration_{iteration}")
    if not os.path.isdir(it_dir):
        return 0
    pattern = re.compile(r"^candidate_(\d+)$")
    indices = [int(m.group(1)) for item in os.listdir(it_dir) if (m := pattern.match(item))]
    return max(indices) + 1 if indices else 0


def collect_eureka_results(run_dir, iterations=(0, 1), candidates_per_iter=None):
    """
    Return a flat list of (sr, psr) | None for the specified iterations,
    ordered iter0_cand0, iter0_cand1, ..., iter1_cand0, iter1_cand1, ...

    candidates_per_iter: int to use the same count for all iterations, or a
    list/tuple with one count per iteration. None means auto-detect.
    Returns the results list and a list of per-iteration candidate counts.
    """
    eureka_dir = os.path.join(run_dir, "eureka")
    results = []
    counts = []
    for idx, it in enumerate(iterations):
        if candidates_per_iter is None:
            n = _count_candidates(eureka_dir, it)
        elif isinstance(candidates_per_iter, (list, tuple)):
            n = candidates_per_iter[idx]
        else:
            n = candidates_per_iter
        counts.append(n)
        for cand in range(n):
            model_dir = os.path.join(eureka_dir, f"iteration_{it}", f"candidate_{cand}", "model")
            checkpoints = _get_model_dirs(model_dir)
            if not checkpoints:
                results.append(None)
                continue
            er = _load_eval(checkpoints[-1])
            if er is None:
                results.append(None)
                continue
            sr = er["success_runs"] / er["total_runs"] * 100
            psr = er["partial_success_runs"] / er["total_runs"] * 100
            results.append((sr, psr))
    return results, counts


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare VLM refinement vs Eureka reward sample efficiency."
    )
    parser.add_argument(
        "--vlm-dir", type=str, required=True,
        help="Root directory with vlm_* run dirs (or a single vlm_* dir).",
    )
    parser.add_argument(
        "--eureka-dir", type=str, required=True,
        help="Directory containing timestamped Eureka runs (each with eureka/ subdir).",
    )
    parser.add_argument(
        "--eureka-iters", type=int, nargs="+", default=[0, 1], metavar="N",
        help="Which Eureka iteration indices to include (default: 0 1). "
             "Pass '--eureka-iters 0' to compare only iter_0 (3 samples).",
    )
    parser.add_argument(
        "--candidates-per-iter", type=int, nargs="+", default=None, metavar="N",
        help="Candidates per Eureka iteration. One value applies to all iters; "
             "multiple values map 1-to-1 with --eureka-iters. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum reward samples to compare per method. "
             "Defaults to sum of candidates across all Eureka iterations.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Base output path. Two files are saved: <base>_success_rate.pdf and "
             "<base>_cumulative_best.pdf. Defaults to <vlm-dir>/vlm_vs_eureka.",
    )
    parser.add_argument("--show", action="store_true", help="Display plot interactively.")
    args = parser.parse_args()

    vlm_root = os.path.abspath(args.vlm_dir)
    eureka_root = os.path.abspath(args.eureka_dir)
    eureka_iters = sorted(set(args.eureka_iters))

    # candidates_per_iter: None → auto-detect; scalar → same for all; list → per-iter
    raw_cpi = args.candidates_per_iter
    if raw_cpi is None:
        candidates_per_iter = None  # auto-detect per run
    elif len(raw_cpi) == 1:
        candidates_per_iter = raw_cpi[0]
    else:
        candidates_per_iter = raw_cpi

    output_path = args.output or os.path.join(vlm_root, "vlm_vs_eureka")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    try:
        plt.rcParams["font.family"] = "P052"
    except Exception:
        pass
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # ── Load Eureka data first (to derive max_samples) ────────────────────────
    eureka_run_dirs = find_eureka_runs(eureka_root)
    print(f"Found {len(eureka_run_dirs)} Eureka run(s)")
    eureka_all = []
    eureka_iter_counts = None  # per-iteration candidate counts (from first run)
    for d in eureka_run_dirs:
        res, counts = collect_eureka_results(d, iterations=eureka_iters,
                                             candidates_per_iter=candidates_per_iter)
        if eureka_iter_counts is None:
            eureka_iter_counts = counts
        evaluated = sum(1 for r in res if r is not None)
        missing = sum(1 for r in res if r is None)
        print(f"  Eureka {os.path.basename(d)}: {len(res)} samples "
              f"({counts}), {evaluated} evaluated, {missing} missing")
        eureka_all.append(res)

    if eureka_iter_counts is None:
        eureka_iter_counts = []

    max_samples = (args.max_samples if args.max_samples is not None
                   else sum(eureka_iter_counts) if eureka_iter_counts
                   else len(eureka_iters) * 3)

    # ── Load VLM data ──────────────────────────────────────────────────────────
    vlm_run_dirs = find_vlm_runs(vlm_root)
    print(f"\nFound {len(vlm_run_dirs)} VLM run(s)")
    vlm_all = []
    for d in vlm_run_dirs:
        res = collect_run_results(d)[:max_samples]
        evaluated = sum(1 for r in res if r is not None)
        missing = sum(1 for r in res if r is None)
        print(f"  VLM {os.path.basename(d)}: {len(res)} samples, "
              f"{evaluated} evaluated, {missing} missing")
        vlm_all.append(res)

    # ── Guard ──────────────────────────────────────────────────────────────────
    any_vlm = any(r is not None for run in vlm_all for r in run)
    any_eureka = any(r is not None for run in eureka_all for r in run)
    if not any_vlm and not any_eureka:
        print("\nNo eval_results.pkl found for either method. Run the eval scripts first.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    vlm_sr, vlm_psr = aggregate_runs(vlm_all) if any_vlm else (np.empty((0, 0)), np.empty((0, 0)))
    eureka_sr, eureka_psr = aggregate_runs(eureka_all) if any_eureka else (np.empty((0, 0)), np.empty((0, 0)))

    # Pad both to max_samples columns so x-axes align
    def _pad_matrix(mat, n_cols):
        if mat.shape[1] >= n_cols:
            return mat[:, :n_cols]
        pad = np.full((mat.shape[0], n_cols - mat.shape[1]), np.nan)
        return np.hstack([mat, pad])

    n_cols = max_samples
    if any_vlm:
        vlm_sr = _pad_matrix(vlm_sr, n_cols)
        vlm_psr = _pad_matrix(vlm_psr, n_cols)
    if any_eureka:
        eureka_sr = _pad_matrix(eureka_sr, n_cols)
        eureka_psr = _pad_matrix(eureka_psr, n_cols)

    # ── Summary tables ─────────────────────────────────────────────────────────
    def _print_table(label, mat):
        print(f"\n{label}")
        print(f"{'Sample':<8} {'Mean SR':>8} {'± Std':>7} {'N eval':>7}")
        for col in range(mat.shape[1]):
            col_data = mat[:, col]
            n_eval = int(np.sum(~np.isnan(col_data)))
            mean_v = float(np.nanmean(col_data)) if n_eval else float("nan")
            std_v = float(np.nanstd(col_data)) if n_eval else float("nan")
            print(f"{col:<8} {mean_v:>7.1f}% {std_v:>6.1f}%  {n_eval:>5}")

    if any_vlm:
        _print_table("VLM Refinement", vlm_sr)
    if any_eureka:
        iter_label = "+".join(str(i) for i in eureka_iters)
        _print_table(f"Eureka (iter {iter_label})", eureka_sr)

    # ── Plot ───────────────────────────────────────────────────────────────────
    x = np.arange(n_cols)

    VLM_COLOR = "#1f77b4"
    EUREKA_COLOR = "#e377c2"

    def _cum_best_matrix(mat):
        return np.array([cumulative_best(mat[i]) for i in range(mat.shape[0])])

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".pdf"
    path_sr = f"{base}_success_rate{ext}"
    path_cum = f"{base}_cumulative_best{ext}"

    def _style_ax(ax):
        ax.grid(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=18)

    # ── Figure 1: per-sample success rate ─────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(10, 6))

    if any_vlm:
        n_vlm = vlm_sr.shape[0]
        for i in range(n_vlm):
            ax.plot(x, vlm_sr[i], color=VLM_COLOR, alpha=0.2, linewidth=1.5)
        mean_v = np.nanmean(vlm_sr, axis=0)
        std_v = np.nanstd(vlm_sr, axis=0)
        ax.plot(x, mean_v, color=VLM_COLOR, linewidth=3, label="VLM refinement")
        ax.fill_between(x, mean_v - std_v, mean_v + std_v, color=VLM_COLOR, alpha=0.1)

    if any_eureka:
        n_eureka = eureka_sr.shape[0]
        for i in range(n_eureka):
            ax.plot(x, eureka_sr[i], color=EUREKA_COLOR, alpha=0.2, linewidth=1.5)
        mean_e = np.nanmean(eureka_sr, axis=0)
        std_e = np.nanstd(eureka_sr, axis=0)
        ax.plot(x, mean_e, color=EUREKA_COLOR, linewidth=3, label="Eureka")
        ax.fill_between(x, mean_e - std_e, mean_e + std_e, color=EUREKA_COLOR, alpha=0.1)
        cum_start = 0
        for k, (it, n) in enumerate(zip(eureka_iters, eureka_iter_counts)):
            if k > 0:
                ax.axvline(x=cum_start - 0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
            mid_x = cum_start + (n - 1) / 2
            ax.text(mid_x, 97, f"iter {it}", ha="center", va="top", fontsize=14,
                    color="gray", transform=ax.get_xaxis_transform())
            cum_start += n

    ax.set_xlabel("Reward sample index", fontsize=28)
    ax.set_ylabel("Success rate (%)", fontsize=28)
    ax.set_title("Success Rate per Reward Sample", fontsize=36)
    ax.set_xticks(x)
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, loc="upper left", fontsize=18)
    _style_ax(ax)
    fig1.tight_layout()
    fig1.savefig(path_sr, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {path_sr}")
    if args.show:
        plt.show()
    plt.close(fig1)

    # ── Figure 2: cumulative best ──────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 6))

    if any_vlm:
        cum_vlm = _cum_best_matrix(vlm_sr)
        for i in range(cum_vlm.shape[0]):
            ax.plot(x, cum_vlm[i], color=VLM_COLOR, alpha=0.2, linewidth=1.5)
        mean_cv = np.nanmean(cum_vlm, axis=0)
        std_cv = np.nanstd(cum_vlm, axis=0)
        ax.plot(x, mean_cv, color=VLM_COLOR, linewidth=3, label="VLM refinement")
        ax.fill_between(x, mean_cv - std_cv, mean_cv + std_cv, color=VLM_COLOR, alpha=0.1)

    if any_eureka:
        cum_eureka = _cum_best_matrix(eureka_sr)
        for i in range(cum_eureka.shape[0]):
            ax.plot(x, cum_eureka[i], color=EUREKA_COLOR, alpha=0.2, linewidth=1.5)
        mean_ce = np.nanmean(cum_eureka, axis=0)
        std_ce = np.nanstd(cum_eureka, axis=0)
        ax.plot(x, mean_ce, color=EUREKA_COLOR, linewidth=3, label="Eureka")
        ax.fill_between(x, mean_ce - std_ce, mean_ce + std_ce, color=EUREKA_COLOR, alpha=0.1)
        cum_start = 0
        for k, (it, n) in enumerate(zip(eureka_iters, eureka_iter_counts)):
            if k > 0:
                ax.axvline(x=cum_start - 0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
            mid_x = cum_start + (n - 1) / 2
            ax.text(mid_x, 97, f"iter {it}", ha="center", va="top", fontsize=14,
                    color="gray", transform=ax.get_xaxis_transform())
            cum_start += n

    ax.set_xlabel("Reward sample index", fontsize=28)
    ax.set_ylabel("Best success rate so far (%)", fontsize=28)
    ax.set_title("Cumulative Best Success Rate", fontsize=36)
    ax.set_xticks(x)
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, loc="upper left", fontsize=18)
    _style_ax(ax)
    fig2.tight_layout()
    fig2.savefig(path_cum, dpi=300, bbox_inches="tight")
    print(f"Saved: {path_cum}")
    if args.show:
        plt.show()
    plt.close(fig2)
    print("Done.")


if __name__ == "__main__":
    main()
