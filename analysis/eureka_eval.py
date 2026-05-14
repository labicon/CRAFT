"""
Evaluate Eureka best-candidate checkpoints and save eval_results.pkl in each
rl_model_N_steps directory, matching the format used by CRAFT runs.

Single run:
    python -m analysis.eureka_eval \
        --dir logs/Eureka_GPT4o_good_param/go2seesaw/05-09_16-14

Batch (scan all Eureka runs under a root directory):
    python -m analysis.eureka_eval \
        --dir logs/Eureka_GPT4o_good_param

    python -m analysis.eureka_eval \
        --dir logs/Eureka_GPT4o

Task is auto-detected from eureka_summary.pkl. Runs are grouped by task so each
Isaac Gym environment is created only once per task.
"""

import os
import pickle
import re
import sys
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUPPORTED_TASKS = {"go2gate", "go2seesaw"}


# ──────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────

def find_eureka_runs(root):
    """
    Recursively find all Eureka run directories under *root*.
    A run directory is any directory that contains eureka/eureka_summary.pkl.
    Returns list of (run_dir, task) tuples.
    """
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        summary_path = os.path.join(dirpath, "eureka", "eureka_summary.pkl")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "rb") as f:
                    summary = pickle.load(f)
                task = summary.get("task")
                if task not in SUPPORTED_TASKS:
                    print(f"  Skipping unsupported task '{task}' in {dirpath}")
                    continue
                runs.append((dirpath, task, summary))
            except Exception as e:
                print(f"  Warning: could not read {summary_path}: {e}")
            # Don't descend into a run dir's subdirectories.
            dirnames[:] = [d for d in dirnames if d != "eureka"]
    return runs


def resolve_runs(dir_arg):
    """
    If dir_arg is itself an Eureka run dir, return it as a single-element list.
    Otherwise scan recursively for all Eureka runs underneath it.
    """
    summary_path = os.path.join(dir_arg, "eureka", "eureka_summary.pkl")
    if os.path.exists(summary_path):
        with open(summary_path, "rb") as f:
            summary = pickle.load(f)
        task = summary.get("task")
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task '{task}' in {dir_arg}")
        return [(dir_arg, task, summary)]
    return find_eureka_runs(dir_arg)


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────

def get_model_checkpoints(model_dir):
    pattern = re.compile(r"^rl_model_(\d+)_steps$")
    dirs = []
    for item in os.listdir(model_dir):
        m = pattern.match(item)
        if m:
            dirs.append((int(m.group(1)), os.path.join(model_dir, item)))
    dirs.sort(key=lambda x: x[0])
    return [path for _, path in dirs]


def best_model_dir(run_dir, summary):
    best = summary["best_candidate"]
    return os.path.join(
        run_dir, "eureka",
        f"iteration_{best['iteration']}",
        f"candidate_{best['candidate_idx']}",
        "model",
    )


# ──────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────

def run_eval(evaluator, agent, env, num_runs):
    all_results = {"reward": [], "total_success": [], "partial_success": []}
    obs = env.reset()
    episode_metrics = evaluator.reset_episode_metrics(obs, env.num_envs)
    metric_keys = list(episode_metrics.keys())
    for k in metric_keys:
        all_results[k] = []

    episode_length = np.zeros(env.num_envs, dtype=np.int32)
    total_reward = np.zeros(env.num_envs, dtype=np.float32)
    finished_runs = 0

    while finished_runs < num_runs:
        action, _ = agent.act(obs)
        new_obs, r, done, _ = env.step(action)
        episode_length += 1
        total_reward += np.sum(np.squeeze(r), axis=-1)
        episode_metrics = evaluator.update_step_metrics(obs, episode_metrics)
        dones = done[:, 0]

        if np.any(dones):
            valid_mask = episode_length > 100
            valid_dones = dones & valid_mask
            if np.sum(valid_dones) > 0:
                final_metrics = evaluator.get_final_metrics(obs, episode_metrics)
                finished_runs += int(np.sum(valid_dones))
                all_results["reward"].extend(total_reward[valid_dones].tolist())
                all_results["total_success"].extend(
                    final_metrics["total_success"][valid_dones].tolist()
                )
                all_results["partial_success"].extend(
                    final_metrics["partial_success"][valid_dones].tolist()
                )
                for k in metric_keys:
                    all_results[k].extend(final_metrics[k][valid_dones].tolist())
                print(f"  {finished_runs}/{num_runs} runs", end="\r")

            episode_length[dones] = 0
            total_reward[dones] = 0.0
            new_start = evaluator.reset_episode_metrics(new_obs, env.num_envs)
            for k in episode_metrics:
                if isinstance(episode_metrics[k], np.ndarray):
                    episode_metrics[k][dones] = new_start[k][dones]

        obs = new_obs

    print()
    actual_runs = len(all_results["reward"])
    eval_results = {
        "total_runs": actual_runs,
        "success_runs": int(np.sum(all_results["total_success"])),
        "partial_success_runs": int(np.sum(all_results["partial_success"])),
        "average_reward": float(np.mean(all_results["reward"])),
        "std_reward": float(np.std(all_results["reward"])),
    }
    for k in metric_keys:
        vals = all_results[k]
        eval_results[f"average_{k}"] = float(np.mean(vals)) if vals else 0.0
        eval_results[f"std_{k}"] = float(np.std(vals)) if vals else 0.0
    return eval_results


def eval_task_group(task, runs, num_runs, sim_device, graphics_device_id, force, last_only=False):
    """Evaluate all runs for a single task under one Isaac Gym environment."""
    from openrl_ws.utils import make_env, get_args
    from openrl_ws.eval import get_evaluator
    from mqe.envs.utils import custom_cfg
    from openrl.modules.common import PPONet
    from openrl.runners.common import PPOAgent

    env_args = get_args()
    env_args.task = task
    env_args.headless = True
    env_args.record_video = False
    env_args.seed = 0
    env_args.separate_policy = True
    env_args.sim_device = sim_device
    env_args.sim_device_id = int(sim_device.split(":")[-1]) if ":" in sim_device else 0
    env_args.rl_device = sim_device
    env_args.graphics_device_id = graphics_device_id

    env, _ = make_env(env_args, custom_cfg(env_args), single_agent=False)
    evaluator = get_evaluator(task, env)
    net = PPONet(env, cfg=env_args, device=env_args.rl_device)
    agent = PPOAgent(net)
    agent.set_env(env)

    for run_dir, summary in runs:
        best = summary["best_candidate"]
        model_dir = best_model_dir(run_dir, summary)
        print(f"\n[{task}] {run_dir}")
        print(f"  Best: iteration {best['iteration']}, candidate {best['candidate_idx']}")

        if not os.path.isdir(model_dir):
            print(f"  Model dir not found, skipping: {model_dir}")
            continue

        checkpoints = get_model_checkpoints(model_dir)
        if not checkpoints:
            print(f"  No checkpoints found in {model_dir}, skipping")
            continue

        if last_only:
            checkpoints = [checkpoints[-1]]

        if not force:
            pending = [c for c in checkpoints if not os.path.exists(os.path.join(c, "eval_results.pkl"))]
            skipped = len(checkpoints) - len(pending)
            if skipped:
                print(f"  Skipping {skipped} already-evaluated checkpoints (--force to re-run)")
            checkpoints = pending

        if not checkpoints:
            print("  All checkpoints already evaluated.")
            continue

        print(f"  Evaluating {len(checkpoints)} checkpoints, {num_runs} runs each")
        for checkpoint in checkpoints:
            print(f"  Checkpoint: {os.path.basename(checkpoint)}")
            agent.load(checkpoint)
            eval_results = run_eval(evaluator, agent, env, num_runs)
            out_path = os.path.join(checkpoint, "eval_results.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(eval_results, f)
            sr = eval_results["success_runs"] / eval_results["total_runs"] * 100
            print(f"  success={sr:.1f}%  saved -> {out_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Eureka best-candidate checkpoints and save eval_results.pkl. "
                    "Pass a single run directory or any ancestor directory to scan all runs beneath it."
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Single Eureka run dir (contains eureka/eureka_summary.pkl) "
             "or a root dir to scan recursively for all Eureka runs",
    )
    parser.add_argument("--num_runs", type=int, default=100, help="Eval episodes per checkpoint")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-run even if eval_results.pkl exists")
    parser.add_argument(
        "--best_only", action="store_true",
        help="Only evaluate the single best run per task (highest best_candidate score)",
    )
    args = parser.parse_args()

    dir_arg = args.dir
    num_runs = args.num_runs
    sim_device = args.sim_device
    graphics_device_id = args.graphics_device_id
    force = args.force
    best_only = args.best_only

    # Clear sys.argv before isaacgym/openrl parse args.
    del args, parser
    sys.argv = [sys.argv[0]]

    if not os.path.isdir(dir_arg):
        raise FileNotFoundError(f"Directory not found: {dir_arg}")

    runs = resolve_runs(dir_arg)
    if not runs:
        print(f"No Eureka runs found under {dir_arg}")
        return

    print(f"Found {len(runs)} Eureka run(s)")

    # Group by task so each Isaac Gym env is created only once per task.
    by_task = defaultdict(list)
    for run_dir, task, summary in runs:
        by_task[task].append((run_dir, summary))

    if best_only:
        for task in by_task:
            task_runs = by_task[task]
            best = max(task_runs, key=lambda x: x[1]["best_candidate"].get("score", float("-inf")))
            by_task[task] = [best]
            print(f"[{task}] best_only: selected {best[0]} (score={best[1]['best_candidate'].get('score')})")

    for task, task_runs in by_task.items():
        print(f"\n{'='*60}")
        print(f"Task: {task}  ({len(task_runs)} run(s))")
        print(f"{'='*60}")
        eval_task_group(task, task_runs, num_runs, sim_device, graphics_device_id, force, last_only=best_only)

    print("\nAll done.")


if __name__ == "__main__":
    main()
