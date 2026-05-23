"""
Evaluate CRAFT best-policy checkpoints and save eval_results.pkl.

For each run in the given directory:
  - If any sample in the final task has "Decision: Success", use the final
    checkpoint of that sample.
  - Otherwise, parse the task-level [TaskName].md decision file for
    "Decision: Experiment N" and use sample_N's final checkpoint.

Single task root:
    python -m analysis.craft_eval --dir logs/updated_go2gate
    python -m analysis.craft_eval --dir logs/updated_go2seesaw

Multiple roots in one call:
    python -m analysis.craft_eval --dir logs/updated_go2gate --dir logs/updated_go2seesaw

Task is auto-detected from the directory path ("go2gate" or "go2seesaw").
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
# Discovery helpers
# ──────────────────────────────────────────────

def detect_task(dir_path):
    """Detect task name from any segment of the directory path."""
    for segment in dir_path.replace("\\", "/").split("/"):
        for task in SUPPORTED_TASKS:
            if task in segment:
                return task
    return None


def find_craft_runs(root):
    """Return sorted list of timestamp-based run directories under root."""
    runs = []
    for item in sorted(os.listdir(root)):
        if re.match(r"^\d{2}-\d{2}_\d{2}-\d{2}", item):
            item_path = os.path.join(root, item)
            if os.path.isdir(item_path):
                runs.append(item_path)
    return runs


def find_final_task_dir(run_dir):
    """Return (task_name, task_path) for the highest-numbered task directory."""
    task_dirs = []
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path) and re.match(r"^\d+_", item):
            try:
                num = int(item.split("_")[0])
                task_dirs.append((num, item, item_path))
            except ValueError:
                continue
    if not task_dirs:
        return None, None
    task_dirs.sort(key=lambda x: x[0])
    _, task_name, task_path = task_dirs[-1]
    return task_name, task_path


def find_successful_sample(task_dir):
    """Return path of the first sample_N dir whose evaluation_answer.md says Success."""
    for item in sorted(os.listdir(task_dir)):
        if not re.match(r"^sample_\d+$", item):
            continue
        eval_path = os.path.join(task_dir, item, "evaluation_answer.md")
        if not os.path.exists(eval_path):
            continue
        with open(eval_path, encoding="utf-8") as f:
            content = f.read()
        if re.search(r"^Decision:\s*Success", content, re.MULTILINE | re.IGNORECASE):
            return os.path.join(task_dir, item)
    return None


def find_llm_selected_sample(run_dir, task_name, task_dir):
    """Parse task-level [task_name].md for 'Decision: Experiment N', return sample_N path."""
    decision_file = os.path.join(run_dir, f"{task_name}.md")
    if not os.path.exists(decision_file):
        return None
    with open(decision_file, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"Decision:\s*Experiment\s*(\d+)", content, re.IGNORECASE)
    if not m:
        return None
    exp_num = int(m.group(1))
    sample_dir = os.path.join(task_dir, f"sample_{exp_num}")
    if os.path.isdir(sample_dir):
        return sample_dir
    return None


def resolve_best_sample(run_dir):
    """
    Return (sample_dir, mode) where mode is 'success' or 'llm'.
    Returns (None, None) if neither can be resolved.
    """
    task_name, task_dir = find_final_task_dir(run_dir)
    if task_dir is None:
        print(f"  No task directories found in {run_dir}")
        return None, None

    success_sample = find_successful_sample(task_dir)
    if success_sample:
        return success_sample, "success"

    llm_sample = find_llm_selected_sample(run_dir, task_name, task_dir)
    if llm_sample:
        return llm_sample, "llm"

    print(f"  Could not resolve best sample in {run_dir} (task: {task_name})")
    return None, None


def get_final_checkpoint(model_dir):
    """Return the rl_model_*_steps directory with the highest step count."""
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


# ──────────────────────────────────────────────
# Per-task evaluation driver
# ──────────────────────────────────────────────

def eval_task_group(task, run_dirs, num_runs, sim_device, graphics_device_id, force):
    """Evaluate CRAFT best-policy for all runs of one task under a single Isaac Gym env."""
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
    env.reset()  # populate env attributes (e.g. gate_pos) before constructing evaluator
    evaluator = get_evaluator(task, env)
    net = PPONet(env, cfg=env_args, device=env_args.rl_device)
    agent = PPOAgent(net)
    agent.set_env(env)

    for run_dir in run_dirs:
        print(f"\n[{task}] {run_dir}")

        sample_dir, mode = resolve_best_sample(run_dir)
        if sample_dir is None:
            continue

        model_dir = os.path.join(sample_dir, "model")
        if not os.path.isdir(model_dir):
            print(f"  Model dir not found: {model_dir}")
            continue

        checkpoint = get_final_checkpoint(model_dir)
        if checkpoint is None:
            print(f"  No rl_model_*_steps checkpoints found in {model_dir}")
            continue

        out_path = os.path.join(checkpoint, "eval_results.pkl")
        print(f"  Selection: {mode}  →  {os.path.relpath(checkpoint)}")

        if not force and os.path.exists(out_path):
            print(f"  Already evaluated, skipping (--force to re-run)")
            continue

        print(f"  Evaluating {num_runs} episodes ...")
        agent.load(checkpoint)
        eval_results = run_eval(evaluator, agent, env, num_runs)

        with open(out_path, "wb") as f:
            pickle.dump(eval_results, f)

        sr = eval_results["success_runs"] / eval_results["total_runs"] * 100
        psr = eval_results["partial_success_runs"] / eval_results["total_runs"] * 100
        print(f"  success={sr:.1f}%  partial={psr:.1f}%  saved -> {out_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CRAFT best-policy checkpoints. "
                    "Pass one or more --dir arguments pointing to task root directories "
                    "(e.g. logs/updated_go2gate). Task is auto-detected from the path."
    )
    parser.add_argument(
        "--dir", type=str, action="append", required=True,
        help="Root directory containing timestamp-based CRAFT run dirs. "
             "Can be repeated to process multiple tasks in one call.",
    )
    parser.add_argument("--num_runs", type=int, default=100, help="Eval episodes per checkpoint")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-run even if eval_results.pkl exists")
    args = parser.parse_args()

    num_runs = args.num_runs
    sim_device = args.sim_device
    graphics_device_id = args.graphics_device_id
    force = args.force
    dirs = args.dir

    # Clear sys.argv before isaacgym/openrl parse args.
    del args, parser
    sys.argv = [sys.argv[0]]

    by_task = defaultdict(list)
    for d in dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            print(f"Directory not found, skipping: {d}")
            continue
        task = detect_task(d)
        if task not in SUPPORTED_TASKS:
            print(f"Cannot detect task from path '{d}', skipping (expected go2gate or go2seesaw)")
            continue
        runs = find_craft_runs(d)
        if not runs:
            print(f"No timestamp-based run dirs found in {d}")
            continue
        print(f"Found {len(runs)} run(s) for task '{task}' in {d}")
        by_task[task].extend(runs)

    if not by_task:
        print("No valid runs found.")
        return

    for task, run_dirs in by_task.items():
        print(f"\n{'='*60}")
        print(f"Task: {task}  ({len(run_dirs)} run(s))")
        print(f"{'='*60}")
        eval_task_group(task, run_dirs, num_runs, sim_device, graphics_device_id, force)

    print("\nAll done.")


if __name__ == "__main__":
    main()
