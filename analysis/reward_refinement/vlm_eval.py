"""
Evaluate each VLM refinement sample's trained policy with full Isaac Gym simulation.

Saves eval_results.pkl in each models/sample_{i}/rl_model_N_steps/ directory,
matching the format used by eureka_eval.py and craft_eval.py.

Usage:
    python -m analysis.reward_refinement.vlm_eval \
        --dir analysis/reward_refinement/logs \
        [--num_runs 100] [--sim_device cuda:0] [--graphics_device_id 0] [--force]
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TASK = "go2gate"


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


def load_failure_reasons(log_dir):
    """Load failure_reasons from state.pkl; returns [] if missing."""
    state_path = os.path.join(log_dir, "state.pkl")
    if not os.path.exists(state_path):
        return []
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    return state.get("failure_reasons", [])


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


def run_eval(evaluator, agent, env, num_runs):
    """Vectorized evaluation loop; returns eval_results dict."""
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


def eval_runs(run_dirs, num_runs, sim_device, graphics_device_id, force):
    """Evaluate all samples across all VLM runs under a single Isaac Gym environment."""
    import isaacgym  # noqa: F401 — must be imported before torch
    from openrl_ws.utils import make_env, get_args
    from openrl_ws.eval import get_evaluator
    from mqe.envs.utils import custom_cfg
    from openrl.modules.common import PPONet
    from openrl.runners.common import PPOAgent

    env_args = get_args()
    env_args.task = TASK
    env_args.headless = True
    env_args.record_video = False
    env_args.seed = 0
    env_args.separate_policy = True
    env_args.sim_device = sim_device
    env_args.sim_device_id = int(sim_device.split(":")[-1]) if ":" in sim_device else 0
    env_args.rl_device = sim_device
    env_args.graphics_device_id = graphics_device_id

    env, _ = make_env(env_args, custom_cfg(env_args), single_agent=False)
    env.reset()
    evaluator = get_evaluator(TASK, env)
    net = PPONet(env, cfg=env_args, device=env_args.rl_device)
    agent = PPOAgent(net)
    agent.set_env(env)

    for log_dir in run_dirs:
        print(f"\n[{TASK}] {log_dir}")
        failure_reasons = load_failure_reasons(log_dir)
        n_samples = count_samples(log_dir)

        if n_samples == 0:
            print("  No model directories found, skipping.")
            continue

        for sample_idx in range(n_samples):
            failed_training = (
                sample_idx < len(failure_reasons)
                and failure_reasons[sample_idx] == "Training subprocess failed."
            )
            if failed_training:
                print(f"  sample_{sample_idx}: skipped (training failed)")
                continue

            checkpoint = get_sample_checkpoint(log_dir, sample_idx)
            if checkpoint is None:
                print(f"  sample_{sample_idx}: no checkpoint found, skipping")
                continue

            out_path = os.path.join(checkpoint, "eval_results.pkl")
            if not force and os.path.exists(out_path):
                try:
                    with open(out_path, "rb") as f:
                        cached = pickle.load(f)
                    sr = cached["success_runs"] / cached["total_runs"] * 100
                    print(f"  sample_{sample_idx}: already evaluated (success={sr:.1f}%), skipping")
                except Exception:
                    print(f"  sample_{sample_idx}: already evaluated, skipping")
                continue

            print(
                f"  sample_{sample_idx}: evaluating {num_runs} episodes"
                f" — {os.path.basename(checkpoint)}"
            )
            agent.load(checkpoint)
            eval_results = run_eval(evaluator, agent, env, num_runs)

            with open(out_path, "wb") as f:
                pickle.dump(eval_results, f)

            sr = eval_results["success_runs"] / eval_results["total_runs"] * 100
            psr = eval_results["partial_success_runs"] / eval_results["total_runs"] * 100
            print(
                f"  sample_{sample_idx}: success={sr:.1f}%  partial={psr:.1f}%"
                f"  -> {out_path}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM refinement sample checkpoints and save eval_results.pkl."
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Root directory with vlm_* subdirs, or a single vlm_* run directory.",
    )
    parser.add_argument("--num_runs", type=int, default=100, help="Eval episodes per checkpoint")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-run even if eval_results.pkl exists")
    args = parser.parse_args()

    dir_arg = os.path.abspath(args.dir)
    num_runs = args.num_runs
    sim_device = args.sim_device
    graphics_device_id = args.graphics_device_id
    force = args.force

    # Clear sys.argv before isaacgym/openrl parse args.
    del args, parser
    sys.argv = [sys.argv[0]]

    if not os.path.isdir(dir_arg):
        raise FileNotFoundError(f"Directory not found: {dir_arg}")

    run_dirs = find_vlm_runs(dir_arg)
    if not run_dirs:
        print(f"No VLM refinement run directories found under {dir_arg}")
        return
    print(f"Found {len(run_dirs)} VLM refinement run(s)")

    eval_runs(run_dirs, num_runs, sim_device, graphics_device_id, force)
    print("\nAll done.")


if __name__ == "__main__":
    main()
