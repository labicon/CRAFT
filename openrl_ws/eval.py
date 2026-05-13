import isaacgym
from openrl_ws.utils import make_env, get_args, MATWrapper
from openrl_ws.test import save_video, save_gif, save_images
from mqe.envs.utils import custom_cfg

from openrl.envs.common import make
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import numpy as np
import os
import pickle
import argparse

# ==========================================
# Task Specific Evaluation Logic
# ==========================================

class GateEvaluator:
    def __init__(self, env):
        self.env = env
        self.gate_pos = env.gate_pos[:,0,:2].cpu().numpy()
        self.target_pos = env.target_pos[0,:2].cpu().numpy()

    def agentwise_success(self, obs):
        gate_pos = self.gate_pos
        agent_0_pos = obs[:, 0, 2:4]
        agent_1_pos = obs[:, 1, 2:4]

        agent_0_success = agent_0_pos[:, 0] > gate_pos[:, 0] + 0.25
        agent_1_success = agent_1_pos[:, 0] > gate_pos[:, 0] + 0.25

        return agent_0_success, agent_1_success

    def partial_success(self, obs):
        agent_0_success, agent_1_success = self.agentwise_success(obs)
        return np.logical_or(agent_0_success, agent_1_success)

    def total_success(self, obs):
        agent_0_success, agent_1_success = self.agentwise_success(obs)
        return np.logical_and(agent_0_success, agent_1_success)

    def distance_to_target(self, obs):
        target_pos = self.target_pos
        agent_0_pos = obs[:, 0, 2:4]
        agent_1_pos = obs[:, 1, 2:4]
        agent_0_dist = np.linalg.norm(agent_0_pos - target_pos, axis=-1)
        agent_1_dist = np.linalg.norm(agent_1_pos - target_pos, axis=-1)
        return np.maximum(agent_0_dist, agent_1_dist)

    def x_traversed(self, obs):
        agent_0_x = obs[:, 0, 2]
        agent_1_x = obs[:, 1, 2]
        return np.minimum(agent_0_x, agent_1_x)
    
    def reset_episode_metrics(self, obs, num_envs):
        # Return initial values for metrics tracked over the episode
        initial_min_dist = self.distance_to_target(obs)
        initial_x_traversed = self.x_traversed(obs)
        
        # Structure: {metric_name: (initial_values, reduction_function)}
        # reduction_function takes (accumulated_val, current_val)
        return {
            "minimum_distance": (initial_min_dist, np.minimum),
            "x_traversed": (initial_x_traversed, np.maximum)
        }

    def update_step_metrics(self, obs, current_metrics):
        # Calculate current step values
        current_min_dist = self.distance_to_target(obs)
        current_x = self.x_traversed(obs)
        
        # Update accumulated metrics
        new_metrics = {}
        new_metrics["minimum_distance"] = np.minimum(current_metrics["minimum_distance"], current_min_dist)
        new_metrics["x_traversed"] = np.maximum(current_metrics["x_traversed"], current_x)
        
        return new_metrics

    def get_final_metrics(self, obs, current_metrics, success_mask=None):
        # Gate task metrics are usually snapshot at end or accumulated max/min
        return {
            "total_success": self.total_success(obs),
            "partial_success": self.partial_success(obs),
            "minimum_distance": current_metrics["minimum_distance"],
            "x_traversed": current_metrics["x_traversed"]
        }


class SeesawEvaluator:
    def __init__(self, env):
        self.env = env
        self.target_pos = np.array([8.0, 0.0, 1.5])
        self.platform_x = 7.7
        self.platform_z = 1.3
        self.partial_platform_z = 0.6

    def eval_total_success(self, obs):
        agent_0_pos = obs[:, 0, 2:5]
        agent_1_pos = obs[:, 1, 2:5]

        agent_0_success = np.logical_and(agent_0_pos[:, 0] > self.platform_x, agent_0_pos[:, 2] > self.platform_z)
        agent_1_success = np.logical_and(agent_1_pos[:, 0] > self.platform_x, agent_1_pos[:, 2] > self.platform_z)

        success = np.logical_or(agent_0_success, agent_1_success)
        return success

    def eval_partial_success(self, obs):
        agent_0_pos = obs[:, 0, 2:5]
        agent_1_pos = obs[:, 1, 2:5]

        agent_0_success = (agent_0_pos[:, 2] > self.partial_platform_z)
        agent_1_success = (agent_1_pos[:, 2] > self.partial_platform_z)

        success = np.logical_or(agent_0_success, agent_1_success)
        return success

    def maximum_height(self, obs):
        agent_0_height = obs[:, 0, 4]
        agent_1_height = obs[:, 1, 4]
        return np.maximum(agent_0_height, agent_1_height)

    def target_distance(self, obs):
        agent_0_pos = obs[:, 0, 2:5]
        agent_1_pos = obs[:, 1, 2:5]

        dist_0 = np.linalg.norm(agent_0_pos - self.target_pos, axis=-1)
        dist_1 = np.linalg.norm(agent_1_pos - self.target_pos, axis=-1)

        return np.minimum(dist_0, dist_1)

    def reset_episode_metrics(self, obs, num_envs):
        initial_max_height = self.maximum_height(obs)
        initial_target_dist = self.target_distance(obs)
        
        # Seesaw also tracks "anytime success" during the episode
        self.anytime_total_success = np.zeros(num_envs, dtype=bool)
        self.anytime_partial_success = np.zeros(num_envs, dtype=bool)

        return {
            "maximum_height": initial_max_height,
            "target_distance": initial_target_dist
        }

    def update_step_metrics(self, obs, current_metrics):
        # Update anytime success flags (stateful)
        self.anytime_total_success = self.anytime_total_success | self.eval_total_success(obs)
        self.anytime_partial_success = self.anytime_partial_success | self.eval_partial_success(obs)

        current_max_height = self.maximum_height(obs)
        current_target_dist = self.target_distance(obs)
        
        new_metrics = {}
        new_metrics["maximum_height"] = np.maximum(current_metrics["maximum_height"], current_max_height)
        new_metrics["target_distance"] = np.minimum(current_metrics["target_distance"], current_target_dist)
        
        return new_metrics

    def get_final_metrics(self, obs, current_metrics, success_mask=None):
        return {
            "total_success": self.anytime_total_success,
            "partial_success": self.anytime_partial_success,
            "maximum_height": current_metrics["maximum_height"],
            "target_distance": current_metrics["target_distance"]
        }


def get_evaluator(task_name, env):
    if "go2gate" in task_name:
        return GateEvaluator(env)
    elif "go2seesaw" in task_name:
        return SeesawEvaluator(env)
    else:
        raise ValueError(f"Unknown task for evaluation: {task_name}")

if __name__ == "__main__":
    args = get_args()
    
    # Ensure vectorized settings
    args.headless = True # Often desirable for fast eval but user might want video
    
    env, _ = make_env(args, custom_cfg(args))
    
    # Initialize evaluator based on task
    try:
        evaluator = get_evaluator(args.task, env)
    except ValueError as e:
        print(e)
        exit(1)

    net = PPONet(env, device="cuda:0")  # Create neural network.
    agent = PPOAgent(net)  # Initialize the agent.

    if getattr(args, "checkpoint") is not None:
        agent.load(args.checkpoint)

    agent.set_env(env)  # The agent requires an interactive environment.
    
    target_finished_runs = 100
    finished_runs = 0
    
    # Metrics storage
    all_results = {
        "reward": [],
        "total_success": [],
        "partial_success": []
    }
    
    # Initialize environment
    obs = env.reset()
    
    # State tracking
    episode_length = np.zeros(env.num_envs, dtype=np.int32)
    total_reward = np.zeros(env.num_envs, dtype=np.float32)
    
    # Initialize task-specific metrics
    episode_metrics = evaluator.reset_episode_metrics(obs, env.num_envs)
    
    # Ensure keys exist in storage
    metric_keys = list(episode_metrics.keys())
    for k in metric_keys:
        all_results[k] = []

    print(f"Starting vectorized evaluation for task: {args.task}")
    print(f"Target runs: {target_finished_runs}")

    while finished_runs < target_finished_runs:
        action, _ = agent.act(obs)
        new_obs, r, done, info = env.step(action)
        
        episode_length += 1
        total_reward += np.sum(np.squeeze(r), axis=-1)
        
        # Update metrics for current step
        episode_metrics = evaluator.update_step_metrics(obs, episode_metrics)
        
        # Check completion
        dones = done[:, 0]
        
        if np.any(dones):
            # Filter valid runs (ignore initialization issues < 100 steps)
            valid_mask = episode_length > 100
            valid_dones = dones & valid_mask
            
            num_valid = np.sum(valid_dones)
            
            if num_valid > 0:
                # Get final metrics for the completed episodes
                final_metrics = evaluator.get_final_metrics(obs, episode_metrics)
                
                # Store results
                finished_runs += num_valid
                
                # Basic metrics
                all_results["reward"].extend(total_reward[valid_dones].tolist())
                all_results["total_success"].extend(final_metrics["total_success"][valid_dones].tolist())
                all_results["partial_success"].extend(final_metrics["partial_success"][valid_dones].tolist())
                
                # Extra metrics
                for k in metric_keys:
                    all_results[k].extend(final_metrics[k][valid_dones].tolist())
                
                print(f"Finished {finished_runs}/{target_finished_runs} runs.")
                
                # Reset tracking for done environments
                episode_length[dones] = 0
                total_reward[dones] = 0.0
                
                # Reset metrics for done environments
                # We need to re-initialize based on new_obs for these envs
                # But evaluator.reset returns for all envs. We selectively update.
                new_start_metrics = evaluator.reset_episode_metrics(new_obs, env.num_envs)
                for k in episode_metrics:
                    # Only update the done indices
                    if isinstance(episode_metrics[k], np.ndarray):
                        episode_metrics[k][dones] = new_start_metrics[k][dones]
                        
            # Just reset lengths for invalid dones too to prevent infinite growth
            invalid_dones = dones & ~valid_mask
            if np.any(invalid_dones):
                episode_length[invalid_dones] = 0
                total_reward[invalid_dones] = 0.0
                new_start_metrics = evaluator.reset_episode_metrics(new_obs, env.num_envs)
                for k in episode_metrics:
                    if isinstance(episode_metrics[k], np.ndarray):
                        episode_metrics[k][invalid_dones] = new_start_metrics[k][invalid_dones]
                        
        obs = new_obs

    # =================
    # Summary & Save
    # =================
    
    # Trim to target number if user requested exact number
    # though with vectorized it usually overshoots slightly
    # all_results = {k: v[:target_finished_runs] for k, v in all_results.items()}
    actual_runs = len(all_results["reward"])

    success_rate = np.mean(all_results["total_success"]) * 100
    partial_success_rate = np.mean(all_results["partial_success"]) * 100
    avg_reward = np.mean(all_results["reward"])
    std_reward = np.std(all_results["reward"])

    print(f"\nEvaluation Complete:")
    print(f"Total runs: {actual_runs}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Partial success rate: {partial_success_rate:.2f}%")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")

    # Prepare pickle dictionary
    eval_results = {
        "total_runs": actual_runs,
        "success_runs": np.sum(all_results["total_success"]),
        "partial_success_runs": np.sum(all_results["partial_success"]),
        "average_reward": avg_reward,
        "std_reward": std_reward,
    }
    
    # Add extra metrics statistics
    for k in metric_keys:
        vals = all_results[k]
        eval_results[f"average_{k}"] = np.mean(vals) if vals else 0.0
        eval_results[f"std_{k}"] = np.std(vals) if vals else 0.0
        print(f"Average {k}: {eval_results[f'average_{k}']:.3f} ± {eval_results[f'std_{k}']:.3f}")

    eval_results_path = os.path.join(args.checkpoint, "eval_results.pkl")
    with open(eval_results_path, 'wb') as f:
        pickle.dump(eval_results, f)
    print(f"Results saved to {eval_results_path}")
