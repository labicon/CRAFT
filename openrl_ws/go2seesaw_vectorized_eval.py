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

def eval_total_success(obs):
    agent_0_pos = obs[:, 0, 2:5]
    agent_1_pos = obs[:, 1, 2:5]

    platform_x = 7.7
    platform_z = 1.3

    agent_0_success = np.logical_and(agent_0_pos[:, 0] > platform_x, agent_0_pos[:, 2] > platform_z)
    agent_1_success = np.logical_and(agent_1_pos[:, 0] > platform_x, agent_1_pos[:, 2] > platform_z)

    success = np.logical_or(agent_0_success, agent_1_success)

    return success

def eval_partial_success(obs):
    agent_0_pos = obs[:, 0, 2:5]
    agent_1_pos = obs[:, 1, 2:5]

    platform_z = 0.6

    agent_0_success = (agent_0_pos[:, 2] > platform_z)
    agent_1_success = (agent_1_pos[:, 2] > platform_z)

    success = np.logical_or(agent_0_success, agent_1_success)

    return success

def maximum_height(obs):
    agent_0_height = obs[:, 0, 4]
    agent_1_height = obs[:, 1, 4]

    return np.maximum(agent_0_height, agent_1_height)

def target_distance(obs):
    target_pos = np.array([8.0, 0.0, 1.5])
    agent_0_pos = obs[:, 0, 2:5]
    agent_1_pos = obs[:, 1, 2:5]

    dist_0 = np.linalg.norm(agent_0_pos - target_pos, axis=-1)
    dist_1 = np.linalg.norm(agent_1_pos - target_pos, axis=-1)

    return np.minimum(dist_0, dist_1)

if __name__ == "__main__":
    args = get_args()
    env, _ = make_env(args, custom_cfg(args))

    net = PPONet(env, device="cuda:0")  # Create neural network.
    agent = PPOAgent(net)  # Initialize the agent.

    if getattr(args, "checkpoint") is not None:
        agent.load(args.checkpoint)

    # env.start_recording()
    agent.set_env(env)  # The agent requires an interactive environment.
    finished_runs = 0
    success_runs = 0
    partial_success_runs = 0
    reward_per_run = []
    maximum_height_per_run = []
    target_distance_per_run = []

    obs = env.reset()  # Initialize the environment to obtain initial observations and environmental information.
    episode_length = np.zeros(env.num_envs, dtype=np.int32)
    success_run = np.zeros(env.num_envs, dtype=bool)
    partial_success_run = np.zeros(env.num_envs, dtype=bool)
    total_reward = np.zeros(env.num_envs, dtype=np.float32)
    maximum_height_run = maximum_height(obs)
    target_distance_run = target_distance(obs)

    while finished_runs < 100:
        action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
        # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
        new_obs, r, done, info = env.step(action)
        episode_length += 1

        success_run = success_run | eval_total_success(obs)
        partial_success_run = partial_success_run | eval_partial_success(obs)
        maximum_height_run = np.maximum(maximum_height_run, maximum_height(obs))
        target_distance_run = np.minimum(target_distance_run, target_distance(obs))
        total_reward += np.sum(np.squeeze(r), axis=-1)

        done = done[:, 0]

        # If the episode length for an environment does not exceed 100 steps, it is considered an initialization issue and is not counted in the evaluation.
        initialization_issue = (episode_length <= 100) & done
        print("Initialization issues in this step:", np.sum(initialization_issue))

        finished_runs += np.sum(done & ~initialization_issue)
        reward_per_run.extend(total_reward[done & ~initialization_issue].tolist())
        maximum_height_per_run.extend(maximum_height_run[done & ~initialization_issue].tolist())
        target_distance_per_run.extend(target_distance_run[done & ~initialization_issue].tolist())

        print("Finished runs:", finished_runs)
        print("Successful runs:", np.sum(success_run[done & ~initialization_issue]))
        print("Partial successful runs:", np.sum(partial_success_run[done & ~initialization_issue]))
        success_runs += np.sum(success_run[done & ~initialization_issue])
        partial_success_runs += np.sum(partial_success_run[done & ~initialization_issue])
        
        # Reset values
        episode_length[done] = 0
        total_reward[done] = 0.0
        success_run[done] = False
        partial_success_run[done] = False
        maximum_height_run[done] = 0.0
        target_distance_run[done] = np.inf

        obs = new_obs  # Update the observation for the next step

    print(f"Total runs: {finished_runs}, Successful runs: {success_runs}, Success rate: {success_runs / finished_runs * 100:.2f}%")
    print(f"Partial success runs: {partial_success_runs}, Partial success rate: {partial_success_runs / finished_runs * 100:.2f}%")
    print(f"Average reward per run: {np.mean(reward_per_run):.2f}, Std: {np.std(reward_per_run):.2f}")

    # Save the evaluation results
    eval_results = {
        "total_runs": finished_runs,
        "success_runs": success_runs,
        "partial_success_runs": partial_success_runs,
        "average_reward": np.mean(reward_per_run) if reward_per_run else 0.0,
        "std_reward": np.std(reward_per_run) if reward_per_run else 0.0,
        "average_maximum_height": np.mean(maximum_height_per_run) if maximum_height_per_run else 0.0,
        "std_maximum_height": np.std(maximum_height_per_run) if maximum_height_per_run else 0.0,
        "average_target_distance": np.mean(target_distance_per_run) if target_distance_per_run else 0.0,
        "std_target_distance": np.std(target_distance_per_run) if target_distance_per_run else 0.0,
    }
    eval_results_path = os.path.join(args.checkpoint, "eval_results.pkl")
    with open(eval_results_path, 'wb') as f:
        pickle.dump(eval_results, f)