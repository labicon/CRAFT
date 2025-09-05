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

def agentwise_success(obs, gate_pos=None):
    if gate_pos is None:
        gate_pos = obs[:, 0, -2:]
    agent_0_pos = obs[:, 0, 2:4]
    agent_1_pos = obs[:, 1, 2:4]

    agent_0_success = agent_0_pos[:, 0] > gate_pos[:, 0] + 0.25
    agent_1_success = agent_1_pos[:, 0] > gate_pos[:, 0] + 0.25

    return agent_0_success, agent_1_success

def partial_success(obs, gate_pos=None):
    agent_0_success, agent_1_success = agentwise_success(obs, gate_pos)
    return np.logical_or(agent_0_success, agent_1_success)

def total_success(obs, gate_pos=None):
    agent_0_success, agent_1_success = agentwise_success(obs, gate_pos)
    return np.logical_and(agent_0_success, agent_1_success)

def distance_to_target(obs, target_pos=None):
    if target_pos is None:
        target_pos = np.array([5.0, 0.0])
    agent_0_pos = obs[:, 0, 2:4]
    agent_1_pos = obs[:, 1, 2:4]
    agent_0_dist = np.linalg.norm(agent_0_pos - target_pos, axis=-1)
    agent_1_dist = np.linalg.norm(agent_1_pos - target_pos, axis=-1)
    return np.maximum(agent_0_dist, agent_1_dist)

def x_traversed(obs):
    agent_0_x = obs[:, 0, 2]
    agent_1_x = obs[:, 1, 2]
    return np.minimum(agent_0_x, agent_1_x)

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
    minimum_distance_per_run = []
    x_traversed_per_run = []

    obs = env.reset(seed=np.random.randint(0, 10000))  # Initialize the environment to obtain initial observations and environmental information.
    episode_length = np.zeros(env.num_envs, dtype=np.int32)
    done_flags = np.zeros(env.num_envs, dtype=bool)
    total_reward = np.zeros(env.num_envs, dtype=np.float32)
    gate_pos = env.gate_pos[:,0,:2]
    target_pos = env.target_pos[0,:2].cpu().numpy()
    minimum_distance = distance_to_target(obs, target_pos)
    x_traversed_distance = x_traversed(obs)
    while finished_runs < 100:
        action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
        # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
        new_obs, r, done, info = env.step(action)
        episode_length += 1
        minimum_distance = np.minimum(minimum_distance, distance_to_target(new_obs, target_pos))
        x_traversed_distance = np.maximum(x_traversed_distance, x_traversed(new_obs))
        total_reward += np.sum(r, axis=-1)

        done = done[:, 0]
        print(f"In step {episode_length}, {np.sum(done)} environments are done.")

        # If the episode length for an environment does not exceed 30 steps, it is considered an initialization issue and is not counted in the evaluation.
        initialization_issue = (episode_length <= 30) & done
        episode_length[initialization_issue] = 0
        done_flags[initialization_issue] = False
        total_reward[initialization_issue] = 0.0
        minimum_distance[initialization_issue] = np.inf
        x_traversed_distance[initialization_issue] = 0.0

        finished_runs += np.sum(done & ~initialization_issue)
        reward_per_run.extend(total_reward[done & ~initialization_issue].tolist())
        minimum_distance_per_run.extend(minimum_distance[done & ~initialization_issue].tolist())
        x_traversed_per_run.extend(x_traversed_distance[done & ~initialization_issue].tolist())

        success_runs += np.sum(total_success(new_obs[done & ~initialization_issue], gate_pos[done & ~initialization_issue]))
        partial_success_runs += np.sum(partial_success(new_obs[done & ~initialization_issue], gate_pos[done & ~initialization_issue]))

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
        "average_minimum_distance": np.mean(minimum_distance_per_run) if minimum_distance_per_run else 0.0,
        "std_minimum_distance": np.std(minimum_distance_per_run) if minimum_distance_per_run else 0.0,
        "average_x_traversed": np.mean(x_traversed_per_run) if x_traversed_per_run else 0.0,
        "std_x_traversed": np.std(x_traversed_per_run) if x_traversed_per_run else 0.0,
    }
    eval_results_path = os.path.join(args.checkpoint, "eval_results.pkl")
    with open(eval_results_path, 'wb') as f:
        pickle.dump(eval_results, f)