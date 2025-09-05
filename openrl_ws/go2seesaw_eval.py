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
    agent_0_pos = obs[0, 0, 2:5]
    agent_1_pos = obs[0, 1, 2:5]

    platform_x = 7.7
    platform_z = 1.3

    agent_0_success = (agent_0_pos[0] > platform_x) and (agent_0_pos[2] > platform_z)
    agent_1_success = (agent_1_pos[0] > platform_x) and (agent_1_pos[2] > platform_z)

    success = agent_0_success or agent_1_success

    return success

def eval_partial_success(obs):
    agent_0_pos = obs[0, 0, 2:5]
    agent_1_pos = obs[0, 1, 2:5]

    platform_z = 0.6

    agent_0_success = (agent_0_pos[2] > platform_z)
    agent_1_success = (agent_1_pos[2] > platform_z)

    success = agent_0_success or agent_1_success

    return success

def maximum_height(obs):
    agent_0_height = obs[0, 0, 4]
    agent_1_height = obs[0, 1, 4]

    return max(agent_0_height, agent_1_height)

def target_distance(obs):
    target_pos = np.array([8.0, 0.0, 1.5])
    agent_0_pos = obs[0, 0, 2:5]
    agent_1_pos = obs[0, 1, 2:5]

    dist_0 = np.linalg.norm(agent_0_pos - target_pos)
    dist_1 = np.linalg.norm(agent_1_pos - target_pos)

    return min(dist_0, dist_1)

if __name__ == "__main__":
    args = get_args()
    env, _ = make_env(args, custom_cfg(args))

    net = PPONet(env, device="cuda")  # Create neural network.
    agent = PPOAgent(net)  # Initialize the agent.

    if getattr(args, "checkpoint") is not None:
        agent.load(args.checkpoint)

    # env.start_recording()
    agent.set_env(env)  # The agent requires an interactive environment.
    eval_runs = 0
    success_runs = 0
    partial_success_runs = 0
    reward_per_run = []
    maximum_height_per_run = []
    target_distance_per_run = []
    while eval_runs < 100:
        obs = env.reset(seed=np.random.randint(0, 10000))  # Initialize the environment to obtain initial observations and environmental information.
        episode_length = 0
        success_run = False
        partial_success_run = False
        total_reward = 0.0
        maximum_height_run = maximum_height(obs)
        target_distance_run = target_distance(obs)
        while True:
            action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
            # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
            new_obs, r, done, info = env.step(action)
            episode_length += 1
            total_reward += np.sum(r)
            total_success = eval_total_success(new_obs)
            partial_success = eval_partial_success(new_obs)
            maximum_height_run = max(maximum_height_run, maximum_height(new_obs))
            target_distance_run = min(target_distance_run, target_distance(new_obs))
            success_run = success_run or total_success
            partial_success_run = partial_success_run or partial_success
            if done[0, 0]:
                print(f"Run {eval_runs} completed.")
                if episode_length <= 30:
                    print("Terminated due to initialization issue")
                    print("Do not include in the analysis")
                    break
                eval_runs += 1
                if np.linalg.norm(obs[0, 0, 2:4] - obs[0, 1, 2:4]) < 0.51:
                    print("Terminated due to close proximity")
                else:
                    print("Terminated due to other reasons")

                reward_per_run.append(total_reward)
                maximum_height_per_run.append(maximum_height_run)
                target_distance_per_run.append(target_distance_run)

                if success_run:
                    print("Total success!")
                    success_runs += 1
                elif partial_success_run:
                    print("Partial success!")
                    partial_success_runs += 1

                if env.cfg.env.record_video:
                    frames = env.get_complete_frames()
                    video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
                    eval_dir = os.path.join(args.checkpoint, f"eval_run_{eval_runs}.mp4")
                    save_video(video_array, 50, eval_dir)
                break
            obs = new_obs  # Update the observation for the next step
                
    print(f"Total runs: {eval_runs}, Successful runs: {success_runs}, Success rate: {success_runs / eval_runs * 100:.2f}%")
    print(f"Partial success runs: {partial_success_runs}, Partial success rate: {partial_success_runs / eval_runs * 100:.2f}%")
    print(f"Average reward per run: {np.mean(reward_per_run):.2f}, Std: {np.std(reward_per_run):.2f}")

    # Save the evaluation results
    eval_results = {
        "total_runs": eval_runs,
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