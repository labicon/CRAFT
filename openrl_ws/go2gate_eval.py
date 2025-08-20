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
        gate_pos = obs[0, 0, -2:]
    agent_0_pos = obs[0, 0, 2:4]
    agent_1_pos = obs[0, 1, 2:4]

    agent_0_success = agent_0_pos[0] > gate_pos[0] + 0.25
    agent_1_success = agent_1_pos[0] > gate_pos[0] + 0.25

    return agent_0_success, agent_1_success

def partial_success(obs, gate_pos=None):
    agent_0_success, agent_1_success = agentwise_success(obs, gate_pos)
    return agent_0_success or agent_1_success

def total_success(obs, gate_pos=None):
    agent_0_success, agent_1_success = agentwise_success(obs, gate_pos)
    return agent_0_success and agent_1_success

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
    while eval_runs < 2:
        obs = env.reset(seed=eval_runs)  # Initialize the environment to obtain initial observations and environmental information.
        gate_pos = env.gate_pos[0,0,:2]
        total_reward = 0.0
        while True:
            action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
            # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
            new_obs, r, done, info = env.step(action)
            total_reward += np.sum(r)
            if done[0, 0]:
                print(f"Run {eval_runs} completed.")
                eval_runs += 1
                reward_per_run.append(total_reward)
                print("Observation:", obs)
                if np.linalg.norm(obs[0, 0, 2:4] - obs[0, 1, 2:4]) < 0.51:
                    print("Terminated due to close proximity")
                else:
                    print("Terminated due to other reasons")

                if total_success(obs, gate_pos):
                    print("Total success!")
                    success_runs += 1
                elif partial_success(obs, gate_pos):
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
    }
    eval_results_path = os.path.join(args.checkpoint, "eval_results.pkl")
    with open(eval_results_path, 'wb') as f:
        pickle.dump(eval_results, f)