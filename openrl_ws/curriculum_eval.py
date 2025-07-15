from openrl_ws.utils import make_env
# from openrl_ws.test import save_gif
from mqe.envs.utils import custom_cfg

from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import argparse
import pickle as pkl
import os, sys

def eval(load_dir):
    from openrl_ws.utils import get_args
    from openrl_ws.test import save_gif
    args = get_args()
    args.headless = True
    args.record_video = True
    env, _ = make_env(args, custom_cfg(args), single_agent=False)
    net = PPONet(env, cfg=args, device=args.rl_device)
    agent = PPOAgent(net)

    agent.load(load_dir)
    agent.set_env(env)
    obs = env.reset()
    obs_buffer = [obs]
    reward_buffer = {"agent_0": 0.0, "agent_1": 0.0}
    target_pos = env.target_pos

    while True:
        action, _ = agent.act(obs)
        obs, reward, done, info = env.step(action)
        obs_buffer.append(obs)
        reward_buffer["agent_0"] += reward[0, 0]
        reward_buffer["agent_1"] += reward[0, 1]
        if done[0, 0]:
            traj_dict, rew_dict = analyze_go1gate_trajectory(obs_buffer, reward_buffer, target_pos)
            print(f"Total reward for agent 0: {reward_buffer['agent_0']}, agent 1: {reward_buffer['agent_1']}")
            for key, value in traj_dict.items():
                print(f"{key}: {value}")
            # Save the trajectory and reward data
            save_path = os.path.join(load_dir, "traj_dict.pkl")
            with open(save_path, 'wb') as f:
                pkl.dump(traj_dict, f)
            save_path = os.path.join(load_dir, "rew_dict.pkl")
            with open(save_path, 'wb') as f:
                pkl.dump(rew_dict, f)

            # Save the video
            # frames = env.get_complete_frames()
            # video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            # print(video_array.shape)
            # print(np.mean(video_array))
            # save_gif(video_array, 1 / env.dt)

            break

def analyze_go1gate_trajectory(traj_buffer, rew_buffer, target_pos):
    import numpy as np
    stepsize = len(traj_buffer)
    max_samples = 20
    step = max(1, stepsize // max_samples)

    # Downsample the trajectory buffer
    sampled_traj_buffer = traj_buffer[::step]

    # traj_buffer is a list of obs arrays, each with shape (num_envs, num_agents, obs_dim)
    # Stack them to get a single array of shape (T, num_envs, num_agents, obs_dim)
    obs_trajectory = np.stack(sampled_traj_buffer, axis=0)

    # The observation for each agent is:
    # [agent_id(2), self_base_info(6), other_agent_base_info(6), gate_pos(2)]
    # self_base_info is [pos(3), rpy(3)]

    # We can extract all info from agent 0's perspective.
    # Assuming num_envs = 1, which is typical for evaluation.
    # Shape becomes (T, num_agents, obs_dim)
    obs_agent_0 = obs_trajectory[:, 0, 0, :] # Trajectory of obs for agent 0 in env 0

    # Extract XY positions
    # agent_0_pos is indices 2,3 (x,y)
    agent_0_pos = obs_agent_0[:, 2:4]
    # agent_1_pos is indices 8,9 (x,y)
    agent_1_pos = obs_agent_0[:, 8:10]
    # gate_pos is indices 14,15 (x,y)
    gate_pos = obs_agent_0[:, 14:16]

    target_pos = np.array(target_pos.reshape(-1, 2).detach().cpu().numpy())

    traj_dict = {
        "gate_pos": gate_pos.tolist()[0],
        "target_pos": target_pos.tolist(),
        "agent_0_pos": agent_0_pos.tolist(),
        "agent_1_pos": agent_1_pos.tolist(),
    }

    rew_dict = {
        "total_reward_agent_0": rew_buffer["agent_0"],
        "total_reward_agent_1": rew_buffer["agent_1"],
    }

    return traj_dict, rew_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--run_date", type=str, default=None, help="Run date for the experiment")
    parser.add_argument("--curriculum_task", type=str, default=None, help="Curriculum task name")
    parser.add_argument("--sample_idx", type=int, default=None, help="Sample index for the experiment")
    
    args = parser.parse_args()
    run_date = args.run_date
    curriculum_task = args.curriculum_task
    sample_idx = args.sample_idx

    del args, parser
    sys.argv = [sys.argv[0]]

    save_dir = os.path.join("runs", run_date, curriculum_task, f"sample_{sample_idx}")
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Log directory {save_dir} does not exist. Please check the run date and curriculum task.")
    exp_name = f"{run_date}_{curriculum_task}_sample_{sample_idx}"

    print(f"Evaluating model from {save_dir}")
    eval(load_dir=save_dir)