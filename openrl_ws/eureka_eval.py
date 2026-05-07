from openrl_ws.utils import make_env, get_args
from openrl_ws.curriculum_eval import (
    analyze_go2gate_trajectory,
    analyze_go2pushbox_trajectory,
    analyze_go2seesaw_trajectory,
)
from mqe.envs.utils import custom_cfg
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import argparse
import os
import pickle as pkl
import sys

import numpy as np


def eval_candidate(task, load_dir, output_dir, seed=0):
    from openrl_ws.test import save_images, save_video

    args = get_args()
    args.task = task
    args.headless = False
    args.record_video = True
    args.seed = seed
    args.separate_policy = True

    env, _ = make_env(args, custom_cfg(args), single_agent=False)
    net = PPONet(env, cfg=args, device=args.rl_device)
    agent = PPOAgent(net)
    agent.load(load_dir)
    agent.set_env(env)

    obs = env.reset(seed=seed)
    obs_buffer = [obs]
    reward_buffer = {"agent_0": 0.0, "agent_1": 0.0}

    while True:
        action, _ = agent.act(obs)
        obs, reward, done, info = env.step(action)
        obs_buffer.append(obs)
        reward_buffer["agent_0"] += reward[0, 0]
        reward_buffer["agent_1"] += reward[0, 1]
        if done[0, 0]:
            traj_dict, rew_dict = analyze_trajectory(task, obs_buffer, reward_buffer, env)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "traj_dict.pkl"), "wb") as file:
                pkl.dump(traj_dict, file)
            with open(os.path.join(output_dir, "rew_dict.pkl"), "wb") as file:
                pkl.dump(rew_dict, file)

            frames = env.get_complete_frames()
            video_array = np.concatenate(
                [np.expand_dims(frame, axis=0) for frame in frames],
                axis=0,
            ).swapaxes(1, 3).swapaxes(2, 3)
            save_video(video_array, 50, output_path=os.path.join(output_dir, f"rollout_{seed}.mp4"))
            save_images(video_array, output_dir=os.path.join(output_dir, "images"))
            break


def analyze_trajectory(task, obs_buffer, reward_buffer, env):
    if task == "go2gate":
        return analyze_go2gate_trajectory(obs_buffer, reward_buffer, env.target_pos)
    if task == "go2seesaw":
        return analyze_go2seesaw_trajectory(obs_buffer, reward_buffer, env)
    if task == "go2pushbox":
        return analyze_go2pushbox_trajectory(obs_buffer, reward_buffer, env)
    raise ValueError(f"Unsupported Eureka eval task: {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Eureka reward candidate")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--load_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task = args.task
    load_dir = args.load_dir
    output_dir = args.output_dir
    seed = args.seed

    del args, parser
    sys.argv = [sys.argv[0]]

    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Log directory {load_dir} does not exist.")
    eval_candidate(task, load_dir, output_dir, seed)
