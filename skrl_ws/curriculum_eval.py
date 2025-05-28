import isaacgym

import torch
import torch.nn as nn

import os, sys
import re
import argparse
from skrl_ws.utils import create_env
import pickle
import tqdm
import imageio

# import the skrl components to build the RL system
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

from datetime import datetime
from skrl_ws.utils import Policy, Value, create_env, get_cfg, find_last_checkpoint, save_gif

def eval(env, cfg, save_dir):
    device = env.device

    # instantiate memories as rollout buffer (any memory can be used for this)
    memories = {}
    for agent_name in env.possible_agents:
        memories[agent_name] = RandomMemory(memory_size=12, num_envs=env.num_envs, device=device)

    models = {}
    # shared_policy = Policy(env.observation_space(env.possible_agents[0]), env.action_space(env.possible_agents[0]), device)
    # shared_value = Value(env.state_space(env.possible_agents[0]), env.action_space(env.possible_agents[0]), device)
    for agent_name in env.possible_agents:
        models[agent_name] = {}
        models[agent_name]["policy"] = Policy(env.observation_space(env.possible_agents[0]), env.action_space(env.possible_agents[0]), device)
        models[agent_name]["value"] = Value(env.state_space(env.possible_agents[0]), env.action_space(env.possible_agents[0]), device)

    agent = MAPPO(possible_agents=env.possible_agents,
                models=models,
                memories=memories,
                cfg=cfg,
                observation_spaces=env.observation_spaces,
                action_spaces=env.action_spaces,
                device=device,
                shared_observation_spaces=env.state_spaces)

    checkpoint_path = find_last_checkpoint(save_dir)
    print(f"Loading checkpoint from {checkpoint_path}")
    agent.load(path=checkpoint_path)

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start evaluation
    # rew_buffer, traj_buffer, frames = multi_go2_rollout(trainer)
    rew_buffer, traj_buffer = multi_go2_rollout(trainer)
    traj_dict, rew_dict = analyze_trajectory(traj_buffer, rew_buffer)

    # Save the trajectory buffer to a file
    with open(os.path.join(save_dir, "traj_dict.pkl"), "wb") as f:
        pickle.dump(traj_dict, f)
    # Save the reward buffer to a file
    with open(os.path.join(save_dir, "rew_dict.pkl"), "wb") as f:
        pickle.dump(rew_dict, f)
    # save_gif(frames, fps=1/env._env.dt, save_dir=save_dir)

def analyze_trajectory(traj_buffer, rew_buffer):
    stepsize = len(traj_buffer["agent_0_pos"])
    max_samples = 20
    step = max(1, stepsize // max_samples + 1)  # Ensure at least step size of 1

    rew_dict = {}
    traj_dict = {}

    total_reward = rew_buffer["agent_0"]
    eval_idx = 0

    traj_dict["target_pos"] = traj_buffer["target_pos"]
    traj_dict["obstacle_1_pos"] = traj_buffer["obstacle_1_pos"]
    traj_dict["obstacle_2_pos"] = traj_buffer["obstacle_2_pos"]
    traj_dict["agent_0_pos"] = torch.cat(traj_buffer["agent_0_pos"], dim=0)[::step]
    traj_dict["agent_1_pos"] = torch.cat(traj_buffer["agent_1_pos"], dim=0)[::step]

    rew_dict["total_reward"] = total_reward

    return traj_dict, rew_dict

def multi_go2_rollout(trainer: SequentialTrainer, max_timesteps=500):
    env = trainer.env
    # env._env.start_recording()
    agent = trainer.agents

    states, infos = env.reset()
    shared_states = env.state()

    # Initialize the rollout buffer
    traj_buffer = {}
    traj_buffer["target_pos"] = shared_states[:, :2]
    traj_buffer["obstacle_1_pos"] = shared_states[:, 2:4]
    traj_buffer["obstacle_2_pos"] = shared_states[:, 4:6]
    traj_buffer["agent_0_pos"] = [shared_states[:, 6:8]]
    traj_buffer["agent_0_yaw"] = [shared_states[:, 8]]
    traj_buffer["agent_1_pos"] = [shared_states[:, 9:11]]
    traj_buffer["agent_1_yaw"] = [shared_states[:, 11]]
    episode_length = 0

    rew_buffer = {"agent_0": 0.0, "agent_1": 0.0}

    for timestep in tqdm.tqdm(
        range(max_timesteps), disable=True, file=sys.stdout
    ):
        # pre-interaction
        agent.pre_interaction(timestep=timestep, timesteps=max_timesteps)

        with torch.no_grad():
            outputs = agent.act(states, timestep=timestep, timesteps=max_timesteps)
            actions = (
                {k: outputs[-1][k].get("mean_actions", outputs[0][k]) for k in outputs[-1]}
            )
            
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            shared_next_states = env.state()
            infos["shared_states"] = shared_states
            infos["shared_next_states"] = shared_next_states

            traj_buffer = parse_observation(shared_next_states, traj_buffer)
            episode_length += 1

            rew_buffer["agent_0"] += rewards["agent_0"]
            rew_buffer["agent_1"] += rewards["agent_1"]

            # post-interaction
            super(MAPPO, agent).post_interaction(timestep=timestep, timesteps=max_timesteps)

            first_agent = next(iter(terminated))
            if terminated[first_agent].any() or truncated[first_agent].any():
                break
            else:
                states = next_states
                shared_states = shared_next_states

    # frames = env._env.get_complete_frames()

    return rew_buffer, traj_buffer #, frames

def parse_observation(observation, rollout_buffer):
    rollout_buffer["agent_0_pos"].append(observation[:, 6:8])
    rollout_buffer["agent_0_yaw"].append(observation[:, 8])
    rollout_buffer["agent_1_pos"].append(observation[:, 9:11])
    rollout_buffer["agent_1_yaw"].append(observation[:, 11])

    return rollout_buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Go1 Navigation agent")
    parser.add_argument("--run_date", type=str, default=None, help="Run date for curriculum learning")
    parser.add_argument("--curriculum_task", type=str, default=None, help="Curriculum task for curriculum learning")
    parser.add_argument("--sample_idx", type=int, default=None, help="Sample index for curriculum learning")

    args = parser.parse_args()
    run_date = args.run_date
    curriculum_task = args.curriculum_task
    sample_idx = args.sample_idx

    del args, parser

    save_dir = os.path.join("runs", run_date, curriculum_task, f"sample_{sample_idx}")
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Log directory {save_dir} does not exist. Please check the run date and curriculum task.")
    exp_name = f"{run_date}_{curriculum_task}_sample_{sample_idx}"

    env = create_env(eval=True)
    cfg = get_cfg(env, save_dir, exp_name, train=False)

    eval(env, cfg, save_dir)