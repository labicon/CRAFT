import isaacgym

import torch
import torch.nn as nn
import shutil

# import the skrl components to build the RL system
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

from skrl_ws.utils import Policy, Value, create_env, get_cfg, find_last_checkpoint

def train(env, cfg, training_iter):
    device = env.device

    # instantiate memories as rollout buffer (any memory can be used for this)
    memories = {}
    for agent_name in env.possible_agents:
        memories[agent_name] = RandomMemory(memory_size=cfg['rollouts'], num_envs=env.num_envs, device=device)

    models = {}
    shared_policy = Policy(env.observation_space(env.possible_agents[0]), env.action_space(env.possible_agents[0]), device)
    shared_value = Value(env.state_space(env.possible_agents[0]), env.action_space(env.possible_agents[0]), device)
    for agent_name in env.possible_agents:
        models[agent_name] = {}
        models[agent_name]["policy"] = shared_policy
        models[agent_name]["value"] = shared_value

    agent = MAPPO(possible_agents=env.possible_agents,
                models=models,
                memories=memories,
                cfg=cfg,
                observation_spaces=env.observation_spaces,
                action_spaces=env.action_spaces,
                device=device,
                shared_observation_spaces=env.state_spaces)


    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": training_iter, "headless": True, "environment_info": "agent_0"}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()

def load_train(env, cfg, training_iter, save_dir):
    device = env.device

    # instantiate memories as rollout buffer (any memory can be used for this)
    memories = {}
    for agent_name in env.possible_agents:
        memories[agent_name] = RandomMemory(memory_size=cfg['rollout'], num_envs=env.num_envs, device=device)

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

    previous_value = agent.models[env.possible_agents[0]]["value"].to(device)

    # Reset the MAPPO's value functions
    shared_value = Value(env.state_space(env.possible_agents[0]), env.action_space(env.possible_agents[0]), device)
    for agent_name in env.possible_agents:
        agent.models[agent_name]["value"] = shared_value

    # Compare the weights of the value function before and after resetting
    reset_value = agent.models[env.possible_agents[0]]["value"].to(device)
    for param1, param2 in zip(previous_value.net.parameters(), reset_value.net.parameters()):
        if torch.equal(param1, param2):
            print("Weights are equal")
        else:
            print("Weights are different")

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": training_iter, "headless": True, "environment_info": "agent_0"}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()

import argparse
from skrl_ws.utils import create_env
import os

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Train a Go1 Navigation agent")
    parser.add_argument("--run_date", type=str, default=None, help="Run date for curriculum learning")
    parser.add_argument("--curriculum_task", type=str, default=None, help="Curriculum task for curriculum learning")
    parser.add_argument("--sample_idx", type=int, default=None, help="Sample index for curriculum learning")
    parser.add_argument("--training_iter", type=int, default=None, help="Training iteration for curriculum learning")
    parser.add_argument("--load", type=bool, default=False, help="Load previous training checkpoint")
    parser.add_argument("--load_task", type=str, default=None, help="Load task for curriculum learning")
    parser.add_argument("--load_sample_idx", type=int, default=None, help="Load sample index for curriculum learning")

    args = parser.parse_args()
    run_date = args.run_date
    curriculum_task = args.curriculum_task
    sample_idx = args.sample_idx
    training_iter = args.training_iter
    load = args.load
    load_task = args.load_task
    load_sample_idx = args.load_sample_idx

    del args, parser

    save_dir = os.path.join("runs", run_date, curriculum_task, f"sample_{sample_idx}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    exp_name = f"{run_date}_{curriculum_task}_sample_{sample_idx}"
    wandb_name = f"{curriculum_task}_sample_{sample_idx}"

    if load:
        load_dir = os.path.join("runs", run_date, load_task, f"sample_{load_sample_idx}")
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Log directory {load_dir} does not exist. Please check the run date and curriculum task.")

    env = create_env()
    cfg = get_cfg(env, save_dir, exp_name, train=True, wandb_name=wandb_name)

    if not load:
        train(env, cfg, training_iter)
    else:
        load_train(env, cfg, training_iter, load_dir)
