import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
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

import argparse
from skrl_ws.utils import create_env
import os
import datetime

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Train a Go1 Navigation agent")
    parser.add_argument("--training_iter", type=int, default=1_000_000, help="Training iteration for curriculum learning")

    args = parser.parse_args()
    run_date = datetime.datetime.now().strftime("%m-%d_%H-%M")
    training_iter = args.training_iter

    del args, parser

    save_dir = os.path.join("runs", "multiagent", run_date)
    os.makedirs(save_dir)
    exp_name = "multiagent" + "_" + run_date
    wandb_name = "multiagent" + "_" + run_date
    env = create_env(task="go1gate")
    cfg = get_cfg(env, save_dir, exp_name, train=True, wandb_name=wandb_name)

    train(env, cfg, training_iter)