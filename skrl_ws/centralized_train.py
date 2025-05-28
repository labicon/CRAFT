import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl_ws.wrapper import MQECentralizedWrapper

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

from skrl_ws.utils import Policy, Value, create_env, get_ppo_cfg, find_last_checkpoint

def train(env:MQECentralizedWrapper, cfg, training_iter):
    device = env.device

    # instantiate memories as rollout buffer (any memory can be used for this)
    memories = RandomMemory(memory_size=cfg['rollouts'], num_envs=env.num_envs, device=device)

    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.action_space, device)

    agent = PPO(models=models,
                memory=memories,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": training_iter, "headless": True, "environment_info": "reward"}
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

    save_dir = os.path.join("runs", "centralized", run_date)
    os.makedirs(save_dir)
    exp_name = "centralized" + "_" + run_date
    wandb_name = "centralized" + "_" + run_date
    env = create_env(wrapper='centralized')
    cfg = get_ppo_cfg(env, save_dir, exp_name, train=True, wandb_name=wandb_name)

    train(env, cfg, training_iter)