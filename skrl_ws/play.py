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

from skrl_ws.wrapper import MQEMultiAgentWrapper

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

from datetime import datetime
from skrl_ws.utils import Policy, Value, create_env, get_cfg, find_last_checkpoint, save_gif
from mqe.utils import get_args
from mqe.envs.utils import make_mqe_env, custom_cfg

args = get_args()
args.headless = False
args.task = "go1navigation"
args.num_envs = 1
args.record_video = False

env, _ = make_mqe_env(args.task, args, custom_cfg(args))
env = MQEMultiAgentWrapper(env)
cfg = get_cfg(env, "runs", "mappo_play", train=False, wandb_name=False)
device = env.device

memories = {}
for agent_name in env.possible_agents:
    memories[agent_name] = RandomMemory(memory_size=12, num_envs=env.num_envs, device=device)

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

path = "/home/kang/MANavigation/runs/05-06_18-08/3_Final Task(05-06_18-08)/sample_1"
checkpoint_path = find_last_checkpoint(path)
agent.load(path=checkpoint_path)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 48000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

trainer.eval()

# env = trainer.env
# # env._env.start_recording()
# agent = trainer.agents

# states, infos = env.reset()
# shared_states = env.state()

# episode_length = 0

# max_timesteps = 50000

# for timestep in tqdm.tqdm(
#     range(max_timesteps), disable=True, file=sys.stdout
# ):
#     # pre-interaction
#     agent.pre_interaction(timestep=timestep, timesteps=max_timesteps)

#     with torch.no_grad():
#         outputs = agent.act(states, timestep=timestep, timesteps=max_timesteps)
#         actions = (
#             {k: outputs[-1][k].get("mean_actions", outputs[0][k]) for k in outputs[-1]}
#         )
        
#         next_states, rewards, terminated, truncated, infos = env.step(actions)
#         shared_next_states = env.state()
#         print("global state: ", shared_next_states)
#         print("rewards: ", rewards)

#         episode_length += 1

#         # post-interaction
#         super(MAPPO, agent).post_interaction(timestep=timestep, timesteps=max_timesteps)

#         first_agent = next(iter(terminated))
#         if terminated[first_agent].any() or truncated[first_agent].any():
#             break
#         else:
#             states = next_states
#             shared_states = shared_next_states

# frames = env._env.get_complete_frames()
