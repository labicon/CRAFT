import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl_ws.wrapper import MQECentralizedWrapper

from datetime import datetime
from skrl_ws.utils import Policy, Value, create_env, get_cfg, find_last_checkpoint, save_gif
from mqe.utils import get_args
from mqe.envs.utils import make_mqe_env, custom_cfg

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

from skrl_ws.utils import Policy, Value, create_env, get_ppo_cfg, find_last_checkpoint

args = get_args()
args.headless = False
args.task = "go1navigation"
args.num_envs = 1
args.record_video = False

env, _ = make_mqe_env(args.task, args, custom_cfg(args))
env = MQECentralizedWrapper(env)
cfg = get_ppo_cfg(env, "runs", "centralized_play", train=False, wandb_name=False)
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

# path = ""
# checkpoint_path = find_last_checkpoint(path)
checkpoint_path = "/home/kang/MANavigation/runs/centralized/05-26_19-23/25-05-26_19-23-29-280274_PPO/checkpoints/agent_1000000.pt"
agent.load(checkpoint_path)

cfg_trainer = {"timesteps": 50000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# trainer.eval()

env = trainer.env
agent = trainer.agents

states, infos = env.reset()

episode_length = 0

max_timesteps = 50000

import tqdm
import sys

for timestep in tqdm.tqdm(
    range(max_timesteps), disable=True, file=sys.stdout
):
    # pre-interaction
    agent.pre_interaction(timestep=timestep, timesteps=max_timesteps)

    with torch.no_grad():
        # compute actions
        outputs = agent.act(states, timestep=timestep, timesteps=max_timesteps)
        actions = outputs[-1].get("mean_actions", outputs[0])

        # step the environments
        next_states, rewards, terminated, truncated, infos = env.step(actions)

        print("global state: ", next_states)
        print("rewards: ", rewards)

        episode_length += 1

        # post-interaction
        super(PPO, agent).post_interaction(timestep=timestep, timesteps=max_timesteps)

        if terminated.any() or truncated.any():
            with torch.no_grad():
                states, infos = env.reset()
        else:
            states = next_states

