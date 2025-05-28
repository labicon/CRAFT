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

from skrl_ws.wrapper import MQEMultiAgentWrapper

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

from datetime import datetime

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU())

        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        self._shared_output = self.net(inputs["states"])
        return self.mean_layer(self._shared_output), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU())
        
        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        shared_output = self.net(inputs["states"])
        return self.value_layer(shared_output), {}


from mqe.utils import get_args
from mqe.envs.utils import make_mqe_env, custom_cfg

# load and wrap the Isaac Gym environment
args = get_args()
args.headless = False
args.task = "go1navigation"
args.num_envs = 1
args.record_video = False
env, _ = make_mqe_env(args.task, args, custom_cfg(args))
env = MQEMultiAgentWrapper(env)

device = env.device

# instantiate memories as rollout buffer (any memory can be used for this)
memories = {}
for agent_name in env.possible_agents:
    memories[agent_name] = RandomMemory(memory_size=12, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# MAPPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/multi_agents/mappo.html#models
# models = {}
# for agent_name in env.possible_agents:
#     models[agent_name] = {}
#     models[agent_name]["policy"] = Policy(env.observation_space(agent_name), env.action_space(agent_name), device)
#     models[agent_name]["value"] = Value(env.state_space(agent_name), env.action_space(agent_name), device)

models = {}
shared_policy = Policy(env.observation_space(env.possible_agents[0]), env.action_space(env.possible_agents[0]), device)
shared_value = Value(env.state_space(env.possible_agents[0]), env.action_space(env.possible_agents[0]), device)
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = shared_policy
    models[agent_name]["value"] = shared_value

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = MAPPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 24 * 4096 / 24576
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": next(iter(env.observation_spaces.values())), "device": device}
cfg["shared_state_preprocessor"] = RunningStandardScaler
cfg["shared_state_preprocessor_kwargs"] = {"size": next(iter(env.state_spaces.values())), "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["wandb"] = False

agent = MAPPO(possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            cfg=cfg,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device,
            shared_observation_spaces=env.state_spaces)



path = "/home/kang/MANavigation/runs/multiagent/05-13_17-15/25-05-13_17-15-21-228975_MAPPO/checkpoints/agent_1000000.pt"
agent.load(path)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 48000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start evaluation
trainer.eval()