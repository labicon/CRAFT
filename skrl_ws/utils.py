import isaacgym
import torch
import torch.nn as nn

import sys, os, re
import numpy as np
import imageio

from skrl_ws.wrapper import MQEMultiAgentWrapper, MQESingleAgentWrapper, MQECentralizedWrapper
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from mqe.utils import get_args
from mqe.envs.utils import make_mqe_env, custom_cfg

def create_env(task, eval=False, wrapper='multiagent'):
    sys.argv = [sys.argv[0]]
    # load and wrap the Isaac Gym environment
    args = get_args()
    args.headless = True
    args.task = task
    if not eval:
        args.num_envs = 512
        args.record_video = False
        args.seed = np.random.randint(0, 10000)
    else:
        args.num_envs = 1
        args.record_video = True
        args.seed = 42

    env, _ = make_mqe_env(args.task, args, custom_cfg(args))
    if wrapper == 'singleagent':
        env = MQESingleAgentWrapper(env)
    elif wrapper == 'centralized':
        env = MQECentralizedWrapper(env)
    elif wrapper == 'multiagent':
        env = MQEMultiAgentWrapper(env)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper}. Choose from 'singleagent', 'centralized', or 'multiagent'.")

    return env

def get_cfg(env, save_dir, exp_name, train=True, wandb_name=False):
    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = MAPPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 256  # memory_size
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 1  # horizon_length * num_actors / minibatch_size = 1000 * 64 / 16000
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 5e-5
    cfg["learning_rate_scheduler"] = KLAdaptiveLR
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "max_lr": 5e-4}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.01
    cfg["value_loss_scale"] = 0.5
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = None
    cfg["time_limit_bootstrap"] = False
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": next(iter(env.observation_spaces.values())), "device": env.device}
    cfg["shared_state_preprocessor"] = RunningStandardScaler
    cfg["shared_state_preprocessor_kwargs"] = {"size": next(iter(env.state_spaces.values())), "device": env.device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
    if train:
        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]['name'] = exp_name
        cfg["experiment"]["write_interval"] = 100
        cfg["experiment"]["checkpoint_interval"] = 5000
        cfg["experiment"]["directory"] = save_dir
        cfg["experiment"]["wandb"] = train
        cfg["experiment"]["wandb_kwargs"] = {"project": "go1navigation", "name": wandb_name}

    return cfg

def get_ppo_cfg(env, save_dir, exp_name, train=True, wandb_name=False):
    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 1024  # memory_size
    cfg["learning_epochs"] = 4
    cfg["mini_batches"] = 16  # horizon_length * num_actors / minibatch_size = 1000 * 64 / 16000
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 5e-5
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
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
    if train:
        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]['name'] = exp_name
        cfg["experiment"]["write_interval"] = 100
        cfg["experiment"]["checkpoint_interval"] = 1000
        cfg["experiment"]["directory"] = save_dir
        cfg["experiment"]["wandb"] = train
        cfg["experiment"]["wandb_kwargs"] = {"project": "go1navigation", "name": wandb_name}

    return cfg

def find_last_checkpoint(path):
    # Find the only folder inside the parent directory
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if len(folders) != 1:
        raise ValueError(f"Expected exactly one folder in {path}, but found {len(folders)}.")
    agent_folder = os.path.join(path, folders[0])

    # Get a list of all files in the directory
    checkpoint_path = os.path.join(agent_folder, "checkpoints")
    files = os.listdir(checkpoint_path)

    # Filter for files that match the checkpoint pattern
    checkpoints = [f for f in files if re.match(r"agent_\d+\.pt", f)]

    # Sort the checkpoints by their number
    checkpoints.sort(key=lambda x: int(re.search(r"\d+", x).group()))

    # Return the last checkpoint file
    return os.path.join(checkpoint_path, checkpoints[-1]) if checkpoints else None

def save_gif(frames, fps, save_dir=''):
    # Frames are from env.get_complete_frames()

    # Define the output GIF file name
    output_gif_path = os.path.join(save_dir, 'output_animation.gif')
    # Convert the frames to uint8 (assuming it's in range 0-1)
    video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
    video_array = np.transpose(video_array, (0, 2, 3, 1))
    video_array_uint8 = video_array.astype(np.uint8)

    frames = [video_array_uint8[i] for i in range(len(video_array_uint8))]

    # Save frames as GIF
    imageio.mimsave(output_gif_path, frames, fps=fps)

    print("GIF created successfully.")

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
        return torch.tanh(self.mean_layer(self._shared_output)), self.log_std_parameter, {}

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