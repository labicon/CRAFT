import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym.torch_utils import *

class Go2PushboxWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(9 + self.num_agents + 2,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[1.0, 1.0, 1.0],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)

        self.box_x_movement_reward_scale = 1

        self.reward_buffer = {
            "box movement reward": 0,
            "step count": 0
        }

    def reset(self):
        obs_buf = self.env.reset()

        # Randomize target position per env
        self.target_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.target_pos[:, 0] = torch.empty(self.num_envs, device=self.device).uniform_(2.0, 3.5)
        self.target_pos[:, 1] = torch.empty(self.num_envs, device=self.device).uniform_(-1.5, 1.5)

        box_pos = self.root_states_npc[:, :3] - self.env.env_origins
        # get_euler_xyz returns (roll, pitch, yaw) as tuple of tensors
        _, _, box_yaw = get_euler_xyz(self.root_states_npc[:, 3:7])
        box_yaw = (box_yaw + np.pi) % (2 * np.pi) - np.pi

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy # (num_envs * num_agents, 3)
        # Use only yaw (index 2) from base_rpy
        base_yaw = base_rpy[:, 2:3]
        base_yaw = (base_yaw + np.pi) % (2 * np.pi) - np.pi
        base_info = torch.cat([base_pos[:, :2], base_yaw], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])

        target_obs = self.target_pos.unsqueeze(1).repeat(1, self.num_agents, 1)

        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         box_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1),
                         box_yaw.unsqueeze(1).unsqueeze(2).repeat(1, self.num_agents, 1),
                         target_obs,
                         ], dim=2)

        self.last_box_pos = box_pos.clone()

        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        box_pos = self.root_states_npc[:, :3] - self.env.env_origins
        # get_euler_xyz returns (roll, pitch, yaw) as tuple of tensors
        _, _, box_yaw = get_euler_xyz(self.root_states_npc[:, 3:7])
        box_yaw = (box_yaw + np.pi) % (2 * np.pi) - np.pi

        # Re-randomize target and update last_box_pos for auto-reset envs
        if len(self.env.reset_ids) > 0:
            n = len(self.env.reset_ids)
            self.target_pos[self.env.reset_ids, 0] = torch.empty(n, device=self.device).uniform_(2.0, 3.5)
            self.target_pos[self.env.reset_ids, 1] = torch.empty(n, device=self.device).uniform_(-1.5, 1.5)
            self.last_box_pos[self.env.reset_ids] = box_pos[self.env.reset_ids].clone()

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy # (num_envs * num_agents, 3)
        # Use only yaw (index 2) from base_rpy
        base_yaw = base_rpy[:, 2:3]
        base_yaw = (base_yaw + np.pi) % (2 * np.pi) - np.pi
        base_info = torch.cat([base_pos[:, :2], base_yaw], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])

        target_obs = self.target_pos.unsqueeze(1).repeat(1, self.num_agents, 1)

        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         box_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1),
                         box_yaw.unsqueeze(1).unsqueeze(2).repeat(1, self.num_agents, 1),
                         target_obs,
                         ], dim=2)

        global_state = {
            "box_pos": box_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1),
            "box_yaw": box_yaw.unsqueeze(1).unsqueeze(2).repeat(1, self.num_agents, 1),
            "agent_pos": base_info[:, :, :2],
            "agent_yaw": base_info[:, :, 2:3],
            "target_pos": target_obs,
        }

        self.reward_buffer["step count"] += 1

        reward, reward_dict, max_reward = self.gpt_reward(global_state, action)

        for key, value in reward_dict.items():
            reward_key = f"Reward/{key}"
            self.reward_buffer[reward_key] = value

        eval_dict = self._eval(global_state, action)
        for key, value in eval_dict.items():
            eval_key = f"Eval/{key}"
            self.reward_buffer[eval_key] = value

        return obs, reward, termination, info

    def _box_forward_movement(self, state, action):
        box_pos = state["box_pos"]
        if self.last_box_pos is None:
             return torch.zeros(self.num_envs, self.num_agents, device=self.device)

        current_x = box_pos[:, :, 0]
        last_x = self.last_box_pos[:, 0].unsqueeze(1).repeat(1, self.num_agents)

        movement = (current_x - last_x) * 50
        movement[self.env.reset_ids] = 0
        return movement

    def _box_distance_to_target(self, state, action):
        box_pos = state["box_pos"]
        target = state["target_pos"]
        return torch.norm(box_pos - target, dim=-1)

    def _box_movement_toward_target(self, state, action):
        box_pos = state["box_pos"]
        target = state["target_pos"]
        if self.last_box_pos is None:
            return torch.zeros(self.num_envs, self.num_agents, device=self.device)

        last_pos = self.last_box_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1)
        initial_dist = torch.norm(last_pos - target, dim=-1)
        current_dist = torch.norm(box_pos - target, dim=-1)

        movement = (initial_dist - current_dist) * 50
        movement[self.env.reset_ids] = 0
        return movement

    def _box_y_deviation(self, state, action):
        box_pos = state["box_pos"]
        return torch.abs(box_pos[:, :, 1])

    def _agent_distance(self, state, action):
        agent_pos = state["agent_pos"]
        p0 = agent_pos[:, 0, :]
        p1 = agent_pos[:, 1, :]
        dist = torch.norm(p0 - p1, dim=1).unsqueeze(1).repeat(1, self.num_agents)
        return dist

    def _agents_distance_to_box(self, state, action):
        box_pos = state["box_pos"]
        agent_pos = state["agent_pos"]
        return torch.norm(box_pos - agent_pos, dim=-1)

    def _success_evaluation(self, state, action):
        dist = self._box_distance_to_target(state, action)
        success = (dist <= 0.5).float()
        return success

    def _check_reward_shape(self, reward):
        if reward.shape == (self.num_envs, self.num_agents):
            return reward
        elif reward.shape == (self.num_envs, 1):
            return reward.repeat(1, self.num_agents)
        elif reward.shape == (self.num_envs,):
            return reward.unsqueeze(1).repeat(1, self.num_agents)
        elif reward.shape == (self.num_envs, self.num_agents, 1):
            return reward.squeeze(2)
        else:
            raise ValueError(f"Invalid reward shape: {reward.shape}. Expected ({self.num_envs}, {self.num_agents}) or ({self.num_envs}, 1) or ({self.num_envs}, ) but got {reward.shape}.")

    def _eval(self, state, action):
        success = self._success_evaluation(state, action)
        agent_0_success_rate = success[:, 0].mean().item()
        agent_1_success_rate = success[:, 1].mean().item()

        box_progress = self._box_movement_toward_target(state, action)
        box_progress = box_progress.mean().item()

        box_distance = self._box_distance_to_target(state, action)
        box_distance = box_distance.mean().item()

        agent_distance = self._agent_distance(state, action)
        collision = agent_distance[agent_distance < 0.5].mean().item() if agent_distance[agent_distance < 0.5].numel() > 0 else 0.0

        eval_info = {
            "agent_0_success_rate": agent_0_success_rate,
            "agent_1_success_rate": agent_1_success_rate,
            "box_progress": box_progress,
            "box_distance_to_target": box_distance,
            "collision": collision,
        }
        return eval_info

