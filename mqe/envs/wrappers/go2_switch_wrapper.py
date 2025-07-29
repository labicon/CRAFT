import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go2SwitchWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(12,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        self.reward_buffer = {
            "target reward": 0,
            "success reward": 0,
            "punishment": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        self.target_pos = self.root_states_npc[:, :3].reshape([self.env.num_envs, self.env.num_agents, -1]) - \
            self.env.env_origins.reshape([self.env.num_envs, 1, -1]).repeat([1, self.env.num_agents, 1])
        self.target_pos = self.target_pos[:, :, :2]
        self.center_pos = self.target_pos[:, 0, :2].clone() + self.target_pos[:, 1, :2].clone()
        self.center_pos = self.center_pos / 2

    def reset(self):
        obs_buf = self.env.reset()
        base_pos = obs_buf.base_pos[:, :2]
        base_yaw = obs_buf.base_rpy[:, 2].unsqueeze(1)
        base_info = torch.cat([base_pos, base_yaw], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        self._init_extras(obs_buf)

        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), self.target_pos, torch.flip(self.target_pos, [1])], dim=2)

        obs[:, 0, 2:4] = obs[:, 0, 2:4] - self.center_pos
        obs[:, 0, 5:7] = obs[:, 0, 5:7] - self.center_pos
        obs[:, 0, 8:10] = obs[:, 0, 8:10] - self.center_pos
        obs[:, 0, 10:12] = obs[:, 0, 10:12] - self.center_pos
        obs[:, 1, 2:4] = -(obs[:, 1, 2:4] - self.center_pos)
        obs[:, 1, 5:7] = -(obs[:, 1, 5:7] - self.center_pos)
        obs[:, 1, 8:10] = obs[:, 1, 8:10] - self.center_pos
        obs[:, 1, 10:12] = obs[:, 1, 10:12] - self.center_pos
        obs[:, 1, 4] = obs[:, 1, 4] - torch.pi
        obs[:, 1, 7] = obs[:, 1, 7] - torch.pi

        # Normalize angles to [0, 2π]
        obs[:, 0, 4] = ((obs[:, 0, 4] + torch.pi) % (2 * torch.pi)) - torch.pi
        obs[:, 0, 7] = ((obs[:, 0, 7] + torch.pi) % (2 * torch.pi)) - torch.pi
        obs[:, 1, 4] = ((obs[:, 1, 4] + torch.pi) % (2 * torch.pi)) - torch.pi
        obs[:, 1, 7] = ((obs[:, 1, 7] + torch.pi) % (2 * torch.pi)) - torch.pi

        return obs

    def step(self, action):
        # action[:, 1, 1:] = -action[:, 1, 1:]
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        self.reward_buffer["step count"] += 1
        
        reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device, dtype=torch.float)

        base_pos = obs_buf.base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])

        if self.success_reward_scale != 0:
            success_reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
            agent_0_success = torch.norm(base_pos[:, 0, :2] - self.target_pos[:, 0, :2], dim=1) < 0.5
            agent_1_success = torch.norm(base_pos[:, 1, :2] - self.target_pos[:, 1, :2], dim=1) < 0.5
            success_reward[:, 0] = agent_0_success.float() * self.success_reward_scale
            success_reward[:, 1] = agent_1_success.float() * self.success_reward_scale
            reward += success_reward
            self.reward_buffer["agent_0 success reward"] = torch.sum(success_reward[:, 0]).cpu()
            self.reward_buffer["agent_1 success reward"] = torch.sum(success_reward[:, 1]).cpu()

        if self.punishment_scale != 0:
            punishment = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
            collision = torch.norm(base_pos[:, 0, :2] - base_pos[:, 1, :2], dim=1) < 0.8
            punishment[:, 0] = collision.float() * self.punishment_scale
            punishment[:, 1] = collision.float() * self.punishment_scale
            reward += punishment
            self.reward_buffer["punishment"] = torch.sum(punishment).cpu()

        if self.target_reward_scale != 0:
            target_reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
            agent_0_target_distance = torch.norm(base_pos[:, 0, :2] - self.target_pos[:, 0, :2], dim=1)
            agent_1_target_distance = torch.norm(base_pos[:, 1, :2] - self.target_pos[:, 1, :2], dim=1)
            if not hasattr(self, "agent_0_last_distance_to_target"):
                self.agent_0_last_distance_to_target = agent_0_target_distance
            if not hasattr(self, "agent_1_last_distance_to_target"):
                self.agent_1_last_distance_to_target = agent_1_target_distance
            agent_0_progress = self.agent_0_last_distance_to_target - agent_0_target_distance
            agent_1_progress = self.agent_1_last_distance_to_target - agent_1_target_distance
            self.agent_0_last_distance_to_target = agent_0_target_distance
            self.agent_1_last_distance_to_target = agent_1_target_distance
            target_reward[:, 0] = agent_0_progress * self.target_reward_scale
            target_reward[:, 1] = agent_1_progress * self.target_reward_scale
            target_reward = torch.clip(target_reward, min=0)
            reward += target_reward
            self.reward_buffer["agent_0 target reward"] = torch.sum(target_reward[:, 0]).cpu()
            self.reward_buffer["agent_1 target reward"] = torch.sum(target_reward[:, 1]).cpu()

        base_pos = obs_buf.base_pos[:, :2]
        base_yaw = obs_buf.base_rpy[:, 2].unsqueeze(1)
        base_info = torch.cat([base_pos, base_yaw], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])

        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), self.target_pos, torch.flip(self.target_pos, [1])], dim=2)

        obs[:, 0, 2:4] = obs[:, 0, 2:4] - self.center_pos
        obs[:, 0, 5:7] = obs[:, 0, 5:7] - self.center_pos
        obs[:, 0, 8:10] = obs[:, 0, 8:10] - self.center_pos
        obs[:, 0, 10:12] = obs[:, 0, 10:12] - self.center_pos
        obs[:, 1, 2:4] = -(obs[:, 1, 2:4] - self.center_pos)
        obs[:, 1, 5:7] = -(obs[:, 1, 5:7] - self.center_pos)
        obs[:, 1, 8:10] = obs[:, 1, 8:10] - self.center_pos
        obs[:, 1, 10:12] = obs[:, 1, 10:12] - self.center_pos
        obs[:, 1, 4] = obs[:, 1, 4] - torch.pi
        obs[:, 1, 7] = obs[:, 1, 7] - torch.pi

        # Normalize angles to [0, 2π]
        obs[:, 0, 4] = ((obs[:, 0, 4] + torch.pi) % (2 * torch.pi)) - torch.pi
        obs[:, 0, 7] = ((obs[:, 0, 7] + torch.pi) % (2 * torch.pi)) - torch.pi
        obs[:, 1, 4] = ((obs[:, 1, 4] + torch.pi) % (2 * torch.pi)) - torch.pi
        obs[:, 1, 7] = ((obs[:, 1, 7] + torch.pi) % (2 * torch.pi)) - torch.pi

        return obs, reward, termination, self.reward_buffer
    

    # def _check_reward_shape(self, reward):
    #     if reward.shape == (self.num_envs, self.num_agents):
    #         return reward
    #     elif reward.shape == (self.num_envs, 1):
    #         return reward.repeat(1, self.num_agents)
    #     elif reward.shape == (self.num_envs,):
    #         return reward.unsqueeze(1).repeat(1, self.num_agents)
    #     else:
    #         raise ValueError(f"Invalid reward shape: {reward.shape}. Expected (num_envs, num_agents) or (num_envs, 1).")
        
    # def _eval(self, state, action):
    #     success = self._success_evaluation(state, action)
    #     agent_0_success_rate = success[:, 0].mean().item()
    #     agent_1_success_rate = success[:, 1].mean().item()

    #     agent_0_progress = self.progress[:, 0].mean().item()
    #     agent_1_progress = self.progress[:, 1].mean().item()

    #     agent_distance = self._agent_distance(state, action)
    #     collision = agent_distance[agent_distance < 0.25].mean().item() if agent_distance[agent_distance < 0.25].numel() > 0 else 0.0

    #     base_contact = self._contact_termination(state, action)
    #     base_contact = base_contact.mean().item()

    #     eval_dict = {
    #         "agent_0_success_rate": agent_0_success_rate,
    #         "agent_1_success_rate": agent_1_success_rate,
    #         "agent_0_progress": agent_0_progress,
    #         "agent_1_progress": agent_1_progress,
    #         "collision": collision,
    #         "base_contact": base_contact
    #     }

    #     return eval_dict
    