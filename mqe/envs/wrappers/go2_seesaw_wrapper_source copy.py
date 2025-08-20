import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go2SeesawWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(12 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        self.reward_buffer = {
            "step count": 0
        }

    def _init_extras(self, obs):
        return
    
    def reset(self):
        obs_buf = self.env.reset()

        self.seesaw_center = torch.tensor([5.5, 0.0, 1.0], device="cuda").unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.num_agents, 1)
        self.seesaw_start = torch.tensor([4.0, 0.0], device="cuda").unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.num_agents, 1)
        self.seesaw_end = torch.tensor([7.0, 0.0], device="cuda").unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.num_agents, 1)
        self.target_pos = torch.tensor([7.7, 0.0, 1.5], device="cuda").unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.num_agents, 1)

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1])], dim=2)

        self.reward_buffer = {
            "step count": 0
        }

        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1])], dim=2)

        self.reward_buffer["step count"] += 1
        
        global_state = {
            "agent_pos": base_info[:, :, :3],
            "target_pos": self.target_pos,
            "seesaw_center": self.seesaw_center,
            "seesaw_start": self.seesaw_start,
            "seesaw_end": self.seesaw_end,
        }

        reward, reward_dict, max_reward = self.gpt_reward(global_state, action)

        for key, value in reward_dict.items():
            reward_key = f"Reward/{key}"
            self.reward_buffer[reward_key] = value
        
        eval_dict = self._eval(global_state, action)
        for key, value in eval_dict.items():
            eval_key = f"Eval/{key}"
            self.reward_buffer[eval_key] = value

        return obs, reward, termination, self.reward_buffer

    def _progress_in_x(self, state, action):
        base_pos = state["agent_pos"]
        x_pos = base_pos[:, :, 0]
        
        if not hasattr(self, "last_x_pos"):
            self.last_x_pos = copy(x_pos)
        x_movement = x_pos - self.last_x_pos
        x_movement[self.env.reset_ids] = 0.0

        # Scale up the x_movement to match the action scale
        x_movement = x_movement * 10.0

        self.last_x_pos = copy(x_pos)

        return x_movement
    
    def _distance_to_seesaw_start(self, state, action):
        base_pos = state["agent_pos"]
        seesaw_start_distance = torch.norm(base_pos[:, :, :2] - self.seesaw_start, p=2, dim=-1) / 3.0

        return seesaw_start_distance
    
    def _distance_to_seesaw_end(self, state, action):
        base_pos = state["agent_pos"]
        seesaw_end_distance = torch.norm(base_pos[:, :, :2] - self.seesaw_end, p=2, dim=-1) / 6.0

        return seesaw_end_distance

    def _distance_to_seesaw_center(self, state, action):
        base_pos = state["agent_pos"]
        seesaw_center_distance = torch.norm(base_pos[:, :, :] - self.seesaw_center, p=2, dim=-1) / 5.0

        return seesaw_center_distance
    
    def _distance_to_target(self, state, action):
        base_pos = state["agent_pos"]
        target_distance = torch.norm(base_pos[:, :, :] - self.target_pos[:, :, :], p=2, dim=-1) / 8.0

        return target_distance
    
    def _agent_distance(self, state, action):
        base_pos = state["agent_pos"]
        agent_dis = torch.norm(base_pos[:, 0, :] - base_pos[:, 1, :], p=2, dim=-1)

        agent_dis = agent_dis.unsqueeze(1).repeat(1, self.num_agents)

        return agent_dis.float()
    
    def _normalized_height(self, state, action):
        height = state["agent_pos"][:, :, 2]
        normalized_height = height - 0.4
        normalized_height = torch.clip(normalized_height, min=0.0, max=1.0)

        return normalized_height
    
    def _wall_collision(self, state, action):
        collision = self.env.collide_buf.unsqueeze(1).repeat(1, self.num_agents)

        return collision.float()
    
    def _fall(self, state, action):
        fall = self.env.r_term_buff | self.env.p_term_buff
        fall = fall.unsqueeze(1).repeat(1, self.num_agents)

        return fall.float()
    
    def _success(self, state, action):
        base_pos = state["agent_pos"]
        success = (base_pos[:, :, 0] > 7.7) * (base_pos[:, :, 2] > 1.3)
        success = torch.any(success, dim=1).unsqueeze(1).repeat(1, self.num_agents)

        return success.float()
    
    def _command_value(self, state, action):
        return torch.norm(action, p=2, dim=-1)
    
    def _yaw_penalty(self, state, action):
        base_rpy = state["agent_pos"][:, :, 3:6]
        yaw = base_rpy[:, :, 2]
        yaw_penalty = torch.abs(yaw) / 3.14

        return yaw_penalty

    def _check_reward_shape(self, reward):
        if reward.shape == (self.num_envs, self.num_agents):
            return reward
        elif reward.shape == (self.num_envs, 1):
            return reward.repeat(1, self.num_agents)
        elif reward.shape == (self.num_envs,):
            return reward.unsqueeze(1).repeat(1, self.num_agents)
        else:
            raise ValueError(f"Invalid reward shape: {reward.shape}. Expected (num_envs, num_agents) or (num_envs, 1).")

    def _eval(self, state, action):
        success = self._success(state, action).mean().item()

        agent_0_height = state["agent_pos"][:, 0, 2].mean().item()
        agent_1_height = state["agent_pos"][:, 1, 2].mean().item()

        agent_distance = self._agent_distance(state, action)
        agent_collision = (agent_distance < 0.5).float().mean().item()

        base_contact = self._wall_collision(state, action)
        base_contact = base_contact.mean().item()

        fall = self._fall(state, action)
        fall = fall.mean().item()

        eval_dict = {
            "success_rate": success,
            "agent_0_height": agent_0_height,
            "agent_1_height": agent_1_height,
            "agent_collision": agent_collision,
            "base_contact": base_contact,
            "fall": fall
        }

        return eval_dict
    
