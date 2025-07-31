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
            "agent_pos": base_info[:, :, :2],
            "agent_height": base_info[:, :, 2],
            "agent_rpy": base_info[:, :, 3:],
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

    def _x_progress(self, state, action):
        base_pos = state["agent_pos"]
        x_pos = base_pos[:, :, 0]
        
        if not hasattr(self, "last_x_pos"):
            self.last_x_pos = x_pos
        x_movement = x_pos - self.last_x_pos
        self.last_x_pos = x_pos
        
        return x_movement
    
    def _normalized_height(self, state, action):
        height = state["agent_height"]
        normalized_height = height - 0.4

        return normalized_height
    
    def _wall_collision(self, state, action):
        collision = self.env.collide_buf.unsqueeze(1).repeat(1, self.num_agents)

        return collision.float()
    
    def _agent_collision(self, state, action):
        base_pos = state["agent_pos"]
        agent_dis = torch.norm(base_pos[:, 0, :] - base_pos[:, 1, :], p=2, dim=-1)

        collision = agent_dis < 0.5
        print("Agent collision:", collision)
        collision = collision.unsqueeze(1).repeat(1, self.num_agents)

        return collision.float()

    def _success(self, state, action):
        base_pos = state["agent_pos"]
        base_height = state["agent_height"]
        success = (base_pos[:, :, 0] > 7.7) * (base_height[:, :] > 1.3)

        return success.float()
    
    def _fall(self, state, action):
        fall = self.env.r_term_buff | self.env.p_term_buff
        fall = fall.unsqueeze(1).repeat(1, self.num_agents)

        return fall.float()
    
    def _eval(self, state, action):
        success = self._success(state, action)
        agent_0_success = success[:, 0].mean().item()
        agent_1_success = success[:, 1].mean().item()

        agent_0_height = state["agent_height"][:, 0].mean().item()
        agent_1_height = state["agent_height"][:, 1].mean().item()

        agent_collision = self._agent_collision(state, action)
        agent_collision = agent_collision.mean().item()

        base_contact = self._wall_collision(state, action)
        base_contact = base_contact.mean().item()

        fall = self._fall(state, action)
        fall = fall.mean().item()

        eval_dict = {
            "agent_0_success_rate": agent_0_success,
            "agent_1_success_rate": agent_1_success,
            "agent_0_height": agent_0_height,
            "agent_1_height": agent_1_height,
            "agent_collision": agent_collision,
            "base_contact": base_contact,
            "fall": fall
        }

        return eval_dict