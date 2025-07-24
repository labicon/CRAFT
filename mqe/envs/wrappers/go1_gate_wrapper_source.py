import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1GateWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(14 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        self.reward_buffer = {
            "step count": 0
        }

    def _init_extras(self, obs):

        self.gate_pos = obs.env_info["gate_deviation"]
        self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        self.frame_left = self.gate_pos.reshape(-1, 2).clone()
        self.frame_right = self.gate_pos.reshape(-1, 2).clone()
        self.frame_left[:, 1] += self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.frame_right[:, 1] -= self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

        self.target_pos = self.gate_pos.clone()
        # self.target_pos = torch.zeros_like(self.gate_pos, dtype=self.gate_pos.dtype, device=self.gate_pos.device)
        self.target_pos[:, :, 0] = self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] + self.BarrierTrack_kwargs["plane"]["block_length"] / 2
        # self.target_pos[:, 0, 1] = self.BarrierTrack_kwargs["track_width"] / 4
        # self.target_pos[:, 1, 1] = - self.BarrierTrack_kwargs["track_width"] / 4
        self.target_pos = self.target_pos.reshape(-1, 2)

        self.progress = torch.zeros((self.num_envs, self.num_agents), device=self.env.device)

        return

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), self.gate_pos], dim=2)

        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), self.gate_pos], dim=2)

        global_state = {
            "gate_pos": self.gate_pos.reshape(self.num_envs, self.num_agents, -1),
            "gate_left": self.frame_left.reshape(self.num_envs, self.num_agents, -1),
            "gate_right": self.frame_right.reshape(self.num_envs, self.num_agents, -1),
            "target_pos": self.target_pos.reshape(self.num_envs, self.num_agents, -1),
            "agent_pos": base_info[:, :, :2],
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

        return obs, reward, termination, self.reward_buffer
    
    def _progress_to_gate(self, state, action):
        base_pos = state["agent_pos"]
        gate_pos = state["gate_pos"]

        # approach reward
        distance_to_gate = torch.norm(base_pos - gate_pos, p=2, dim=-1)

        if not hasattr(self, "last_distance_to_gate"):
            self.last_distance_to_gate = copy(distance_to_gate)

        progress = (self.last_distance_to_gate - distance_to_gate) * 10.0
        progress[self.env.reset_ids] = 0
        self.progress = progress

        self.last_distance_to_gate = copy(distance_to_gate)

        return progress

    def _distance_to_gate(self, state, action):
        base_pos = state["agent_pos"]
        gate_pos = state["gate_pos"]

        # approach reward
        distance_to_gate = torch.norm(base_pos[:, :, :2] - gate_pos, p=2, dim=-1)
        return distance_to_gate
    
    def _distance_to_target(self, state, action):
        base_pos = state["agent_pos"]
        target_pos = state["target_pos"]

        # approach reward
        distance_to_target = torch.norm(base_pos[:, :, :2] - target_pos, p=2, dim=-1)
        return distance_to_target
    
    def _get_gate_frame(self, state, action):
        frame_left = self.frame_left.reshape(self.num_envs, self.num_agents, -1)
        frame_left = frame_left[:, :, 1].unsqueeze(2)
        frame_right = self.frame_right.reshape(self.num_envs, self.num_agents, -1)
        frame_right = frame_right[:, :, 1].unsqueeze(2)

        gate_frame = torch.cat([frame_left, frame_right], dim=-1)

        return gate_frame

    def _contact_punishment(self, state, action):
        collide_reward = self.env.collide_buf.clone().detach().float()
        return collide_reward.unsqueeze(1).repeat(1, self.num_agents)

    def _agent_distance(self, state, action):
        base_pos = state["agent_pos"]
        # Compute pairwise distance between agents
        agent_0_pos = base_pos[:, 0, :2]
        agent_1_pos = base_pos[:, 1, :2]
        distance = torch.norm(agent_0_pos - agent_1_pos, p=2, dim=-1)

        distance = distance.unsqueeze(1).repeat(1, self.num_agents)
        return distance
    
    def _command_lin_vel_y(self, state, action):
        # command lin_vel.y punishment
        v_y_punishment = action[:, :, 1] * 0.01
        return v_y_punishment
    
    def _command_lin_vel_x(self, state, action):
        # lin_vel.x reward
        v_x_reward = action[:, :, 0] * 0.01
        return v_x_reward
    
    def _command_value(self, state, action):
        # command value punishment
        command_value_punishment = torch.norm(action, p=2, dim=2) * 0.01
        return command_value_punishment
    
    def _success_evaluation(self, state, action):
        # Check if each agents have passed the gate
        gate_pos = state["gate_pos"]
        base_pos = state["agent_pos"]
        agent_0_pos = base_pos[:, 0, :2]
        agent_1_pos = base_pos[:, 1, :2]

        agent_0_passed = agent_0_pos[:, 0] > gate_pos[:, 0, 0] + 0.25
        agent_1_passed = agent_1_pos[:, 0] > gate_pos[:, 1, 0] + 0.25

        success = torch.zeros([self.num_envs, self.num_agents], device=self.env.device)
        success[agent_0_passed, 0] = 1.0
        success[agent_1_passed, 1] = 1.0
        return success

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
        success = self._success_evaluation(state, action)
        agent_0_success_rate = success[:, 0].mean().item()
        agent_1_success_rate = success[:, 1].mean().item()

        agent_0_progress = self.progress[:, 0].mean().item()
        agent_1_progress = self.progress[:, 1].mean().item()

        agent_distance = self._agent_distance(state, action)
        collision = agent_distance[agent_distance < 0.25].mean().item() if agent_distance[agent_distance < 0.25].numel() > 0 else 0.0

        base_contact = self._contact_punishment(state, action)
        base_contact = base_contact.mean().item()

        eval_dict = {
            "agent_0_success_rate": agent_0_success_rate,
            "agent_1_success_rate": agent_1_success_rate,
            "agent_0_progress": agent_0_progress,
            "agent_1_progress": agent_1_progress,
            "collision": collision,
            "base_contact": base_contact
        }

        return eval_dict
    
