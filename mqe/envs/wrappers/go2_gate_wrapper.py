import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go2GateWrapper(EmptyWrapper):
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

        progress = (self.last_distance_to_gate - distance_to_gate) * 30.0
        progress[self.env.reset_ids] = 0
        self.progress = progress

        self.last_distance_to_gate = copy(distance_to_gate)

        return progress

    def _distance_to_gate(self, state, action):
        base_pos = state["agent_pos"]
        gate_pos = state["gate_pos"]

        # approach reward
        distance_to_gate = torch.norm(base_pos[:, :, :2] - gate_pos, p=2, dim=-1) / 2.0
        return distance_to_gate
    
    def _distance_to_target(self, state, action):
        base_pos = state["agent_pos"]
        target_pos = state["target_pos"]

        # approach reward
        distance_to_target = torch.norm(base_pos[:, :, :2] - target_pos, p=2, dim=-1) / 2.0
        return distance_to_target
    
    def _get_gate_frame(self, state, action):
        frame_left = self.frame_left.reshape(self.num_envs, self.num_agents, -1)
        frame_left = frame_left[:, :, 1].unsqueeze(2)
        frame_right = self.frame_right.reshape(self.num_envs, self.num_agents, -1)
        frame_right = frame_right[:, :, 1].unsqueeze(2)

        gate_frame = torch.cat([frame_left, frame_right], dim=-1)

        return gate_frame

    def _contact_termination(self, state, action):
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
        v_y_punishment = torch.abs(action[:, :, 1]) * 1.0
        return v_y_punishment
    
    def _command_lin_vel_x(self, state, action):
        # lin_vel.x reward
        v_x_reward = torch.abs(action[:, :, 0]) * 0.5
        return v_x_reward
    
    def _command_value(self, state, action):
        # command value punishment
        command_value_punishment = self._command_lin_vel_y(state, action) + self._command_lin_vel_x(state, action)
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

        base_contact = self._contact_termination(state, action)
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
    

    def gpt_reward(self, state, action):

        reward = torch.zeros((self.num_envs, self.num_agents), device=self.env.device)
        rew_dict = {}
        max_reward = 25.0

        # Reward for progressing towards the gate - max 3.0
        progress_reward = 3.0 * self._progress_to_gate(state, action)
        # Check reward shape
        progress_reward = self._check_reward_shape(progress_reward)
        rew_dict["agent_0_progress"] = torch.mean(progress_reward[:, 0])
        rew_dict["agent_1_progress"] = torch.mean(progress_reward[:, 1])

        # Reward for not colliding with each other - max 2.0
        agent_distance = self._agent_distance(state, action)
        distance_threshold = 0.25
        penalty_threshold = 0.5
        safe_distance_reward = torch.zeros_like(agent_distance)
        safe_distance_reward[agent_distance >= penalty_threshold] = 2.0
        safe_distance_reward[agent_distance <= distance_threshold] = 0.0
        mask = (agent_distance > distance_threshold) & (agent_distance < penalty_threshold)
        safe_distance_reward[mask] = (agent_distance[mask] - distance_threshold) / (penalty_threshold - distance_threshold) * 2.0
        safe_distance_reward = self._check_reward_shape(safe_distance_reward)
        rew_dict["safe_distance"] = torch.mean(safe_distance_reward)

        # Reward for maintaining small command values - max 1.0
        command_values = self._command_value(state, action)
        small_command_reward = 1.0 * torch.exp(-2.0 * command_values)
        small_command_reward = self._check_reward_shape(small_command_reward)
        rew_dict["small_command"] = torch.mean(small_command_reward)

        # Reward for sequential success - high weight of 10.0
        success = self._success_evaluation(state, action)
        sequential_success_reward = 10.0 * success.float()
        sequential_success_reward = self._check_reward_shape(sequential_success_reward)
        rew_dict["agent_0_sequential_success"] = torch.mean(sequential_success_reward[:, 0])
        rew_dict["agent_1_sequential_success"] = torch.mean(sequential_success_reward[:, 1])

        # Reward to ensure one agent pauses for sequential passing - max 5.0
        # Here we reward agents that don't pass the gate at the same time
        only_one_agent_success = torch.logical_or(success[:,0] > success[:,1], success[:,1] > success[:,0]).float()
        only_one_agent_reward = 5.0 * only_one_agent_success
        only_one_agent_reward = self._check_reward_shape(only_one_agent_reward)
        rew_dict["sequential_coordination"] = torch.mean(only_one_agent_reward)

        # Calculate other rewards combined
        other_rewards = progress_reward + safe_distance_reward + small_command_reward + sequential_success_reward + only_one_agent_reward

        # Reward for getting agent pairs to coordinate properly to pass the gate
        both_agents_success = torch.all(success > 0, dim=1, keepdim=True).float()
        both_agents_success = self._check_reward_shape(both_agents_success)
        reward = torch.where(both_agents_success > 0.5, max_reward, other_rewards)

        # Normalize the reward
        reward = self._check_reward_shape(reward)
        reward *= 1 / max_reward

        return reward, rew_dict, max_reward

