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
        self.frame_left = self.gate_pos.reshape(-1, 2)
        self.frame_right = self.gate_pos.reshape(-1, 2)
        self.frame_left[:, 1] += self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.frame_right[:, 1] -= self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

        self.target_pos = self.gate_pos.clone()
        self.target_pos[:, :, 0] += 1.0
        # self.target_pos = torch.zeros_like(self.gate_pos, dtype=self.gate_pos.dtype, device=self.gate_pos.device)
        # self.target_pos[:, :, 0] = self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] + self.BarrierTrack_kwargs["plane"]["block_length"] / 2
        # self.target_pos[:, 0, 1] = self.BarrierTrack_kwargs["track_width"] / 4
        # self.target_pos[:, 1, 1] = - self.BarrierTrack_kwargs["track_width"] / 4
        # self.target_pos = self.target_pos.reshape(-1, 2)

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
            "agent_yaw": base_info[:, :, 5],
        }

        self.reward_buffer["step count"] += 1

        reward, reward_dict, max_reward = self.gpt_reward(global_state, action)
        for key, value in reward_dict.items():
            reward_key = f"Reward/{key}"
            self.reward_buffer[reward_key] = value

        return obs, reward, termination, self.reward_buffer
    
    def _progress_to_gate(self, state, action):
        base_pos = state["agent_pos"]
        gate_pos = state["gate_pos"]

        # approach reward
        distance_to_gate = torch.norm(base_pos - gate_pos, p=2, dim=-1)

        if not hasattr(self, "last_distance_to_gate"):
            self.last_distance_to_gate = copy(distance_to_gate)

        progress = (self.last_distance_to_gate - distance_to_gate)
        progress[self.env.reset_ids] = 0

        self.last_distance_to_gate = copy(distance_to_gate)

        return progress

    def _distance_to_gate(self, state, action):
        base_pos = state["agent_pos"]
        gate_pos = state["gate_pos"]

        # approach reward
        distance_to_gate = torch.norm(base_pos[:, :, :2] - gate_pos, p=2, dim=-1)
        return distance_to_gate
    
    def _get_gate_frame(self, state, action):
        frame_left = self.frame_left.reshape(self.num_envs, self.num_agents, -1)
        frame_left = frame_left[:, :, 1].unsqueeze(2)
        frame_right = self.frame_right.reshape(self.num_envs, self.num_agents, -1)
        frame_right = frame_right[:, :, 1].unsqueeze(2)

        gate_frame = torch.cat([frame_left, frame_right], dim=-1)

        return gate_frame

    def _contact_punishment(self, state, action):
        collide_reward = torch.tensor(self.env.collide_buf, dtype=torch.float)
        return collide_reward.unsqueeze(1).repeat(1, self.num_agents)

    def _agent_distance(self, state, action):
        base_pos = state["agent_pos"]
        # Compute pairwise distance between agents
        agent_0_pos = base_pos[:, 0, :2]
        agent_1_pos = base_pos[:, 1, :2]
        distance = torch.norm(agent_0_pos - agent_1_pos, p=2, dim=-1)

        # Reshape to match the number of agents
        distance = distance.unsqueeze(1)
        return distance
    
    def _command_lin_vel_y(self, state, action):
        # command lin_vel.y punishment
        v_y_punishment = action[:, :, 1]
        return v_y_punishment
    
    def _command_lin_vel_x(self, state, action):
        # lin_vel.x reward
        v_x_reward = action[:, :, 0]
        return v_x_reward
    
    def _command_value(self, state, action):
        # command value punishment
        command_value_punishment = torch.norm(action, p=2, dim=2)
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
    

    def gpt_reward(self, state, action):
        reward = torch.zeros((self.num_envs, self.num_agents), device=self.env.device)
        rew_dict = {}
        max_reward = 4.0

        # Reward for both agents successfully passing the gate - max 1.0
        success = self._success_evaluation(state, action)
        team_success_reward = torch.mean(success, dim=1, keepdim=True).float()
        rew_dict["team_success"] = torch.mean(team_success_reward)

        # Progress to gate, gated by safe separation - max 1.0
        progress_to_gate = self._progress_to_gate(state, action)
        agent_distance = self._agent_distance(state, action)
        safe_distance = agent_distance >= 0.25
        gated_progress_reward = progress_to_gate * safe_distance.float()
        rew_dict["agent_0_progress"] = torch.mean(gated_progress_reward[:, 0])
        rew_dict["agent_1_progress"] = torch.mean(gated_progress_reward[:, 1])

        # Separation bonus - max 1.0
        D_max = 0.5
        separation_bonus = torch.clamp(agent_distance - 0.25, 0.0, D_max) / D_max
        rew_dict["separation_bonus"] = torch.mean(separation_bonus)

        # Synchronization reward - max 1.0
        delta_progress = torch.abs(gated_progress_reward[:, 0] - gated_progress_reward[:, 1])
        alpha = 5.0
        synchronization_reward = torch.exp(-alpha * delta_progress)
        rew_dict["synchronization"] = torch.mean(synchronization_reward)

        # Combined rewards
        reward_components = team_success_reward + gated_progress_reward + separation_bonus.unsqueeze(-1) + synchronization_reward.unsqueeze(-1)
        reward = reward_components.sum(dim=1)

        # Normalize the reward
        reward = torch.clamp(reward, 0.0, max_reward) * (1.0 / max_reward)

        return reward, rew_dict, max_reward

