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
            "agent_yaw": base_info[:, :, 5],
            "target_pos": self.target_pos,
            "seesaw_center": self.seesaw_center,
            "seesaw_start": self.seesaw_start,
            "seesaw_end": self.seesaw_end,
        }

        reward, reward_dict, max_reward = self.gpt_reward(global_state, action)
        
        # Update distances
        self._update_distances(global_state, action)

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

        return x_movement
    
    def _y_alignment(self, state, action):
        base_pos = state["agent_pos"]
        y_alignment = 1.0 - base_pos[:, :, 1] ** 2
        y_alignment = torch.clip(y_alignment, min=0.0, max=1.0)

        return y_alignment.float()
    
    def _progress_to_seesaw_start(self, state, action):
        base_pos = state["agent_pos"]
        seesaw_start_distance = torch.norm(base_pos[:, :, :2] - self.seesaw_start, p=2, dim=-1)

        if not hasattr(self, "last_seesaw_start_distance"):
            self.last_seesaw_start_distance = copy(seesaw_start_distance)
        seesaw_start_progress = self.last_seesaw_start_distance - seesaw_start_distance
        seesaw_start_progress[self.env.reset_ids] = 0.0

        # Scale the seesaw_start_progress to match the action scale
        seesaw_start_progress = seesaw_start_progress * 10.0

        # if agent is already at the start, return 1.0
        seesaw_start_progress[seesaw_start_distance < 0.5] = 1.0

        return seesaw_start_progress
    
    def _progress_to_seesaw_end(self, state, action):
        base_pos = state["agent_pos"]
        seesaw_end_distance = torch.norm(base_pos[:, :, :2] - self.seesaw_end, p=2, dim=-1)

        if not hasattr(self, "last_seesaw_end_distance"):
            self.last_seesaw_end_distance = copy(seesaw_end_distance)
        seesaw_end_progress = self.last_seesaw_end_distance - seesaw_end_distance
        seesaw_end_progress[self.env.reset_ids] = 0.0

        # Scale the seesaw_end_progress to match the action scale
        seesaw_end_progress = seesaw_end_progress * 10.0

        # if agent is already at the end, return 1.0
        seesaw_end_progress[seesaw_end_distance < 0.5] = 1.0

        return seesaw_end_progress

    def _progress_to_seesaw_center(self, state, action):
        base_pos = state["agent_pos"]
        seesaw_center_distance = torch.norm(base_pos[:, :, :] - self.seesaw_center, p=2, dim=-1)

        if not hasattr(self, "last_seesaw_center_distance"):
            self.last_seesaw_center_distance = copy(seesaw_center_distance)
        seesaw_center_progress = self.last_seesaw_center_distance - seesaw_center_distance
        seesaw_center_progress[self.env.reset_ids] = 0.0

        # Scale the seesaw_center_progress to match the action scale
        seesaw_center_progress = seesaw_center_progress * 10.0

        # if agent is already at the center, return 1.0
        seesaw_center_progress[seesaw_center_distance < 0.5] = 1.0

        return seesaw_center_progress

    def _progress_to_target(self, state, action):
        base_pos = state["agent_pos"]
        target_distance = torch.norm(base_pos[:, :, :] - self.target_pos[:, :, :], p=2, dim=-1)

        if not hasattr(self, "last_target_distance"):
            self.last_target_distance = copy(target_distance)
        target_progress = self.last_target_distance - target_distance
        target_progress[self.env.reset_ids] = 0.0

        # Scale the target_progress to match the action scale
        target_progress = target_progress * 10.0

        # if agent is already at the target, return 1.0
        target_progress[target_distance < 0.5] = 1.0

        return target_progress
    
    def _update_distances(self, state, action):
        self.last_x_pos = state["agent_pos"][:, :, 0].clone()
        self.last_seesaw_start_distance = torch.norm(state["agent_pos"][:, :, :2] - self.seesaw_start, p=2, dim=-1)
        self.last_seesaw_end_distance = torch.norm(state["agent_pos"][:, :, :2] - self.seesaw_end, p=2, dim=-1)
        self.last_seesaw_center_distance = torch.norm(state["agent_pos"][:, :, :] - self.seesaw_center, p=2, dim=-1)
        self.last_target_distance = torch.norm(state["agent_pos"][:, :, :] - self.target_pos[:, :, :], p=2, dim=-1)

    def _agent_distance(self, state, action):
        base_pos = state["agent_pos"]
        agent_dis = torch.norm(base_pos[:, 0, :] - base_pos[:, 1, :], p=2, dim=-1)

        agent_dis = agent_dis.unsqueeze(1).repeat(1, self.num_agents)

        return agent_dis.float()
    
    def _normalized_height(self, state, action):
        height = state["agent_pos"][:, :, 2]
        normalized_height = height - 0.3
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
        command_norm = 0.5 * torch.abs(action[:,:, 0]) + 2.0 * torch.abs(action[:, :, 1]) + 2.0 * torch.abs(action[:, :, 2])
        return command_norm
    
    def _yaw_alignment(self, state, action):
        yaw = state["agent_yaw"]
        yaw_penalty = torch.ones_like(yaw) - torch.abs(yaw) / 3.14

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
    
    def gpt_reward(self, state, action):
        device = self.env.device

        reward = torch.zeros((self.num_envs, self.num_agents), device=device)
        rew_dict = {}
        max_reward = 18.0

        agent_pos = state["agent_pos"]
        seesaw_start = state["seesaw_start"]
        seesaw_end = state["seesaw_end"]
        target_pos = state["target_pos"]

        # Dense forward/task progress
        x_prog = self._progress_in_x(state, action)
        x_prog = self._check_reward_shape(x_prog)
        reward += 0.6 * x_prog
        rew_dict["x_progress"] = torch.mean(x_prog)

        # Agent 0 should get onto the seesaw first and stabilize near the start
        a0_to_start = self._progress_to_seesaw_start(state, action)
        a0_to_start = self._check_reward_shape(a0_to_start)
        a0_start_reward = torch.zeros_like(a0_to_start)
        a0_start_reward[:, 0] = a0_to_start[:, 0]
        reward += 1.4 * a0_start_reward
        rew_dict["agent0_to_start"] = torch.mean(a0_start_reward[:, 0])

        # Keep agent 0 near the start zone once it has entered, to help stabilize the seesaw
        start_x = seesaw_start[..., 0]
        start_y = seesaw_start[..., 1]
        a0_x = agent_pos[:, 0, 0]
        a0_y = agent_pos[:, 0, 1]
        a0_hold = torch.exp(-((a0_x - start_x[:, 0]) ** 2) / 0.45) * torch.exp(-((a0_y - start_y[:, 0]) ** 2) / 0.20)
        a0_hold = torch.clamp(a0_hold, 0.0, 1.0)
        a0_hold_full = torch.zeros((self.num_envs, self.num_agents), device=device)
        a0_hold_full[:, 0] = a0_hold
        reward += 0.9 * a0_hold_full
        rew_dict["agent0_hold_start"] = torch.mean(a0_hold_full[:, 0])

        # Agent 1 should progress toward the far end and then the target platform
        a1_to_end = self._progress_to_seesaw_end(state, action)
        a1_to_end = self._check_reward_shape(a1_to_end)
        a1_end_reward = torch.zeros_like(a1_to_end)
        a1_end_reward[:, 1] = a1_to_end[:, 1]
        reward += 1.8 * a1_end_reward
        rew_dict["agent1_to_end"] = torch.mean(a1_end_reward[:, 1])

        a1_to_target = self._progress_to_target(state, action)
        a1_to_target = self._check_reward_shape(a1_to_target)
        a1_target_reward = torch.zeros_like(a1_to_target)
        a1_target_reward[:, 1] = a1_to_target[:, 1]
        reward += 2.2 * a1_target_reward
        rew_dict["agent1_to_target"] = torch.mean(a1_target_reward[:, 1])

        # Height shaping for successful climb onto the platform
        height = self._normalized_height(state, action)
        height = self._check_reward_shape(height)
        a1_height = torch.zeros_like(height)
        a1_height[:, 1] = torch.clamp(height[:, 1], 0.0, 1.0)
        reward += 1.0 * a1_height
        rew_dict["agent1_height"] = torch.mean(a1_height[:, 1])

        # Keep both agents near the seesaw center line to avoid falling off
        y_align = self._y_alignment(state, action)
        y_align = self._check_reward_shape(y_align)
        reward += 0.8 * y_align
        rew_dict["y_alignment"] = torch.mean(y_align)

        # Coordination spacing: avoid overlap but remain coordinated
        dist = self._agent_distance(state, action)
        dist_reward = torch.exp(-((dist - 1.5) ** 2) / 0.9)
        dist_reward = torch.clamp(dist_reward, 0.0, 1.0)
        dist_reward = self._check_reward_shape(dist_reward)
        reward += 0.6 * dist_reward
        rew_dict["agent_distance"] = torch.mean(dist_reward)

        # Small bonus if the leading agent is farther along x than the stabilizing agent
        lead_margin = torch.clamp(agent_pos[:, 1, 0] - agent_pos[:, 0, 0], min=-2.0, max=2.0)
        lead_reward = torch.sigmoid(lead_margin)
        lead_reward = lead_reward.unsqueeze(-1)
        lead_full = torch.zeros((self.num_envs, self.num_agents), device=device)
        lead_full[:, 1] = lead_reward[:, 0]
        reward += 0.4 * lead_full
        rew_dict["agent1_leads"] = torch.mean(lead_full[:, 1])

        # Safety penalties
        wall_collision = self._wall_collision(state, action).float()
        wall_collision = self._check_reward_shape(wall_collision)
        wall_pen = -1.5 * wall_collision
        reward += wall_pen
        rew_dict["wall_collision_penalty"] = torch.mean(wall_pen)

        fall = self._fall(state, action).float()
        fall = self._check_reward_shape(fall)
        fall_pen = -2.5 * fall
        reward += fall_pen
        rew_dict["fall_penalty"] = torch.mean(fall_pen)

        # Mild action regularization
        action_pen = -0.02 * torch.sum(action ** 2, dim=-1)
        action_pen = self._check_reward_shape(action_pen)
        reward += action_pen
        rew_dict["action_penalty"] = torch.mean(action_pen)

        # Terminal success bonus for either agent reaching the target platform
        success = self._success(state, action).float()
        success = self._check_reward_shape(success)
        success_reward = 10.0 * success
        reward += success_reward
        rew_dict["success"] = torch.mean(success_reward)

        reward = self._check_reward_shape(reward)
        reward = torch.clamp(reward / max_reward, -1.0, 1.0)

        return reward, rew_dict, max_reward

