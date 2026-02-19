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

        self.contact_force_threshold = 0.1
        self.contact_distance_threshold = 0.7
        self.head_contact_reward_scale = 2.0
        self.non_head_contact_penalty_scale = -3.0
        self.debug_contact_reward = False
        self.debug_print_interval = 50
        self._head_body_indices = []
        self._non_head_body_indices = []
        self._cache_contact_body_indices()

    def _cache_contact_body_indices(self):
        if not hasattr(self.env, "body_name_to_index"):
            return
        head_names = ["Head_upper", "Head_lower"]
        non_head_keywords = ["thigh", "calf", "hip", "base"]
        head_indices = [self.env.body_name_to_index.get(name) for name in head_names]
        head_indices = [idx for idx in head_indices if idx is not None]
        non_head_indices = []
        for name, idx in self.env.body_name_to_index.items():
            lname = name.lower()
            if any(keyword in lname for keyword in non_head_keywords):
                non_head_indices.append(idx)
        non_head_indices = [idx for idx in non_head_indices if idx not in head_indices]
        self._head_body_indices = sorted(set(head_indices))
        self._non_head_body_indices = sorted(set(non_head_indices))

    def _agent_body_contact_forces(self, body_indices):
        if not body_indices:
            return torch.zeros(self.num_envs, self.num_agents, 0, device=self.device)
        base = torch.tensor(body_indices, device=self.device)
        agent_offsets = torch.arange(self.num_agents, device=self.device).unsqueeze(1) * self.env.num_bodies
        idxs = (base.unsqueeze(0) + agent_offsets).reshape(-1)
        forces = self.env.contact_forces[:, idxs, :].view(self.num_envs, self.num_agents, -1, 3)
        return torch.norm(forces, dim=-1)

    def _contact_mask_for_bodies(self, body_indices):
        if not body_indices or not hasattr(self.env, "robot_rigid_body_state"):
            return torch.zeros(self.num_envs, self.num_agents, 0, device=self.device, dtype=torch.bool)
        body_pos = self.env.robot_rigid_body_state[:, :, body_indices, 0:3]
        # Use world-frame box position to match rigid-body world-frame coordinates.
        box_pos = self.root_states_npc.view(self.num_envs, self.num_npcs, 13)[:, 0, :3]
        box_pos = box_pos.unsqueeze(1).unsqueeze(2)
        dist = torch.norm(body_pos[..., :2] - box_pos[..., :2], dim=-1)
        forces = self._agent_body_contact_forces(body_indices)
        return (forces > self.contact_force_threshold) & (dist < self.contact_distance_threshold)

    def _raw_contact_rate(self, body_indices):
        forces = self._agent_body_contact_forces(body_indices)
        if forces.numel() == 0:
            return 0.0
        return float((forces > self.contact_force_threshold).any(dim=-1).float().mean().item())

    def _gated_contact_rate(self, body_indices):
        contact_mask = self._contact_mask_for_bodies(body_indices)
        if contact_mask.numel() == 0:
            return 0.0
        return float(contact_mask.any(dim=-1).float().mean().item())

    def _head_box_collision_flag(self):
        """Return batched 0/1 flags with shape (num_envs, num_agents)."""
        head_contact_mask = self._contact_mask_for_bodies(self._head_body_indices)
        if head_contact_mask.numel() == 0:
            return torch.zeros(self.num_envs, self.num_agents, device=self.device)
        return head_contact_mask.any(dim=-1).float()

    def _non_head_box_collision_flag(self):
        """Return batched 0/1 flags with shape (num_envs, num_agents)."""
        non_head_contact_mask = self._contact_mask_for_bodies(self._non_head_body_indices)
        if non_head_contact_mask.numel() == 0:
            return torch.zeros(self.num_envs, self.num_agents, device=self.device)
        return non_head_contact_mask.any(dim=-1).float()

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

        if self.debug_contact_reward and self.reward_buffer["step count"] % self.debug_print_interval == 0:
            def _to_float(v):
                if isinstance(v, torch.Tensor):
                    return float(v.detach().mean().item())
                return float(v)
            print(
                "[Go2PushBox Debug] "
                f"step={self.reward_buffer['step count']} "
                f"raw_head_contact_rate={self._raw_contact_rate(self._head_body_indices):.4f} "
                f"raw_non_head_contact_rate={self._raw_contact_rate(self._non_head_body_indices):.4f} "
                f"gated_head_contact_rate={self._gated_contact_rate(self._head_body_indices):.4f} "
                f"gated_non_head_contact_rate={self._gated_contact_rate(self._non_head_body_indices):.4f} "
                f"head_contact={_to_float(reward_dict.get('head_contact', 0.0)):.4f} "
                f"non_head_contact_penalty={_to_float(reward_dict.get('non_head_contact_penalty', 0.0)):.4f} "
                f"collision_avoidance={_to_float(reward_dict.get('collision_avoidance', 0.0)):.4f} "
                f"reward_mean={float(reward.detach().mean().item()):.4f}"
            )

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

