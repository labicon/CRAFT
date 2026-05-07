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

    @property
    def target_pos(self):
        """Target position relative to env_origin, read from the marker's sim state (NPC 1)."""
        npc_states = self.root_states_npc.view(self.num_envs, self.num_npcs, 13)
        return npc_states[:, 1, :2] - self.env.env_origins[:, :2]

    def reset(self):
        obs_buf = self.env.reset()

        # Box is NPC 0; target marker is NPC 1.  Target position is read directly
        # from the sim (set by Go2PushboxObject._reset_root_states).
        npc_states = self.root_states_npc.view(self.num_envs, self.num_npcs, 13)
        box_state = npc_states[:, 0, :]
        box_pos = box_state[:, :3] - self.env.env_origins
        # get_euler_xyz returns (roll, pitch, yaw) as tuple of tensors
        _, _, box_yaw = get_euler_xyz(box_state[:, 3:7])
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

        # Box is NPC 0; target marker is NPC 1.  Target is read from the sim —
        # Go2PushboxObject._reset_root_states already randomized it for reset envs.
        npc_states = self.root_states_npc.view(self.num_envs, self.num_npcs, 13)
        box_state = npc_states[:, 0, :]
        box_pos = box_state[:, :3] - self.env.env_origins
        # get_euler_xyz returns (roll, pitch, yaw) as tuple of tensors
        _, _, box_yaw = get_euler_xyz(box_state[:, 3:7])
        box_yaw = (box_yaw + np.pi) % (2 * np.pi) - np.pi

        # Update last_box_pos for auto-reset envs
        if len(self.env.reset_ids) > 0:
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
            self.reward_buffer[reward_key] = self.reward_buffer.get(reward_key, 0) + value * self.num_envs * self.num_agents

        eval_dict = self._eval(global_state, action)
        for key, value in eval_dict.items():
            eval_key = f"Eval/{key}"
            self.reward_buffer[eval_key] = self.reward_buffer.get(eval_key, 0) + value * self.num_envs * self.num_agents

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
        success = (dist <= 0.2).float()
        return success

    def _agent_yaw_error_to_box(self, state, action):
        """Absolute angular error (radians, [0, pi]) between each agent's yaw and the box's yaw.
        Returns tensor (num_envs, num_agents). Smaller means better alignment for pushing."""
        agent_yaw = state["agent_yaw"]  # (num_envs, num_agents, 1)
        box_yaw = state["box_yaw"]      # (num_envs, num_agents, 1)
        diff = agent_yaw - box_yaw
        error = torch.abs((diff + torch.pi) % (2 * torch.pi) - torch.pi)
        return self._check_reward_shape(error.squeeze(-1))

    def _box_yaw_to_target(self, state, action):
        """Absolute angular error (radians, [0, pi]) between the box's current yaw and the
        direction from the box toward the target. Returns tensor (num_envs, num_agents).
        Smaller means the box is oriented to be pushed straight toward the target."""
        box_pos = state["box_pos"]      # (num_envs, num_agents, 2)
        target_pos = state["target_pos"]  # (num_envs, num_agents, 2)
        box_yaw = state["box_yaw"]      # (num_envs, num_agents, 1)
        delta = target_pos - box_pos    # (num_envs, num_agents, 2)
        desired_yaw = torch.atan2(delta[..., 1], delta[..., 0])  # (num_envs, num_agents)
        diff = desired_yaw - box_yaw.squeeze(-1)
        error = torch.abs((diff + torch.pi) % (2 * torch.pi) - torch.pi)
        return self._check_reward_shape(error)

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

    def gpt_reward(self, state, action):
        device = self.env.device
        reward = torch.zeros((self.num_envs, self.num_agents), device=device)
        rew_dict = {}
        max_reward = 10.0

        # Core task signal: make progress toward the target
        toward_target = self._box_movement_toward_target(state, action)
        dist_to_target = self._box_distance_to_target(state, action)
        box_yaw_to_target = self._box_yaw_to_target(state, action)

        # Stronger emphasis on actual progress, but bounded for PPO stability
        progress_reward = 2.2 * torch.tanh(1.2 * toward_target)
        progress_reward = self._check_reward_shape(progress_reward)
        rew_dict["box_progress"] = torch.mean(progress_reward)

        # Dense proximity shaping to keep the box moving toward the goal
        proximity_reward = 1.6 * torch.exp(-1.4 * dist_to_target)
        proximity_reward = self._check_reward_shape(proximity_reward)
        rew_dict["target_proximity"] = torch.mean(proximity_reward)

        # Encourage the box to rotate toward the target direction for steering
        # This helps when the target is off-axis and reduces sideways drift
        yaw_to_target_reward = 0.8 * torch.exp(-2.0 * box_yaw_to_target)
        yaw_to_target_reward = self._check_reward_shape(yaw_to_target_reward)
        rew_dict["box_yaw_to_target"] = torch.mean(yaw_to_target_reward)

        # Both agents should align yaw with the box yaw for efficient push contact
        agent_yaw_error = self._agent_yaw_error_to_box(state, action)
        yaw_alignment_reward = 1.15 * torch.exp(-2.2 * agent_yaw_error)
        yaw_alignment_reward = self._check_reward_shape(yaw_alignment_reward)
        rew_dict["agent_yaw_alignment"] = torch.mean(yaw_alignment_reward)

        # Keep agents near the back face of the box so they can sustain push contact
        agent_box_dist = self._agents_distance_to_box(state, action)
        push_distance_reward = 0.65 * torch.exp(-1.6 * agent_box_dist)
        push_distance_reward = self._check_reward_shape(push_distance_reward)
        rew_dict["agent_box_proximity"] = torch.mean(push_distance_reward)

        # Reward head contact specifically, since the task uses the head links for pushing
        head_contact = self._head_box_collision_flag().float()
        head_contact_reward = 1.05 * head_contact
        head_contact_reward = self._check_reward_shape(head_contact_reward)
        rew_dict["head_contact"] = torch.mean(head_contact_reward)

        # Penalize any box contact from non-head body parts to discourage unsafe pushing
        non_head_collision = self._non_head_box_collision_flag().float()
        non_head_collision_penalty = -1.8 * non_head_collision
        non_head_collision_penalty = self._check_reward_shape(non_head_collision_penalty)
        rew_dict["non_head_collision_penalty"] = torch.mean(non_head_collision_penalty)

        # Mild penalty for excessive lateral displacement of the box.
        # This still allows turning, but discourages uncontrolled side-slip.
        box_y_dev = self._box_y_deviation(state, action)
        y_deviation_penalty = -0.25 * torch.tanh(1.0 * box_y_dev)
        y_deviation_penalty = self._check_reward_shape(y_deviation_penalty)
        rew_dict["box_y_deviation_penalty"] = torch.mean(y_deviation_penalty)

        # Success bonus for reaching the target region
        success = self._success_evaluation(state, action).float()
        success_bonus = 2.5 * success
        success_bonus = self._check_reward_shape(success_bonus)
        rew_dict["success_bonus"] = torch.mean(success_bonus)

        # Mild action regularization to reduce thrashing and unstable oscillations
        action_l2 = torch.sum(action ** 2, dim=-1)
        action_penalty = -0.02 * action_l2
        action_penalty = self._check_reward_shape(action_penalty)
        rew_dict["action_penalty"] = torch.mean(action_penalty)

        # Total reward
        reward = (
            progress_reward
            + proximity_reward
            + yaw_to_target_reward
            + yaw_alignment_reward
            + push_distance_reward
            + head_contact_reward
            + non_head_collision_penalty
            + y_deviation_penalty
            + success_bonus
            + action_penalty
        )

        # Keep the reward in a practical PPO range and avoid extreme outliers
        reward = torch.clamp(reward, -2.0, max_reward)
        reward = reward / max_reward

        return reward, rew_dict, max_reward
