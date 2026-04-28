import os
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from mqe import LEGGED_GYM_ROOT_DIR
from mqe.envs.npc.go2_object import Go2Object


class Go2PushboxObject(Go2Object):
    """Go2Object subclass for the pushbox task.

    Adds a visualization-only target marker (NPC index 1) placed at a random
    target position each episode.  The marker has no collision and a fixed base,
    so it floats at the target location.  Its world position is the authoritative
    source of truth for target_pos — the wrapper reads it directly from
    root_states_npc rather than maintaining a separate tensor.
    """

    def _prepare_npc(self):
        super()._prepare_npc()
        marker_path = self.cfg.asset.file_npc_marker.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        marker_opts = gymapi.AssetOptions()
        marker_opts.fix_base_link = True
        marker_opts.disable_gravity = True
        self.asset_marker = self.gym.load_asset(
            self.sim, os.path.dirname(marker_path), os.path.basename(marker_path), marker_opts)

    def _create_npc(self, env_handle, env_id):
        pos = self.env_origins[env_id].clone()
        self.start_pose_npc.p = gymapi.Vec3(*pos)
        box_handle = self.gym.create_actor(
            env_handle, self.asset_npc, self.start_pose_npc,
            self.cfg.asset.name_npc, env_id, not self.npc_collision, 0)
        marker_handle = self.gym.create_actor(
            env_handle, self.asset_marker, self.start_pose_npc,
            self.cfg.asset.name_npc_marker, env_id, 1, 0)
        return [box_handle, marker_handle]

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        agent_ids = self.env_agent_indices[env_ids].reshape(-1)
        npc_ids = self.env_npc_indices[env_ids].reshape(-1)
        self.root_states[agent_ids] = self.base_init_state[agent_ids]
        self.root_states[agent_ids, :3] += self.agent_origins[env_ids].reshape(-1, 3)
        if self.num_npcs:
            self.root_states_npc[npc_ids] = self.base_init_state_npc[npc_ids]
            self.root_states_npc[npc_ids, :3] += self.env_origins[env_ids].unsqueeze(1).repeat(1, self.num_npcs, 1).reshape(-1, 3)

        # # Only randomize the target marker's position, which is the second NPC (index 1)
        # target_marker_ids = npc_ids[1::self.num_npcs]  # Assuming the marker is every second NPC
        # target_x_range = [-0.2, 0.5]  # Example range for target x position
        # target_y_range = [-1.5, 1.5]  # Example range for target y position
        # self.root_states_npc[target_marker_ids, 0:1] += torch_rand_float(*target_x_range, (len(target_marker_ids), 1), device=self.device)
        # self.root_states_npc[target_marker_ids, 1:2] += torch_rand_float(*target_y_range, (len(target_marker_ids), 1), device=self.device)

        # Only randomize the target marker's position, which is the second NPC (index 1)
        target_marker_ids = npc_ids[1::self.num_npcs]  # Assuming the marker is every second NPC
        target_x_range = [-0.5, 0.5]  # Example range for target x position
        target_y_range = [-2.75, 2.75]  # Example range for target y position
        self.root_states_npc[target_marker_ids, 0:1] += torch_rand_float(*target_x_range, (len(target_marker_ids), 1), device=self.device)
        self.root_states_npc[target_marker_ids, 1:2] += torch_rand_float(*target_y_range, (len(target_marker_ids), 1), device=self.device)

        # base velocities
        if getattr(self.cfg.domain_rand, "init_base_vel_range", None) is None:
            base_vel_range = (-0.5, 0.5)
        else:
            base_vel_range = self.cfg.domain_rand.init_base_vel_range
        self.root_states[agent_ids, 7:13] = torch_rand_float(
            *base_vel_range,
            (len(agent_ids), 6),
            device=self.device, 
        ) # [7:10]: lin vel, [10:13]: ang vel
        agent_indices_long = self.agent_indices[env_ids].reshape(-1).long()
        npc_indices_long = self.npc_indices[env_ids].reshape(-1).long()
        self.all_root_states[agent_indices_long] = self.root_states[agent_ids]
        self.all_root_states[npc_indices_long] = self.root_states_npc[npc_ids]
        actor_ids_int32 = self.actor_indices[env_ids].view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))