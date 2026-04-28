import numpy as np
from mqe.envs.go1.go1_config import Go1Cfg
from mqe.utils.helpers import merge_dict
from mqe.envs.go2.go2 import Go2Cfg

class Go2PushboxCfg(Go2Cfg):

    class env(Go2Cfg.env):
        env_name = "go2pushbox"
        num_envs = 1
        num_agents = 2
        num_npcs = 2
        episode_length_s = 25

    class asset(Go2Cfg.asset):
        terminate_after_contacts_on = []
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/box.urdf"
        name_npc = "box"
        npc_collision = True
        fix_npc_base_link = False
        npc_gravity = True
        file_npc_marker = "{LEGGED_GYM_ROOT_DIR}/resources/objects/cylinder_red.urdf"
        # file_npc_marker = "{LEGGED_GYM_ROOT_DIR}/resources/objects/cylinder_blue.urdf"
        name_npc_marker = "target_marker"
    
    class terrain(Go2Cfg.terrain):

        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go2Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "wall",
            ],
            # wall_thickness= 0.2,
            track_width = 7.2,
            init = dict(
                block_length = 8.4,
                room_size = (8.4, 3.6),
                border_width = 0.0,
                offset = (0, 0),
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False,
       ))
        
    class command(Go2Cfg.command):

        class cfg(Go2Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go2Cfg.init_state):
        multi_init_state = True
        init_state_class = Go2Cfg.init_state
        init_states = [ # Quadruped 1, Quadruped 2
            init_state_class(
                pos = [-3.3, 0.9, 0.36],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [-3.3, -0.9, 0.36],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            init_state_class(
                pos = [2.5, 0.0, 0.6], # Box
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            # Target marker (visualization only; position overridden per episode in Go2PushboxObject)
            init_state_class(
                pos = [5.5, 0.0, 0.05],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class control(Go2Cfg.control):
        control_type = 'C'

    class termination(Go2Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
        ]

    class domain_rand(Go2Cfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False # use for non-virtual training
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )
        init_npc_base_pos_range = dict(
            x= [-0.5, 0.5],
            y= [-0.5, 0.5],
        )

    class rewards(Go2Cfg.rewards):
        class scales:
            box_x_movement_reward_scale = 10
            # tracking_ang_vel = 0.05
            # world_vel_l2norm = -1.
            # legs_energy_substeps = -1e-5
            # alive = 2.
            # penetrate_depth = -3e-3
            # penetrate_volume = -3e-3
            # exceed_dof_pos_limits = -1e-1
            # exceed_torque_limits_i = -2e-1

    class viewer(Go2Cfg.viewer):
        pos = [4.8, 0.0, 5.5]
        lookat = [4.8, 3.5, 0.0]