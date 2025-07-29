import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go2.go2 import Go2Cfg

class Go2SwitchCfg(Go2Cfg):

    class env(Go2Cfg.env):
        env_name = "go2switch"
        num_envs = 1
        num_agents = 2
        num_npcs = 2
        episode_length_s = 30 # episode length in seconds

    class asset(Go2Cfg.asset):
        terminate_after_contacts_on = [] # ["base", "Head_upper"]
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/cylinder_blue.urdf"
        name_npc = "goal_0"
        npc_collision = False
        fix_npc_base_link = True
        npc_gravity = True

        terminate_after_contacts_on = [] # ["base", "Head_upper"]
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/cylinder_blue.urdf"
        name_npc = "goal_1"
        npc_collision = False
        fix_npc_base_link = True
        npc_gravity = True

    class terrain(Go2Cfg.terrain):

        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go2Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "plane",
                "wall",
            ],
            # wall_thickness= 0.2,
            track_width = 6.0,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 0.5,
                room_size = (0.0, 0.0),
                border_width = 0.00,
                offset = (0, 0),
            ),
            plane = dict(
                block_length = 5.0,
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
        init_states = [
            init_state_class(
                pos = [1.0, 0.0, 0.36],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [4.0, 0.0, 0.36],
                rot = [0.0, 0.0, 1.0, 0.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            init_state_class(
                pos = [4.0, 0.0, 0.0],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [1.0, 0.0, 0.0],
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
            "z_low",
            "z_high",
        ]

    class domain_rand(Go2Cfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False # use for non-virtual training
        init_dof_pos_ratio_range = None
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

            target_reward_scale = 1
            punishment_scale = -1
            success_reward_scale = 10

    class viewer(Go2Cfg.viewer):
        pos = [3.2, 0., 4.]
        lookat = [3.2, 2.5, 0.]