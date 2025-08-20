import isaacgym
from openrl_ws.utils import make_env, get_args, MATWrapper
from openrl_ws.test import save_video, save_gif, save_images
from mqe.envs.utils import custom_cfg

from openrl.envs.common import make
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import numpy as np
import os

def eval_success(obs):
    agent_0_pos = obs[0, 0, 2:5]
    agent_1_pos = obs[0, 1, 2:5]

    platform_x = 7.7
    platform_z = 1.3

    agent_0_success = (agent_0_pos[0] > platform_x) and (agent_0_pos[2] > platform_z)
    agent_1_success = (agent_1_pos[0] > platform_x) and (agent_1_pos[2] > platform_z)

    success = agent_0_success or agent_1_success

    return success

def eval_partial_success(obs):
    agent_0_pos = obs[0, 0, 2:5]
    agent_1_pos = obs[0, 1, 2:5]

    platform_z = 0.6

    agent_0_success = (agent_0_pos[2] > platform_z)
    agent_1_success = (agent_1_pos[2] > platform_z)

    success = agent_0_success or agent_1_success

    return success

if __name__ == "__main__":
    args = get_args()
    env, _ = make_env(args, custom_cfg(args))

    # args.config = "./openrl_ws/cfgs/ppo.yaml" # Somehow this does not work
    args.lr = 7e-4
    args.critic_lr = 7e-4
    args.log_interval = 5
    # args.use_recurrent_policy = True 
    args.use_joint_action_loss = False
    args.use_valuenorm = True
    args.use_adv_normalize = True

    net = PPONet(env, device="cuda")  # Create neural network.
    agent = PPOAgent(net)  # Initialize the agent.

    if getattr(args, "checkpoint") is not None:
        agent.load(args.checkpoint)

    # env.start_recording()
    agent.set_env(env)  # The agent requires an interactive environment.
    eval_runs = 0
    success_runs = 0
    partial_success_runs = 0
    while eval_runs < 100:
        success = False
        partial_success = False
        obs = env.reset(seed=eval_runs)  # Initialize the environment to obtain initial observations and environmental information.
        while True:
            action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
            # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
            new_obs, r, done, info = env.step(action)
            success_step = eval_success(new_obs)
            partial_success_step = eval_partial_success(new_obs)
            success = success or success_step
            partial_success = partial_success or partial_success_step
            if done[0, 0]:
                print(f"Run {eval_runs} completed.")
                eval_runs += 1
                if np.linalg.norm(obs[0, 0, 2:4] - obs[0, 1, 2:4]) < 0.51:
                    print("Terminated due to close proximity")
                else:
                    print("Terminated due to other reasons")

                if success:
                    print("Total success!")
                    success_runs += 1
                elif partial_success:
                    print("Partial success!")
                    partial_success_runs += 1

                if env.cfg.env.record_video:
                    frames = env.get_complete_frames()
                    video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
                    eval_dir = os.path.join(args.checkpoint, f"eval_run_{eval_runs}.mp4")
                    save_video(video_array, 50, eval_dir)
                break
            obs = new_obs  # Update the observation for the next step
                
    print(f"Total runs: {eval_runs}, Successful runs: {success_runs}, Success rate: {success_runs / eval_runs * 100:.2f}%")
    print(f"Partial success runs: {partial_success_runs}, Partial success rate: {partial_success_runs / eval_runs * 100:.2f}%")