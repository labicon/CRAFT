import isaacgym
from openrl_ws.utils import make_env, get_args, MATWrapper
from openrl_ws.test import save_video, save_gif, save_images
from mqe.envs.utils import custom_cfg

from openrl.envs.common import make
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import numpy as np
import os

def gate_success(obs):
    gate_pos = obs[0, 0, 8:10]
    agent_0_pos = obs[0, 0, 2:4]
    agent_1_pos = obs[0, 1, 2:4]

    agent_0_success = agent_0_pos[0] > gate_pos[0] + 0.25
    agent_1_success = agent_1_pos[0] > gate_pos[0] + 0.25

    return agent_0_success, agent_1_success

if __name__ == "__main__":
    args = get_args()
    env, _ = make_env(args, custom_cfg(args))

    # args.config = "./openrl_ws/cfgs/ppo.yaml" # Somehow this does not work
    args.lr = 7e-4
    args.critic_lr = 7e-4
    args.log_interval = 5
    args.use_recurrent_policy = True
    args.use_joint_action_loss = False
    args.use_valuenorm = True
    args.use_adv_normalize = True

    net = PPONet(env, device="cuda")  # Create neural network.
    agent = PPOAgent(net)  # Initialize the agent.

    if args.algo == "jrpo" or args.algo == "ppo":
        from openrl.modules.common import PPONet
        from openrl.runners.common import PPOAgent
        net = PPONet(env, cfg=args, device=args.rl_device)
        agent = PPOAgent(net)
    else:
        from openrl.modules.common import MATNet
        from openrl.runners.common import MATAgent
        env = MATWrapper(env)
        net = MATNet(env, cfg=args, device=args.rl_device)
        agent = MATAgent(net, use_wandb=args.use_wandb)

    if getattr(args, "checkpoint") is not None:
        agent.load(args.checkpoint)

    # env.start_recording()
    agent.set_env(env)  # The agent requires an interactive environment.
    seed = np.random.randint(0, 10000)
    obs = env.reset(seed=seed)  # Initialize the environment to obtain initial observations and environmental information.
    eval_runs = 0
    success_runs = 0
    while eval_runs < 100:
        action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
        # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
        obs, r, done, info = env.step(action)
        # print("Obs:", obs)
        # print("Action:", action)
        # print("Target pos:", env.target_pos)
        # print("Gate pos:", env.gate_pos)
        # for key, value in info[0].items():
        #     print(f"{key}: {value}")
        if done[0, 0]:
            eval_runs += 1
            print(f"Run {eval_runs} completed.")
            if np.linalg.norm(obs[0, 0, 2:4] - obs[0, 1, 2:4]) < 0.51:
                print("Terminated due to close proximity")
            else:
                print("Terminated due to other reasons")

            if gate_success(obs):
                print("Gate success!")
                success_runs += 1
        if done[0, 0] and env.cfg.env.record_video:
            frames = env.get_complete_frames()
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            save_video(video_array, 50, f"eval_run_{eval_runs}.mp4")

    print(f"Total runs: {eval_runs}, Successful runs: {success_runs}, Success rate: {success_runs / eval_runs * 100:.2f}%")