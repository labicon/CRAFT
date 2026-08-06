from openrl_ws.utils import make_env, get_args
from mqe.envs.utils import custom_cfg
from openrl.utils.logger import Logger
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import argparse
import os
import shutil
import sys


def train(task, save_dir, exp_name, training_iter, seed, sim_device="cuda:0", rl_device="cuda:0", graphics_device_id=0):
    from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback

    args = get_args()
    args.task = task
    args.seed = seed
    args.num_envs = 500
    args.headless = True
    args.separate_policy = True
    args.train_timesteps = training_iter
    args.sim_device = sim_device
    args.sim_device_id = int(sim_device.split(":")[-1]) if ":" in sim_device else 0
    args.rl_device = rl_device
    args.graphics_device_id = graphics_device_id

    env, _ = make_env(args, custom_cfg(args), single_agent=False)

    args.lr = 1e-3
    args.critic_lr = 1e-3
    args.log_interval = 5
    # args.use_recurrent_policy = True
    args.use_joint_action_loss = False
    args.use_valuenorm = True
    args.use_adv_normalize = True
    args.entropy_coef = 0.001
    args.num_mini_batch = 4

    callback = CheckpointCallback(save_freq=5000, save_path=save_dir)
    net = PPONet(env, cfg=args, device=args.rl_device)
    agent = PPOAgent(net)
    logger = Logger(
        cfg=net.cfg,
        project_name="MQE-Eureka",
        scenario_name=args.task,
        wandb_entity="kh-ryu-university-of-california-berkeley",
        exp_name=exp_name,
        log_path="./training-log",
        use_wandb=True,
        use_tensorboard=True,
    )

    print(f"Start Eureka training for {task}: {exp_name}")
    agent.train(total_time_steps=args.train_timesteps, callback=callback, logger=logger)
    agent.save(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one Eureka reward candidate")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--training_iter", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim_device", type=str, default="cuda:0", help="Physics simulation device (e.g. cuda:0, cuda:1)")
    parser.add_argument("--rl_device", type=str, default="cuda:0", help="RL algorithm device (e.g. cuda:0, cuda:1)")
    parser.add_argument("--graphics_device_id", type=int, default=0, help="GPU index for rendering (e.g. 0, 1)")
    args = parser.parse_args()

    task = args.task
    save_dir = args.save_dir
    exp_name = args.exp_name
    training_iter = args.training_iter
    seed = args.seed
    sim_device = args.sim_device
    rl_device = args.rl_device
    graphics_device_id = args.graphics_device_id

    del args, parser
    sys.argv = [sys.argv[0]]

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train(task, save_dir, exp_name, training_iter, seed, sim_device, rl_device, graphics_device_id)
