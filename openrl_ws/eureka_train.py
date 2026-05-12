from openrl_ws.utils import make_env, get_args
from mqe.envs.utils import custom_cfg
from openrl.utils.logger import Logger
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import argparse
import os
import shutil
import sys


def train(task, save_dir, exp_name, training_iter, seed):
    from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback

    args = get_args()
    args.task = task
    args.seed = seed
    args.num_envs = 500
    args.headless = True
    args.separate_policy = True
    args.train_timesteps = training_iter

    env, _ = make_env(args, custom_cfg(args), single_agent=False)

    args.lr = 7e-4
    args.critic_lr = 7e-4
    args.log_interval = 5
    args.use_joint_action_loss = False
    args.use_valuenorm = True
    args.use_adv_normalize = True
    args.entropy_coef = 0.0001

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
    args = parser.parse_args()

    task = args.task
    save_dir = args.save_dir
    exp_name = args.exp_name
    training_iter = args.training_iter
    seed = args.seed

    del args, parser
    sys.argv = [sys.argv[0]]

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train(task, save_dir, exp_name, training_iter, seed)
