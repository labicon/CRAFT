from openrl_ws.utils import make_env, get_args
from mqe.envs.utils import custom_cfg
from openrl.utils.logger import Logger
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import os
import shutil
import argparse
import sys

def train(save_dir, exp_name, training_iter=1000000):
    from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
    args = get_args()  
    args.num_envs = 200
    args.headless = True
    env, env_cfg = make_env(args, custom_cfg(args), single_agent=False)
    
    args.config = "./openrl_ws/cfgs/ppo.yaml"
    args.train_timesteps = training_iter

    callback = CheckpointCallback(
        save_freq=5000,
        save_path=save_dir)

    net = PPONet(env, cfg=args, device=args.rl_device)
    agent = PPOAgent(net)
    logger = Logger(
        cfg=net.cfg,
        project_name="MQE-Curriculum",
        scenario_name=args.task,
        wandb_entity="kh-ryu-university-of-california-berkeley",
        exp_name=exp_name,
        log_path="./log",
        use_wandb=True,
        use_tensorboard=False,
    )

    print("Start training from scratch")
    agent.train(
        total_time_steps=args.train_timesteps,
        callback=callback,
        logger=logger
    )

    agent.save(save_dir)

def load_train(save_dir, exp_name, load_dir, training_iter=1000000):  
    from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
    args = get_args()
    args.num_envs = 200
    args.headless = True
    env, env_cfg = make_env(args, custom_cfg(args), single_agent=False)
    
    args.config = "./openrl_ws/cfgs/ppo.yaml"
    args.train_timesteps = training_iter

    callback = CheckpointCallback(
        save_freq=5000,
        save_path=save_dir)
    
    net = PPONet(env, cfg=args, device=args.rl_device)
    agent = PPOAgent(net)
    logger = Logger(
        cfg=net.cfg,
        project_name="MQE-Curriculum",
        scenario_name=args.task,
        wandb_entity="kh-ryu-university-of-california-berkeley",
        exp_name=exp_name,
        log_path="./log",
        use_wandb=True,
        use_tensorboard=False,
    )

    print(f"Loading model from {load_dir}")
    agent.load(load_dir)
    agent.set_env(env)

    agent.train(
        total_time_steps=args.train_timesteps,
        callback=callback,
        logger=logger
    )

    agent.save(save_dir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train a Go1 Navigation agent")
    parser.add_argument("--run_date", type=str, default=None, help="Run date for curriculum learning")
    parser.add_argument("--curriculum_task", type=str, default=None, help="Curriculum task for curriculum learning")
    parser.add_argument("--sample_idx", type=int, default=None, help="Sample index for curriculum learning")
    parser.add_argument("--training_iter", type=int, default=None, help="Training iteration for curriculum learning")
    parser.add_argument("--load", type=bool, default=False, help="Load previous training checkpoint")
    parser.add_argument("--load_task", type=str, default=None, help="Load task for curriculum learning")
    parser.add_argument("--load_sample_idx", type=int, default=None, help="Load sample index for curriculum learning")

    args = parser.parse_args()
    run_date = args.run_date
    curriculum_task = args.curriculum_task
    sample_idx = args.sample_idx
    training_iter = args.training_iter
    load = args.load
    load_task = args.load_task
    load_sample_idx = args.load_sample_idx

    del args, parser
    sys.argv = [sys.argv[0]]

    save_dir = os.path.join("runs", run_date, curriculum_task, f"sample_{sample_idx}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    exp_name = f"{run_date}_{curriculum_task}_sample_{sample_idx}"

    if load:
        load_dir = os.path.join("runs", run_date, load_task, f"sample_{load_sample_idx}")
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Log directory {load_dir} does not exist. Please check the run date and curriculum task.")

    if not load:
        train(save_dir, exp_name, training_iter)
    else:
        load_train(save_dir, exp_name, load_dir, training_iter)
