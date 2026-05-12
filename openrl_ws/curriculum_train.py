from openrl_ws.utils import make_env, get_args, reset_value_network, reset_policy_std
from mqe.envs.utils import custom_cfg
from openrl.utils.logger import Logger
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import os
import shutil
import argparse
import sys

def train(task, save_dir, exp_name, training_iter=1000000, sim_device="cuda:0", rl_device="cuda:0", graphics_device_id=0):
    from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
    args = get_args()
    args.train_timesteps = training_iter
    args.task = task
    args.num_envs = 500
    args.headless = True
    args.separate_policy = True
    args.sim_device = sim_device
    args.sim_device_id = int(sim_device.split(":")[-1]) if ":" in sim_device else 0
    args.rl_device = rl_device
    args.graphics_device_id = graphics_device_id

    env, env_cfg = make_env(args, custom_cfg(args), single_agent=False)
    
    # args.config = "./openrl_ws/cfgs/ppo.yaml" # Somehow this does not work
    args.lr = 1e-3
    args.critic_lr = 1e-3
    args.log_interval = 5
    # args.use_recurrent_policy = True
    args.use_joint_action_loss = False
    args.use_valuenorm = True
    args.use_adv_normalize = True
    args.entropy_coef = 0.001
    args.num_mini_batch = 4

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
        log_path="./training-log",
        use_wandb=True,
        use_tensorboard=True,
    )

    print("Start training from scratch")
    agent.train(
        total_time_steps=args.train_timesteps,
        callback=callback,
        logger=logger
    )

    agent.save(save_dir)

def load_train(task, save_dir, exp_name, load_dir, training_iter=1000000, sim_device="cuda:0", rl_device="cuda:0", graphics_device_id=0):
    from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
    args = get_args()
    args.task = task
    args.num_envs = 500
    args.headless = True
    args.train_timesteps = training_iter
    args.separate_policy = True
    args.sim_device = sim_device
    args.sim_device_id = int(sim_device.split(":")[-1]) if ":" in sim_device else 0
    args.rl_device = rl_device
    args.graphics_device_id = graphics_device_id

    env, env_cfg = make_env(args, custom_cfg(args), single_agent=False)

    # args.config = "./openrl_ws/cfgs/ppo.yaml" # Somehow this does not work
    args.lr = 1e-3
    args.critic_lr = 1e-3
    args.log_interval = 5
    # args.use_recurrent_policy = True
    args.use_joint_action_loss = False
    args.use_valuenorm = True
    args.use_adv_normalize = True
    args.entropy_coef = 0.001
    args.num_mini_batch = 4

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
        log_path="./training-log",
        use_wandb=True,
        use_tensorboard=True,
    )

    print(f"Loading model from {load_dir}")
    agent.load(load_dir)
    agent.set_env(env)

    agent = reset_value_network(args, agent)
    agent = reset_policy_std(agent)

    agent.train(
        total_time_steps=args.train_timesteps,
        callback=callback,
        logger=logger
    )

    agent.save(save_dir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train a Go1 Navigation agent")
    parser.add_argument("--task", type=str, required=True, help="Environment task name (e.g. go2gate, go2pushbox)")
    parser.add_argument("--run_date", type=str, default=None, help="Run date for curriculum learning")
    parser.add_argument("--curriculum_task", type=str, default=None, help="Curriculum task for curriculum learning")
    parser.add_argument("--sample_idx", type=int, default=None, help="Sample index for curriculum learning")
    parser.add_argument("--training_iter", type=int, default=None, help="Training iteration for curriculum learning")
    parser.add_argument("--load", type=bool, default=False, help="Load previous training checkpoint")
    parser.add_argument("--load_task", type=str, default=None, help="Load task for curriculum learning")
    parser.add_argument("--load_sample_idx", type=int, default=None, help="Load sample index for curriculum learning")
    parser.add_argument("--sim_device", type=str, default="cuda:0", help="Physics simulation device (e.g. cuda:0, cuda:1)")
    parser.add_argument("--rl_device", type=str, default="cuda:0", help="RL algorithm device (e.g. cuda:0, cuda:1)")
    parser.add_argument("--graphics_device_id", type=int, default=0, help="GPU index for rendering (e.g. 0, 1)")

    args = parser.parse_args()
    task = args.task
    run_date = args.run_date
    curriculum_task = args.curriculum_task
    sample_idx = args.sample_idx
    training_iter = args.training_iter
    load = args.load
    load_task = args.load_task
    load_sample_idx = args.load_sample_idx
    sim_device = args.sim_device
    rl_device = args.rl_device
    graphics_device_id = args.graphics_device_id

    del args, parser
    sys.argv = [sys.argv[0]]

    save_dir = os.path.join("logs", run_date, curriculum_task, f"sample_{sample_idx}", "model")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    exp_name = f"{curriculum_task}_sample_{sample_idx}"

    if load:
        load_dir = os.path.join("logs", run_date, load_task, f"sample_{load_sample_idx}", "model")
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Log directory {load_dir} does not exist. Please check the run date and curriculum task.")

    if not load:
        train(task, save_dir, exp_name, training_iter, sim_device, rl_device, graphics_device_id)
    else:
        load_train(task, save_dir, exp_name, load_dir, training_iter, sim_device, rl_device, graphics_device_id)
