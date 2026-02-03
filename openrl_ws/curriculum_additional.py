from openrl_ws.utils import make_env, get_args, reset_value_network, reset_policy_std
from mqe.envs.utils import custom_cfg
from openrl.utils.logger import Logger
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import os
import shutil
import argparse
import sys

def load_train(save_dir, exp_name, load_dir, training_iter=1000000):  
    from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
    args = get_args()
    args.task = "go2gate"
    args.num_envs = 500
    args.headless = True
    args.train_timesteps = training_iter

    env, env_cfg = make_env(args, custom_cfg(args), single_agent=False)

    # args.config = "./openrl_ws/cfgs/ppo.yaml" # Somehow this does not work
    args.lr = 7e-4
    args.critic_lr = 7e-4
    args.log_interval = 5
    # args.use_recurrent_policy = True
    args.use_joint_action_loss = False
    args.use_valuenorm = True
    args.use_adv_normalize = True
    args.entropy_coef = 0.0001

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

    # agent = reset_value_network(args, agent)
    # agent = reset_policy_std(agent)

    agent.train(
        total_time_steps=args.train_timesteps,
        callback=callback,
        logger=logger
    )

    agent.save(save_dir)

if __name__ == '__main__':
    
    load_dir = "/home/kanghyun/mqe-curriculum/logs/go2gate/08-16_17-09_no_refine_3/3_Final task(08-16_17-09)/sample_0/model"
    save_dir = "/home/kanghyun/mqe-curriculum/logs/go2gate/08-16_17-09_no_refine_3/4_Additional/sample_0/model"

    os.makedirs(save_dir)
    # exp_name = f"{run_date}_{curriculum_task}_sample_{sample_idx}"
    exp_name = f"08-16_17-09_no_refine_3_4_Additional_sample_0"

    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Log directory {load_dir} does not exist. Please check the run date and curriculum task.")

    load_train(save_dir, exp_name, load_dir, 20000000)
