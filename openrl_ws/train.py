
from openrl_ws.utils import make_env, get_args #, MATWrapper
from mqe.envs.utils import custom_cfg
from openrl.utils.logger import Logger
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

from datetime import datetime

# import argparse

def train(args):
    from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
    # cfg_parser = create_config_parser()
    # cfg = cfg_parser.parse_args()

    start_time = datetime.now()
    start_time_str = start_time.strftime("%m-%d_%H-%M")

    if args.algo == "sppo" or args.algo == "dppo":
        single_agent = True
    else:
        single_agent = False
    
    env, env_cfg = make_env(args, custom_cfg(args), single_agent)
    
    # args.config = "./openrl_ws/cfgs/ppo.yaml"
    args.lr = 7e-4
    args.critic_lr = 7e-4
    args.log_interval = 5
    # args.use_recurrent_policy = True
    args.use_joint_action_loss = False
    args.use_valuenorm = True
    args.use_adv_normalize = True
    args.entropy_coef = 0.0001
    
    dir_name = "./checkpoints/" + args.task + start_time_str
    callback = CheckpointCallback(
        save_freq=5000,
        save_path=dir_name)
    
    if "po" in args.algo:
        net = PPONet(env, cfg=args, device=args.rl_device)
        agent = PPOAgent(net)
        logger = Logger(
            cfg=net.cfg,
            project_name="MQE",
            scenario_name=args.task,
            wandb_entity="kh-ryu-university-of-california-berkeley",
            exp_name=args.exp_name,
            log_path="./log",
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
        )
        agent.train(
            total_time_steps=args.train_timesteps,
            callback=callback,
            logger=logger
        )
    else:
        agent.train(total_time_steps=args.train_timesteps)
    # dir_name = "./checkpoints/" + args.task + "/" + start_time_str
    
    agent.save(dir_name)

if __name__ == '__main__':
    args = get_args()
    train(args)
