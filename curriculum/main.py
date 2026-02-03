import argparse
import yaml
from datetime import datetime
import os

from train.manual_curriculum import Manual_Module
from curriculum.train.curriculum_train import Curriculum_Module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recieve task name, experiment name, and seed number")
    parser.add_argument("--seed", type=int, help="Random seed", default=0)
    parser.add_argument("--logdir", type=str, help="Log directory", default="logs")
    parser.add_argument("--module", type=str, default="curriculum", choices=["manual", "curriculum"], help="Select module: manual or curriculum")
    parser.add_argument("--task", type=str, default="go2gate", choices=["go2gate", "go2seesaw"], help="Select task: go2gate or go2seesaw")
    args = parser.parse_args()

    seed = args.seed

    current_datetime = datetime.now().strftime("%m-%d_%H-%M")
    logger_path = f"{args.logdir}/{current_datetime}/"
    os.makedirs(logger_path, exist_ok=True)

    with open(f"./curriculum/configs/{args.task}.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    env_path = cfg['env_path']

    if args.module == "manual":
        module = Manual_Module(args.task, env_path, logger_path, current_datetime, cfg, seed)
    else:
        module = Curriculum_Module(args.task, env_path, logger_path, current_datetime, cfg, seed)
        
    module.train()