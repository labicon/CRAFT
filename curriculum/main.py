import argparse
import yaml
from datetime import datetime
import os

from train.manual_curriculum import Manual_Module
from train.curriculum_train_multiple_eval import Curriculum_Module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recieve task name, experiment name, and seed number")
    parser.add_argument("--seed", type=int, help="Random seed", default=0)
    parser.add_argument("--logdir", type=str, help="Log directory", default="logs")
    args = parser.parse_args()

    seed = args.seed

    current_datetime = datetime.now().strftime("%m-%d_%H-%M")
    logger_path = f"{args.logdir}/{current_datetime}/"
    os.makedirs(logger_path, exist_ok=True)

    with open("./curriculum/configs/go2seesaw_config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    env_path = cfg['env_path']

    # module = Manual_Module(env_path, logger_path, current_datetime, cfg, seed)
    module = Curriculum_Module(env_path, logger_path, current_datetime, cfg, seed)
    module.train()
    # module.resume_train(resume_idx=2, resume_sample_idx=0, resume_from_training=True)