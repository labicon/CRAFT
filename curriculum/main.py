import argparse
import os

# Parse --gpu early so CUDA_VISIBLE_DEVICES is set before any CUDA initialization.
# With this set, physical GPU X appears as cuda:0 in all processes.
# Subprocesses inherit this env var automatically.
# Note: IsaacGym's renderer uses OpenGL/Vulkan and ignores CUDA_VISIBLE_DEVICES,
# so --graphics_device_id must always be the physical GPU ID (passed via subprocess args).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=0)
_pre_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

import yaml
from datetime import datetime

from curriculum.train.manual_curriculum import Manual_Module
from curriculum.train.curriculum_train import Curriculum_Module
from curriculum.train.eureka_baseline import EurekaBaseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recieve task name, experiment name, and seed number")
    parser.add_argument("--seed", type=int, help="Random seed", default=0)
    parser.add_argument("--logdir", type=str, help="Log directory", default="logs")
    parser.add_argument("--module", type=str, default="curriculum", choices=["manual", "curriculum", "eureka"], help="Select module: manual, curriculum, or eureka")
    parser.add_argument("--task", type=str, default="go2gate", choices=["go2gate", "go2seesaw", "go2pushbox"], help="Select task: go2gate or go2seesaw")
    parser.add_argument("--debug", action="store_true", help="Use a lightweight debug configuration when supported by the selected module")
    parser.add_argument("--resume", type=str, default=None, help="Path to a previous Eureka log directory to resume from (eureka module only)")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU device ID (e.g. 0, 1, 2)")
    args = parser.parse_args()

    seed = args.seed

    if args.resume:
        logger_path = args.resume.rstrip("/") + "/"
        if not os.path.isdir(logger_path):
            raise ValueError(f"Resume path does not exist: {logger_path}")
    else:
        current_datetime = datetime.now().strftime("%m-%d_%H-%M")
        logger_path = f"{args.logdir}/{current_datetime}/"
        os.makedirs(logger_path, exist_ok=True)

    current_datetime = datetime.now().strftime("%m-%d_%H-%M")

    with open(f"./curriculum/configs/{args.task}.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.debug and args.module == "eureka":
        debug_cfg = cfg.get("eureka_debug")
        if debug_cfg is None:
            raise ValueError(f"No eureka_debug configuration found for task {args.task}.")
        cfg["eureka"] = {**cfg.get("eureka", {}), **debug_cfg}

    env_path = cfg['env_path']

    if args.module == "manual":
        module = Manual_Module(args.task, env_path, logger_path, current_datetime, cfg, seed, args.gpu)
    elif args.module == "curriculum":
        module = Curriculum_Module(args.task, env_path, logger_path, current_datetime, cfg, seed, args.gpu)
    else:
        module = EurekaBaseline(args.task, env_path, logger_path, current_datetime, cfg, seed, args.gpu)

    module.train()
