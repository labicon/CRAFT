"""
Eureka-style reward search for go2gate (direct task, no curriculum).
Thin wrapper around EurekaBaseline from curriculum/train/eureka_baseline.py.

Run:
  python -m analysis.reward_refinement.eureka_direct [--seed N] [--logdir DIR]
                                                      [--debug] [--resume PATH]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from curriculum.train.eureka_baseline import EurekaBaseline


def main():
    parser = argparse.ArgumentParser(description="Eureka baseline for go2gate direct task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Short training run for smoke-testing")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing log dir to resume")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "configs" / "go2gate.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        debug_cfg = cfg.get("eureka_debug", {})
        cfg["eureka"] = {**cfg.get("eureka", {}), **debug_cfg}

    if args.resume:
        logger_path = args.resume.rstrip("/") + "/"
        if not os.path.isdir(logger_path):
            raise ValueError(f"Resume path does not exist: {logger_path}")
    else:
        run_datetime = datetime.now().strftime("%m-%d_%H-%M")
        base_logdir = args.logdir or str(Path(__file__).parent / "logs")
        logger_path = os.path.join(base_logdir, f"eureka_{run_datetime}") + "/"
        os.makedirs(logger_path, exist_ok=True)

    run_datetime = datetime.now().strftime("%m-%d_%H-%M")
    print(f"Log dir: {logger_path}")

    baseline = EurekaBaseline(
        task=cfg["Task"],
        env_path=cfg["env_path"],
        logger_path=logger_path,
        run_datetime=run_datetime,
        cfg=cfg,
        seed=args.seed,
    )
    baseline.train()


if __name__ == "__main__":
    main()
