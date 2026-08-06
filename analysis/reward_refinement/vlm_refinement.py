"""
VLM-guided reward refinement for go2gate (direct task, no curriculum).

Loop per sample:
  generate/refine reward → inject → train (eureka_train) → eval (eureka_eval)
  → VLM evaluates snapshots → VLM gives advice → next sample

Run:
  python -m analysis.reward_refinement.vlm_refinement [--seed N] [--logdir DIR]
                                                       [--debug] [--resume PATH]
"""

import argparse
import base64
import glob
import os
import pickle
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.reward_refinement.api import VLMRefinementAPI

N_ROLLOUTS = 5
MAX_GENERATION_ATTEMPTS = 5

TASK_DESCRIPTION = (
    "Two quadruped robots (Agent 0 and Agent 1) must navigate through a narrow gate "
    "without colliding with each other. Both agents must pass through the gate to succeed. "
    "The agents need to coordinate their movement to avoid blocking each other while "
    "both making progress toward and through the gate."
)
TASK_REASON = (
    "This is the direct target task: gate-passing coordination without any curriculum progression."
)


def make_curriculum(run_datetime):
    return [{
        "Task_num": "1",
        "Subtask": "Gate_Passing",
        "Description": TASK_DESCRIPTION,
        "Reason": TASK_REASON,
        "Name": f"1_Gate_Passing({run_datetime})",
    }]


class VLMRefinementExperiment:
    def __init__(self, cfg, log_dir, run_datetime, seed=0,
                 sim_device="cuda:0", rl_device="cuda:0", graphics_device_id=0):
        self.task = cfg["Task"]
        self.env_path = cfg["env_path"]
        self.num_reward_samples = cfg.get("num_reward_samples", 3)
        self.iter_per_task = cfg["iter_per_task"]
        self.final_eval_seeds = cfg.get("final_eval_seeds", [0, 1, 2, 3, 4])
        self.seed = seed
        self.sim_device = sim_device
        self.rl_device = rl_device
        self.graphics_device_id = graphics_device_id
        self.log_dir = log_dir
        self.run_datetime = run_datetime
        self.curriculum = make_curriculum(run_datetime)

        prompt_path = str(REPO_ROOT / "curriculum" / "gpt" / "prompts" / self.task)
        self.api = VLMRefinementAPI(
            prompt_path=prompt_path,
            log_path=log_dir + "/",
            line_num=cfg["line_num"],
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self):
        task = self.curriculum[0]
        reward_samples, failure_reasons, advices = self._load_resume_state()
        start_sample = len(reward_samples)
        best_model_dir = self._find_best_model(reward_samples, failure_reasons)

        if start_sample > 0:
            print(f"Resuming from sample {start_sample} ({len(reward_samples)} already done)")

        # When extending: the final sample of the previous run has no outgoing advice yet.
        if (start_sample > 0
                and len(advices) < start_sample
                and start_sample < self.num_reward_samples
                and failure_reasons
                and failure_reasons[-1] != "Success"):
            last_idx = start_sample - 1
            print(f"Generating missing advice for sample {last_idx} before extending...")
            advice = self._get_advice(last_idx, reward_samples[last_idx], failure_reasons[last_idx])
            advices.append(advice)
            self._save_state(reward_samples, failure_reasons, advices, best_model_dir)

        for sample_num in range(start_sample, self.num_reward_samples):
            print(f"\n=== Sample {sample_num} ===")

            # Generate or refine reward code
            reward_code = self._get_reward_code(sample_num, reward_samples, failure_reasons, advices)
            if reward_code is None:
                print(f"Sample {sample_num}: failed to generate reward code. Skipping.")
                reward_samples.append("")
                failure_reasons.append("Reward code generation failed.")
                self._save_state(reward_samples, failure_reasons, advices, best_model_dir)
                continue

            # Inject reward code into environment wrapper
            self.api.update_env_code(
                self.env_path, self.curriculum, 0, reward_code, version_number=sample_num
            )

            # Train
            model_dir = os.path.join(self.log_dir, "models", f"sample_{sample_num}")
            exp_name = f"vlm_{self.run_datetime}_{self.task}_sample_{sample_num}"
            try:
                self._train(model_dir, exp_name)
            except subprocess.CalledProcessError as e:
                print(f"Sample {sample_num}: training failed — {e}")
                reward_samples.append(reward_code)
                failure_reasons.append("Training subprocess failed.")
                self._save_state(reward_samples, failure_reasons, advices, best_model_dir)
                if sample_num < self.num_reward_samples - 1:
                    advices.append("The previous reward function caused a training error. "
                                   "Please fix any syntax or runtime issues.")
                continue

            # Evaluate with VLM
            success, last_reason = self._evaluate(sample_num, model_dir)

            reward_samples.append(reward_code)
            if success:
                print(f"Sample {sample_num}: SUCCESS")
                best_model_dir = model_dir
                self._save_state(reward_samples, failure_reasons, advices, best_model_dir)
                break

            print(f"Sample {sample_num}: FAILED — {last_reason}")
            failure_reasons.append(last_reason)

            # Get VLM advice for next sample
            if sample_num < self.num_reward_samples - 1:
                event_dir = os.path.join(
                    "training-log", "MQE-Eureka", self.task, exp_name, "tensorboard"
                )
                self.api.set_event_dir(str(REPO_ROOT / event_dir))
                advice = self._get_advice(sample_num, reward_code, last_reason)
                advices.append(advice)

            self._save_state(reward_samples, failure_reasons, advices, best_model_dir)

        # Final evaluation
        if best_model_dir:
            self._run_final_eval(best_model_dir)
        else:
            print("No successful model found — skipping final eval.")

        self._save_state(reward_samples, failure_reasons, advices, best_model_dir)
        print(f"\nDone. Logs: {self.log_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reward_code(self, sample_num, reward_samples, failure_reasons, advices):
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            try:
                if sample_num == 0:
                    code = self.api.generate_rewards(
                        self.curriculum, 0, sample_num,
                        reward_code_history={},
                        reward_samples=[],
                        failure_reasons=[],
                    )
                else:
                    code = self.api.refine_reward(
                        self.curriculum, 0, sample_num,
                        failed_reward=reward_samples[-1],
                        advice=advices[-1],
                    )
                if code is not None:
                    return code
            except Exception:
                traceback.print_exc()
        return None

    def _train(self, model_dir, exp_name):
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        subprocess.run(
            [
                sys.executable, "-m", "openrl_ws.eureka_train",
                "--task", self.task,
                "--save_dir", model_dir,
                "--exp_name", exp_name,
                "--training_iter", str(self.iter_per_task),
                "--seed", str(self.seed),
                "--sim_device", self.sim_device,
                "--rl_device", self.rl_device,
                "--graphics_device_id", str(self.graphics_device_id),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

    def _evaluate(self, sample_num, model_dir):
        success = False
        last_reason = "No evaluation completed."

        for rollout in range(N_ROLLOUTS):
            eval_dir = os.path.join(
                self.log_dir, "eval", f"sample_{sample_num}", f"rollout_{rollout}"
            )
            try:
                self._run_eval(model_dir, eval_dir, seed=rollout)
                traj, snapshots = self._load_eval_data(eval_dir)
                decision = self.api.evaluate_result(
                    self.curriculum, 0, sample_num, snapshots, traj
                )
                if decision and decision.get("decision"):
                    success = True
                    last_reason = "Success"
                    break
                last_reason = decision.get("reason", "Unknown failure") if decision else "VLM parse error"
            except Exception as e:
                traceback.print_exc()
                last_reason = str(e)

        return success, last_reason

    def _run_eval(self, model_dir, output_dir, seed):
        os.makedirs(output_dir, exist_ok=True)
        subprocess.run(
            [
                sys.executable, "-m", "openrl_ws.eureka_eval",
                "--task", self.task,
                "--load_dir", model_dir,
                "--output_dir", output_dir,
                "--seed", str(seed),
                "--sim_device", self.sim_device,
                "--graphics_device_id", str(self.graphics_device_id),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

    def _load_eval_data(self, eval_dir):
        with open(os.path.join(eval_dir, "traj_dict.pkl"), "rb") as f:
            traj = pickle.load(f)

        snapshot_files = sorted(
            glob.glob(os.path.join(eval_dir, "images", "snapshot_*.png")),
            key=lambda x: int(x.rsplit("_", 1)[-1].split(".")[0]),
        )
        snapshots = []
        for sf in snapshot_files:
            with open(sf, "rb") as f:
                snapshots.append(base64.b64encode(f.read()).decode("utf-8"))
        return traj, snapshots

    def _get_advice(self, sample_num, reward_code, failure_reason):
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            try:
                advice = self.api.get_advice(
                    self.curriculum, 0, sample_num,
                    failed_reward=reward_code,
                    failure_reason=failure_reason,
                )
                if advice:
                    return advice
            except Exception:
                traceback.print_exc()
        return "Unable to obtain advice. Please revise the reward function to better align with the task."

    def _run_final_eval(self, model_dir):
        print("\nRunning final evaluation...")
        for seed in self.final_eval_seeds:
            seed_dir = os.path.join(self.log_dir, "final_eval", f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            try:
                self._run_eval(model_dir, seed_dir, seed=seed)
                print(f"  seed {seed}: done")
            except Exception as e:
                print(f"  seed {seed}: failed — {e}")

    # ------------------------------------------------------------------
    # Resume / persistence helpers
    # ------------------------------------------------------------------

    def _state_path(self):
        return os.path.join(self.log_dir, "state.pkl")

    def _save_state(self, reward_samples, failure_reasons, advices, best_model_dir):
        state = {
            "reward_samples": reward_samples,
            "failure_reasons": failure_reasons,
            "advices": advices,
            "best_model_dir": best_model_dir,
            "run_datetime": self.run_datetime,
        }
        with open(self._state_path(), "wb") as f:
            pickle.dump(state, f)

    def _load_resume_state(self):
        path = self._state_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                state = pickle.load(f)
            return (
                state.get("reward_samples", []),
                state.get("failure_reasons", []),
                state.get("advices", []),
            )
        return [], [], []

    def _find_best_model(self, reward_samples, failure_reasons):
        for i in range(len(reward_samples) - 1, -1, -1):
            model_dir = os.path.join(self.log_dir, "models", f"sample_{i}")
            if os.path.isdir(model_dir) and i < len(failure_reasons) and failure_reasons[i] == "Success" or (
                i < len(failure_reasons) and failure_reasons[i] != "Training subprocess failed."
                and os.path.isdir(os.path.join(self.log_dir, "final_eval"))
            ):
                return model_dir
        return None


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VLM-guided reward refinement for go2gate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Short training run for smoke-testing")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing log dir to resume")
    parser.add_argument("--num_reward_samples", type=int, default=None,
                        help="Override num_reward_samples from config (use with --resume to extend)")
    parser.add_argument("--sim_device", type=str, default="cuda:0",
                        help="Physics simulation device (e.g. cuda:0, cuda:1)")
    parser.add_argument("--rl_device", type=str, default="cuda:0",
                        help="RL algorithm device (e.g. cuda:0, cuda:1)")
    parser.add_argument("--graphics_device_id", type=int, default=0,
                        help="GPU index for rendering (e.g. 0, 1)")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "configs" / "go2gate.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["iter_per_task"] = 100000
        cfg["num_reward_samples"] = 2
        cfg["final_eval_seeds"] = [0]

    if args.num_reward_samples is not None:
        cfg["num_reward_samples"] = args.num_reward_samples

    if args.resume:
        log_dir = args.resume.rstrip("/")
        if not os.path.isdir(log_dir):
            raise ValueError(f"Resume path does not exist: {log_dir}")
        run_datetime = os.path.basename(log_dir).replace("vlm_", "")
    else:
        run_datetime = datetime.now().strftime("%m-%d_%H-%M")
        base_logdir = args.logdir or str(Path(__file__).parent / "logs")
        log_dir = os.path.join(base_logdir, f"vlm_{run_datetime}")
        os.makedirs(log_dir, exist_ok=True)

    print(f"Log dir: {log_dir}")
    experiment = VLMRefinementExperiment(
        cfg, log_dir, run_datetime,
        seed=args.seed,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=args.graphics_device_id,
    )
    experiment.train()


if __name__ == "__main__":
    main()
