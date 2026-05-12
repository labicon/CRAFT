import os
import pickle
import shutil
import subprocess
import sys
import traceback
from typing import Dict, List, Optional

from curriculum.gpt.eureka_api import EurekaAPI


class EurekaBaseline:
    def __init__(self, task, env_path, logger_path, run_datetime, cfg, seed=0, gpu_id=0):
        self.task = task
        self.env_path = env_path
        self.prompt_path = f"./curriculum/gpt/prompts/{task}"
        self.logger_path = logger_path
        self.experiment_time = run_datetime
        self.cfg = cfg
        self.seed = seed
        self.gpu_id = gpu_id
        self.eureka_cfg = cfg.get("eureka", {})
        self.num_iterations = int(self.eureka_cfg.get("num_iterations", 3))
        self.num_candidates = int(self.eureka_cfg.get("num_candidates", 3))
        self.training_iter = int(self.eureka_cfg.get("training_iter", cfg["iter_per_task"]))
        self.intermediate_eval_seeds = self.eureka_cfg.get("intermediate_eval_seeds", [0, 1, 2, 3, 4])
        self.final_eval_seeds = self.eureka_cfg.get("final_eval_seeds", [0, 1, 2, 3, 4])
        self.objective_terms = self.eureka_cfg.get("objective_terms", [])
        self.tie_breaker_terms = self.eureka_cfg.get("tie_breaker_terms", [])
        self.gpt_api = EurekaAPI(self.prompt_path, logger_path, line_num=cfg["line_num"])
        self.best_candidate = None

    def train(self):
        self._save_run_state()
        resume = self._load_resume_state()

        if resume["start_iteration"] > 0 or resume["start_candidate"] > 0:
            print(
                f"Resuming Eureka from iteration {resume['start_iteration']}, "
                f"candidate {resume['start_candidate']} "
                f"({len(resume['all_candidates'])} candidates already completed)"
            )

        all_candidates = resume["all_candidates"]
        reflection = resume["reflection"]
        best_reward_code = resume["best_reward_code"]
        self.best_candidate = resume["best_candidate"]

        for iteration in range(resume["start_iteration"], self.num_iterations):
            if iteration == resume["start_iteration"]:
                iteration_candidates = list(resume["iteration_candidates_so_far"])
                start_candidate = resume["start_candidate"]
            else:
                iteration_candidates = []
                start_candidate = 0

            for candidate_idx in range(start_candidate, self.num_candidates):
                candidate = self._run_candidate(
                    iteration=iteration,
                    candidate_idx=candidate_idx,
                    reflection=reflection,
                    best_reward_code=best_reward_code,
                )
                iteration_candidates.append(candidate)
                all_candidates.append(candidate)

            best_this_iteration = self._select_best(iteration_candidates)
            if best_this_iteration is None:
                raise RuntimeError(f"All Eureka candidates failed in iteration {iteration}.")

            if self.best_candidate is None or self._score_tuple(best_this_iteration) > self._score_tuple(self.best_candidate):
                self.best_candidate = best_this_iteration

            best_reward_code = best_this_iteration["reward_code"]
            reflection = self.gpt_api.build_reflection(best_this_iteration, iteration_candidates)
            self._save_iteration_summary(iteration, iteration_candidates, best_this_iteration, reflection, best_reward_code)

        self._save_best_reward()
        self._run_intermediate_eval(all_candidates)
        self._run_final_eval()
        self._save_summary(all_candidates)

    def _run_candidate(self, iteration, candidate_idx, reflection, best_reward_code):
        candidate_dir = os.path.join(
            self.logger_path,
            "eureka",
            f"iteration_{iteration}",
            f"candidate_{candidate_idx}",
        )
        model_dir = os.path.join(candidate_dir, "model")
        exp_name = f"eureka_{self.experiment_time}_{self.task}_iter_{iteration}_candidate_{candidate_idx}"
        candidate = {
            "iteration": iteration,
            "candidate_idx": candidate_idx,
            "candidate_dir": candidate_dir,
            "model_dir": model_dir,
            "exp_name": exp_name,
            "score": float("-inf"),
            "tie_score": float("-inf"),
        }
        os.makedirs(candidate_dir, exist_ok=True)

        try:
            generation = self.gpt_api.generate_candidate(
                task_name=self.task,
                iteration=iteration,
                candidate_idx=candidate_idx,
                reflection=reflection,
                best_reward_code=best_reward_code,
            )
            candidate.update(generation)
            self._save_candidate_generation(candidate_dir, generation)
            self.gpt_api.update_env_code(
                self.env_path,
                generation["reward_code"],
                save_path=os.path.join(candidate_dir, "reward_code.py"),
            )

            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            self._train_candidate(model_dir, exp_name)

            event_dir = os.path.join("training-log", "MQE-Eureka", self.task, exp_name, "tensorboard")
            scalars = self.gpt_api.load_tensorboard_scalars(event_dir)
            scalar_summary = self.gpt_api.summarize_scalars(scalars)
            candidate["event_dir"] = event_dir
            candidate["scalar_summary"] = scalar_summary
            candidate["score"] = self._compute_score(scalar_summary, self.objective_terms)
            candidate["tie_score"] = self._compute_score(scalar_summary, self.tie_breaker_terms)
            self._save_pickle(os.path.join(candidate_dir, "scalars.pkl"), scalars)
            self._save_pickle(os.path.join(candidate_dir, "scalar_summary.pkl"), scalar_summary)
        except Exception as exc:
            candidate["error"] = traceback.format_exc()
            with open(os.path.join(candidate_dir, "error.txt"), "w") as file:
                file.write(candidate["error"])
            print(f"Eureka candidate {iteration}/{candidate_idx} failed: {exc}")

        self._save_pickle(os.path.join(candidate_dir, "candidate.pkl"), candidate)
        return candidate

    def _train_candidate(self, model_dir, exp_name):
        subprocess_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        subprocess.run(
            [
                sys.executable,
                "-m",
                "openrl_ws.eureka_train",
                "--task",
                self.task,
                "--save_dir",
                model_dir,
                "--exp_name",
                exp_name,
                "--training_iter",
                str(self.training_iter),
                "--seed",
                str(self.seed),
                "--sim_device", f"cuda:{self.gpu_id}",
                "--rl_device", f"cuda:{self.gpu_id}",
                "--graphics_device_id", str(self.gpu_id),
            ],
            env=subprocess_env,
            check=True,
        )

    def _run_intermediate_eval(self, candidates):
        eval_root = os.path.join(self.logger_path, "eureka", "intermediate_eval")
        os.makedirs(eval_root, exist_ok=True)
        all_results = []

        # Collect already-completed results from prior runs.
        all_results_pkl = os.path.join(eval_root, "intermediate_eval.pkl")
        if os.path.exists(all_results_pkl):
            with open(all_results_pkl, "rb") as f:
                all_results = pickle.load(f)
            already_done = {
                (r["iteration"], r["candidate_idx"]) for r in all_results
            }
        else:
            already_done = set()

        for candidate in candidates:
            if candidate.get("error"):
                continue

            key = (candidate["iteration"], candidate["candidate_idx"])
            candidate_eval_dir = os.path.join(
                eval_root,
                f"iteration_{candidate['iteration']}",
                f"candidate_{candidate['candidate_idx']}",
            )
            candidate_pkl = os.path.join(candidate_eval_dir, "intermediate_eval.pkl")

            if key in already_done and os.path.exists(candidate_pkl):
                with open(candidate_pkl, "rb") as f:
                    candidate["intermediate_eval"] = pickle.load(f)
                print(
                    f"Skipping intermediate eval for iteration {candidate['iteration']} "
                    f"candidate {candidate['candidate_idx']} (already done)"
                )
                continue

            self.gpt_api.update_env_code(
                self.env_path,
                candidate["reward_code"],
                save_path=os.path.join(
                    candidate["candidate_dir"],
                    "intermediate_eval_reward_code.py",
                ),
            )

            candidate_results = []
            os.makedirs(candidate_eval_dir, exist_ok=True)

            for seed in self.intermediate_eval_seeds:
                seed_dir = os.path.join(candidate_eval_dir, f"seed_{seed}")
                os.makedirs(seed_dir, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "openrl_ws.eureka_eval",
                        "--task",
                        self.task,
                        "--load_dir",
                        candidate["model_dir"],
                        "--output_dir",
                        seed_dir,
                        "--seed",
                        str(seed),
                        "--sim_device", f"cuda:{self.gpu_id}",
                        "--graphics_device_id", str(self.gpu_id),
                    ],
                    env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
                    check=True,
                )
                result = {"seed": seed, "output_dir": seed_dir}
                candidate_results.append(result)
                all_results.append({
                    "iteration": candidate["iteration"],
                    "candidate_idx": candidate["candidate_idx"],
                    **result,
                })

            candidate["intermediate_eval"] = candidate_results
            self._save_pickle(candidate_pkl, candidate_results)
            self._save_pickle(all_results_pkl, all_results)

    def _run_final_eval(self):
        if self.best_candidate is None:
            return

        eval_dir = os.path.join(self.logger_path, "eureka", "final_eval")
        final_eval_pkl = os.path.join(eval_dir, "final_eval.pkl")
        if os.path.exists(final_eval_pkl):
            print("Final eval already completed, skipping.")
            return

        self.gpt_api.update_env_code(
            self.env_path,
            self.best_candidate["reward_code"],
            save_path=os.path.join(self.logger_path, "eureka", "best_reward_code.py"),
        )

        os.makedirs(eval_dir, exist_ok=True)
        results = []
        for seed in self.final_eval_seeds:
            seed_dir = os.path.join(eval_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "openrl_ws.eureka_eval",
                    "--task",
                    self.task,
                    "--load_dir",
                    self.best_candidate["model_dir"],
                    "--output_dir",
                    seed_dir,
                    "--seed",
                    str(seed),
                    "--sim_device", f"cuda:{self.gpu_id}",
                    "--graphics_device_id", str(self.gpu_id),
                ],
                env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
                check=True,
            )
            results.append({"seed": seed, "output_dir": seed_dir})
        self._save_pickle(final_eval_pkl, results)

    def _load_resume_state(self) -> Dict:
        """Scan the log directory for partial progress and return resume state."""
        empty = {
            "start_iteration": 0,
            "start_candidate": 0,
            "all_candidates": [],
            "reflection": None,
            "best_reward_code": None,
            "best_candidate": None,
            "iteration_candidates_so_far": [],
        }

        eureka_dir = os.path.join(self.logger_path, "eureka")
        if not os.path.isdir(eureka_dir):
            return empty

        all_candidates = []
        reflection = None
        best_reward_code = None
        best_candidate = None

        for iteration in range(self.num_iterations):
            iteration_dir = os.path.join(eureka_dir, f"iteration_{iteration}")
            if not os.path.isdir(iteration_dir):
                return {
                    **empty,
                    "start_iteration": iteration,
                    "all_candidates": all_candidates,
                    "reflection": reflection,
                    "best_reward_code": best_reward_code,
                    "best_candidate": best_candidate,
                }

            summary_path = os.path.join(iteration_dir, "iteration_summary.pkl")
            if os.path.exists(summary_path):
                with open(summary_path, "rb") as f:
                    summary = pickle.load(f)
                iter_candidates = summary["candidates"]
                all_candidates.extend(iter_candidates)
                reflection = summary["reflection"]
                best_reward_code = summary.get("best_reward_code")
                if best_reward_code is None:
                    # Fallback for summaries saved before best_reward_code was added.
                    best_idx = summary["best_candidate_idx"]
                    iter_best = next(
                        (c for c in iter_candidates if c["candidate_idx"] == best_idx),
                        None,
                    )
                    if iter_best:
                        best_reward_code = iter_best.get("reward_code")
                else:
                    iter_best = next(
                        (c for c in iter_candidates if c["candidate_idx"] == summary["best_candidate_idx"]),
                        self._select_best(iter_candidates),
                    )
                if iter_best and (
                    best_candidate is None
                    or self._score_tuple(iter_best) > self._score_tuple(best_candidate)
                ):
                    best_candidate = iter_best
            else:
                # Iteration in progress — scan individual candidates.
                iteration_candidates_so_far = []
                for candidate_idx in range(self.num_candidates):
                    candidate_dir = os.path.join(iteration_dir, f"candidate_{candidate_idx}")
                    candidate_pkl = os.path.join(candidate_dir, "candidate.pkl")
                    if os.path.exists(candidate_pkl):
                        with open(candidate_pkl, "rb") as f:
                            candidate = pickle.load(f)
                        iteration_candidates_so_far.append(candidate)
                        all_candidates.append(candidate)
                    else:
                        return {
                            "start_iteration": iteration,
                            "start_candidate": candidate_idx,
                            "all_candidates": all_candidates,
                            "reflection": reflection,
                            "best_reward_code": best_reward_code,
                            "best_candidate": best_candidate,
                            "iteration_candidates_so_far": iteration_candidates_so_far,
                        }
                # All candidates ran but iteration summary is missing — redo summary step.
                return {
                    "start_iteration": iteration,
                    "start_candidate": self.num_candidates,
                    "all_candidates": all_candidates,
                    "reflection": reflection,
                    "best_reward_code": best_reward_code,
                    "best_candidate": best_candidate,
                    "iteration_candidates_so_far": iteration_candidates_so_far,
                }

        # All iterations complete — resume into post-training phase.
        return {
            "start_iteration": self.num_iterations,
            "start_candidate": 0,
            "all_candidates": all_candidates,
            "reflection": reflection,
            "best_reward_code": best_reward_code,
            "best_candidate": best_candidate,
            "iteration_candidates_so_far": [],
        }

    def _save_run_state(self):
        run_state = {
            "experiment_time": self.experiment_time,
            "task": self.task,
            "seed": self.seed,
            "num_iterations": self.num_iterations,
            "num_candidates": self.num_candidates,
            "training_iter": self.training_iter,
        }
        state_path = os.path.join(self.logger_path, "eureka", "run_state.pkl")
        if not os.path.exists(state_path):
            self._save_pickle(state_path, run_state)

    def _select_best(self, candidates: List[Dict]):
        valid_candidates = [candidate for candidate in candidates if not candidate.get("error")]
        if not valid_candidates:
            return None
        return max(valid_candidates, key=self._score_tuple)

    @staticmethod
    def _score_tuple(candidate):
        return (candidate.get("score", float("-inf")), candidate.get("tie_score", float("-inf")))

    @staticmethod
    def _compute_score(scalar_summary, terms):
        if not terms:
            return 0.0
        score = 0.0
        matched = False
        for term in terms:
            tag = term["tag"]
            stats = scalar_summary.get(tag)
            if stats is None:
                continue
            matched = True
            statistic = term.get("stat", "last")
            value = stats.get(statistic, stats["last"])
            weight = float(term.get("weight", 1.0))
            mode = term.get("mode", "max")
            if mode == "min":
                value = -value
            score += weight * value
        return score if matched else float("-inf")

    def _save_best_reward(self):
        best_dir = os.path.join(self.logger_path, "eureka")
        os.makedirs(best_dir, exist_ok=True)
        with open(os.path.join(best_dir, "best_reward_code.txt"), "w") as file:
            file.write(self.best_candidate["reward_code"])

    def _save_iteration_summary(self, iteration, candidates, best_candidate, reflection, best_reward_code):
        iteration_dir = os.path.join(self.logger_path, "eureka", f"iteration_{iteration}")
        summary = {
            "iteration": iteration,
            "best_candidate_idx": best_candidate["candidate_idx"],
            "best_score": best_candidate["score"],
            "best_tie_score": best_candidate["tie_score"],
            "best_reward_code": best_reward_code,
            "reflection": reflection,
            "candidates": candidates,
        }
        self._save_pickle(os.path.join(iteration_dir, "iteration_summary.pkl"), summary)
        with open(os.path.join(iteration_dir, "reflection.txt"), "w") as file:
            file.write(reflection)

    def _save_summary(self, all_candidates):
        summary = {
            "task": self.task,
            "num_iterations": self.num_iterations,
            "num_candidates": self.num_candidates,
            "training_iter": self.training_iter,
            "intermediate_eval_seeds": self.intermediate_eval_seeds,
            "objective_terms": self.objective_terms,
            "tie_breaker_terms": self.tie_breaker_terms,
            "best_candidate": self.best_candidate,
            "all_candidates": all_candidates,
        }
        self._save_pickle(os.path.join(self.logger_path, "eureka", "eureka_summary.pkl"), summary)

    @staticmethod
    def _save_candidate_generation(candidate_dir, generation):
        with open(os.path.join(candidate_dir, "reward_system_prompt.md"), "w") as file:
            file.write(generation["system_prompt"])
        with open(os.path.join(candidate_dir, "reward_user_prompt.md"), "w") as file:
            file.write(generation["user_prompt"])
        with open(os.path.join(candidate_dir, "reward_answer.md"), "w") as file:
            file.write(generation["answer"])

    @staticmethod
    def _save_pickle(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(data, file)
