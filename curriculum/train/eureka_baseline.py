import os
import pickle
import shutil
import subprocess
import sys
import traceback
from typing import Dict, List

from curriculum.gpt.eureka_api import EurekaAPI


class EurekaBaseline:
    def __init__(self, task, env_path, logger_path, run_datetime, cfg, seed=0):
        self.task = task
        self.env_path = env_path
        self.prompt_path = f"./curriculum/gpt/prompts/{task}"
        self.logger_path = logger_path
        self.experiment_time = run_datetime
        self.cfg = cfg
        self.seed = seed
        self.eureka_cfg = cfg.get("eureka", {})
        self.num_iterations = int(self.eureka_cfg.get("num_iterations", 3))
        self.num_candidates = int(self.eureka_cfg.get("num_candidates", 3))
        self.training_iter = int(self.eureka_cfg.get("training_iter", cfg["iter_per_task"]))
        self.intermediate_eval_seeds = self.eureka_cfg.get("intermediate_eval_seeds", [0])
        self.final_eval_seeds = self.eureka_cfg.get("final_eval_seeds", [0, 1, 2, 3, 4])
        self.objective_terms = self.eureka_cfg.get("objective_terms", [])
        self.tie_breaker_terms = self.eureka_cfg.get("tie_breaker_terms", [])
        self.gpt_api = EurekaAPI(self.prompt_path, logger_path, line_num=cfg["line_num"])
        self.best_candidate = None

    def train(self):
        all_candidates = []
        reflection = None
        best_reward_code = None

        for iteration in range(self.num_iterations):
            iteration_candidates = []
            for candidate_idx in range(self.num_candidates):
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
            self._save_iteration_summary(iteration, iteration_candidates, best_this_iteration, reflection)

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

        return candidate

    def _train_candidate(self, model_dir, exp_name):
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
            ],
            check=True,
        )

    def _run_intermediate_eval(self, candidates):
        eval_root = os.path.join(self.logger_path, "eureka", "intermediate_eval")
        os.makedirs(eval_root, exist_ok=True)
        results = []

        for candidate in candidates:
            if candidate.get("error"):
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
            candidate_eval_dir = os.path.join(
                eval_root,
                f"iteration_{candidate['iteration']}",
                f"candidate_{candidate['candidate_idx']}",
            )
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
                    ],
                    check=True,
                )
                result = {"seed": seed, "output_dir": seed_dir}
                candidate_results.append(result)
                results.append({
                    "iteration": candidate["iteration"],
                    "candidate_idx": candidate["candidate_idx"],
                    **result,
                })

            candidate["intermediate_eval"] = candidate_results
            self._save_pickle(os.path.join(candidate_eval_dir, "intermediate_eval.pkl"), candidate_results)

        self._save_pickle(os.path.join(eval_root, "intermediate_eval.pkl"), results)

    def _run_final_eval(self):
        if self.best_candidate is None:
            return

        self.gpt_api.update_env_code(
            self.env_path,
            self.best_candidate["reward_code"],
            save_path=os.path.join(self.logger_path, "eureka", "best_reward_code.py"),
        )

        eval_dir = os.path.join(self.logger_path, "eureka", "final_eval")
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
                ],
                check=True,
            )
            results.append({"seed": seed, "output_dir": seed_dir})
        self._save_pickle(os.path.join(eval_dir, "final_eval.pkl"), results)

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

    def _save_iteration_summary(self, iteration, candidates, best_candidate, reflection):
        iteration_dir = os.path.join(self.logger_path, "eureka", f"iteration_{iteration}")
        summary = {
            "iteration": iteration,
            "best_candidate_idx": best_candidate["candidate_idx"],
            "best_score": best_candidate["score"],
            "best_tie_score": best_candidate["tie_score"],
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
