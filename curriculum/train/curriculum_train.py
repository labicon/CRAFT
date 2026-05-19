import os
import traceback
import pickle
import shutil
import subprocess
import glob
import base64
import sys

from curriculum.gpt.curriculum_api import CurriculumAPI
from curriculum.gpt.utils import *

MAX_ATTEMPT = 10

class Curriculum_Module:
    def __init__(self, task, env_path, logger_path, run_datetime, cfg, seed=0, gpu_id=0):
        self.task = task
        self.env_path = env_path
        self.prompt_path = f"./curriculum/gpt/prompts/{task}"
        self.gpt_api = CurriculumAPI(self.prompt_path, logger_path,
                                     line_num=cfg['line_num'])
        self.logger_path = logger_path
        self.best_reward_code = {}
        self.best_model_idx_list = []
        self.current_reward_code_list = []
        self.best_sim_reward = None
        self.cfg = cfg
        self.seed = seed
        self.gpu_id = gpu_id
        self.terminate_training = False
        self.experiment_time = run_datetime

    def generate_curriculum(self):
        # Generate curriculum and return list of dictionaries with task details
        self.curriculum = self.gpt_api.generate_curriculum()
        self.curriculum_length = len(self.curriculum)

        # Save curriculum pkl file
        file_path = os.path.join(self.logger_path, 'curriculum.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(self.curriculum, file)
            print(f"Curriculum saved to {file_path}")

    def _save_run_state(self):
        state = {"best_model_idx_list": self.best_model_idx_list}
        with open(os.path.join(self.logger_path, "run_state.pkl"), "wb") as f:
            pickle.dump(state, f)

    def _load_resume_state(self):
        curriculum_pkl = os.path.join(self.logger_path, "curriculum.pkl")
        if not os.path.exists(curriculum_pkl):
            self.generate_curriculum()
            return {"start_curriculum_idx": 0, "best_reward_code": {}, "best_model_idx_list": []}

        with open(curriculum_pkl, "rb") as f:
            self.curriculum = pickle.load(f)
        self.curriculum_length = len(self.curriculum)
        print(f"Loaded curriculum from {curriculum_pkl}")

        run_state_pkl = os.path.join(self.logger_path, "run_state.pkl")
        best_model_idx_list = []
        if os.path.exists(run_state_pkl):
            with open(run_state_pkl, "rb") as f:
                run_state = pickle.load(f)
            best_model_idx_list = run_state.get("best_model_idx_list", [])

        best_reward_code = {}
        for idx, task in enumerate(self.curriculum):
            best_code_path = os.path.join(self.logger_path, task["Name"], "best_reward_code.txt")
            if os.path.exists(best_code_path):
                with open(best_code_path, "r") as f:
                    best_reward_code[task["Name"]] = f.read()
            else:
                return {
                    "start_curriculum_idx": idx,
                    "best_reward_code": best_reward_code,
                    "best_model_idx_list": best_model_idx_list,
                }

        return {
            "start_curriculum_idx": self.curriculum_length,
            "best_reward_code": best_reward_code,
            "best_model_idx_list": best_model_idx_list,
        }

    def _is_sample_trained(self, task, sample_num):
        model_dir = os.path.join(self.logger_path, task["Name"], f"sample_{sample_num}", "model")
        return os.path.exists(os.path.join(model_dir, "traj_dict.pkl"))

    def _load_existing_sample_data(self, task, sample_num):
        sample_dir = os.path.join(self.logger_path, task["Name"], f"sample_{sample_num}")
        with open(os.path.join(sample_dir, "reward_code.md"), "r") as f:
            reward_code = f.read()
        model_dir = os.path.join(sample_dir, "model")
        with open(os.path.join(model_dir, "traj_dict.pkl"), "rb") as f:
            traj = pickle.load(f)
        with open(os.path.join(model_dir, "rew_dict.pkl"), "rb") as f:
            rew = pickle.load(f)
        snapshot_files = sorted(
            glob.glob(os.path.join(model_dir, "images", "snapshot_*.png")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        snapshots = []
        for sf in snapshot_files:
            with open(sf, "rb") as img:
                snapshots.append(base64.b64encode(img.read()).decode("utf-8"))
        return reward_code, rew, traj, snapshots

    def train(self):
        resume = self._load_resume_state()
        start_curriculum_idx = resume["start_curriculum_idx"]
        self.best_reward_code = resume["best_reward_code"]
        self.best_model_idx_list = resume["best_model_idx_list"]

        if start_curriculum_idx > 0:
            print(f"Resuming from task index {start_curriculum_idx} "
                  f"({start_curriculum_idx} tasks already completed)")

        curriculum_idx = start_curriculum_idx
        while curriculum_idx < self.curriculum_length:
            if self.terminate_training:
                break
            traj_rollout = []
            rew_rollout = []
            snapshot_rollout = []
            failure_reasons = []
            advices = []
            for sample_num in range(self.cfg["num_reward_samples"]):
                task = self.curriculum[curriculum_idx]

                # Skip training if sample already has a trained model
                if self._is_sample_trained(task, sample_num):
                    print(f"Sample {sample_num} of task {task['Name']} already trained, reloading.")
                    try:
                        reward_code, rew, traj, snapshots = self._load_existing_sample_data(task, sample_num)
                    except Exception as e:
                        print(f"Failed to reload sample {sample_num}: {e}. Will retrain.")
                    else:
                        self.current_reward_code_list.append(reward_code)
                        rew_rollout.append(rew)
                        traj_rollout.append(traj)
                        snapshot_rollout.append(snapshots)
                        decision = self.evaluate_subtask(curriculum_idx, sample_num, snapshots, traj)
                        if decision is None or decision == -1:
                            print(f"Failed to evaluate task {task['Name']} sample {sample_num}. Terminating.")
                            self.terminate_training = True
                            break
                        if decision['decision']:
                            print(f"Successfully trained task {task['Name']} sample {sample_num}.")
                            self.best_model_idx_list.append(sample_num)
                            self.best_reward_code[task['Name']] = reward_code
                            with open(self.logger_path + f"{task['Name']}/best_reward_code.txt", "w") as file:
                                file.write(reward_code)
                            self._save_run_state()
                            self.current_reward_code_list = []
                            break
                        else:
                            print(f"Failed to train task {task['Name']} sample {sample_num}. Retrying...")
                            print(f"Reason: {decision['reason']}")
                            failure_reasons.append(decision['reason'])
                            advice = self.get_advice(curriculum_idx, sample_num, failure_reasons)
                            if advice is None:
                                print(f"Failed to get advice for task {task['Name']} sample {sample_num}. Terminating.")
                                self.terminate_training = True
                                break
                            advices.append(advice)
                        continue

                # Make sure an executable reward function is generated and rollout is successful
                trial_num = 0
                while trial_num < MAX_ATTEMPT:
                    try:
                        if sample_num == 0:
                            self.generate_reward_code(curriculum_idx, sample_num, failure_reasons)
                        else:
                            self.refine_reward_code(curriculum_idx, sample_num, advices)
                        self.train_single(curriculum_idx, task, sample_num)

                    except Exception as e:
                        print(f"Error in training task {task['Name']} sample {sample_num}")
                        print(e)
                        traceback.print_exc()

                        # Save error message in log path and discard current reward code & directory
                        with open(self.logger_path + f"{task['Name']}/sample_{sample_num}_training_error_{trial_num}.txt", "w") as file:
                            file.write(str(e))
                        self.current_reward_code_list.pop()
                        if os.path.exists(self.logger_path + f"{task['Name']}/sample_{sample_num}"):
                            shutil.rmtree(self.logger_path + f"{task['Name']}/sample_{sample_num}")
                        print("Discarded current reward code & directory. Retrying...")

                        trial_num += 1
                        continue # if generating reward code fails, skip evaluation and pass to the next iteration

                    N_ROLLOUTS = 5 # Collect 5 rollouts to evaluate the task
                    task_success = False
                    evaluation_decision = -1
                    for rollout in range(N_ROLLOUTS):
                        try:
                            rew, traj, snapshots = self.collect_evaluation_data(task, sample_num, rollout)
                            rew_rollout.append(rew)
                            traj_rollout.append(traj)
                            snapshot_rollout.append(snapshots)

                            # Check if succeeded in current task
                            decision = self.evaluate_subtask(curriculum_idx, sample_num, snapshot_rollout[-1], traj_rollout[-1])
                            if decision is not None and decision != -1:
                                task_success = task_success or decision['decision']
                                evaluation_decision = decision
                                if task_success is True:
                                    break # Task is successfully trained, break the loop
                                elif rollout == N_ROLLOUTS - 1:
                                    print(f"Task {task['Name']} sample {sample_num} failed in every {N_ROLLOUTS} rollouts.")
                                    break
                                else:
                                    # If the rollout failed, remove the last entries
                                    rew_rollout.pop()
                                    traj_rollout.pop()
                                    snapshot_rollout.pop()

                        except Exception as e:
                            print(f"Error in rollout of task {task['Name']} sample {sample_num}")
                            print(e)
                            traceback.print_exc()

                            # Save error message in log path and discard current reward code & directory
                            with open(self.logger_path + f"{task['Name']}/sample_{sample_num}_evaluation_error_{trial_num}.txt", "w") as file:
                                file.write(str(e))
                            self.current_reward_code_list.pop()
                            if os.path.exists(self.logger_path + f"{task['Name']}/sample_{sample_num}"):
                                shutil.rmtree(self.logger_path + f"{task['Name']}/sample_{sample_num}")
                            print("Discarded current reward code & directory. Retrying...")

                            break # If evaluation fails, break the loop and retry training

                    if task_success or rollout == N_ROLLOUTS - 1:
                        break # If task is successfully trained, break the loop

                if evaluation_decision == -1:
                    print(f"Failed to evaluate task {task['Name']} sample {sample_num}. Terminating.")
                    self.terminate_training = True
                    break

                elif task_success: # Success
                    print(f"Successfully trained task {task['Name']} sample {sample_num}.")

                    # Update best reward
                    self.best_model_idx_list.append(sample_num)
                    self.best_reward_code[task['Name']] = self.current_reward_code_list[sample_num]
                    with open(self.logger_path + f"{task['Name']}/best_reward_code.txt", "w") as file:
                        file.write(self.current_reward_code_list[sample_num])
                    self._save_run_state()
                    self.current_reward_code_list = []
                    break # Move on to next task

                else: # Failure
                    print(f"Failed to train task {task['Name']} sample {sample_num}. Retrying...")
                    print(f"Reason: {evaluation_decision['reason']}")
                    failure_reasons.append(evaluation_decision['reason'])

                    # Get advice
                    advice = self.get_advice(curriculum_idx, sample_num, failure_reasons)
                    if advice is None:
                        print(f"Failed to get advice for task {task['Name']} sample {sample_num}. Terminating.")
                        self.terminate_training = True
                        break
                    advices.append(advice)

                if trial_num == MAX_ATTEMPT:
                    print(f"Failed to train task {task['Name']} sample {sample_num}. Terminating.")
                    with open(self.logger_path + f"{task['Name']}/failed_training.txt", "w") as file:
                        file.write(f"Failed to train task {task['Name']} sample {sample_num}. Terminating.")
                    self.terminate_training = True
                    break

            # Ask LLM to choose the best model
            if self.current_reward_code_list != []:
                print(f"Evaluating subtask alignment for task {task['Name']}...")
                self.evaluate_subtask_aligment(task, curriculum_idx, traj_rollout, rew_rollout)
                self._save_run_state()

            # Move on to the next curriculum task
            curriculum_idx += 1


    def train_single(self, curriculum_idx, task, sample_num):
        iter_per_task = self.cfg["iter_per_task"]
        # Strip CUDA_VISIBLE_DEVICES from subprocess env so IsaacGym sees the full
        # GPU list and the explicit physical device args below work correctly.
        subprocess_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        gpu_args = ["--sim_device", f"cuda:{self.gpu_id}",
                    "--rl_device", f"cuda:{self.gpu_id}",
                    "--graphics_device_id", str(self.gpu_id)]
        if curriculum_idx == 0:
            print(f"Training task {task['Name']} sample {sample_num} from scratch")
            process = subprocess.run([sys.executable,
                                        "-m",
                                        "openrl_ws.curriculum_train",
                                        "--task", self.task,
                                        "--run_date", self.experiment_time,
                                        "--curriculum_task", task['Name'],
                                        "--sample_idx", str(sample_num),
                                        "--training_iter", str(iter_per_task),
                                        ] + gpu_args,
                                        env=subprocess_env,
                                        )

        else:
            previous_task = self.curriculum[curriculum_idx - 1]
            load_sample_num = self.best_model_idx_list[curriculum_idx - 1]
            print(f"Training task {task['Name']} sample {sample_num} from previous task {previous_task['Name']} sample {load_sample_num}")
            process = subprocess.run([sys.executable,
                                        "-m",
                                        "openrl_ws.curriculum_train",
                                        "--task", self.task,
                                        "--run_date", self.experiment_time,
                                        "--curriculum_task", task['Name'],
                                        "--sample_idx", str(sample_num),
                                        "--load", "True",
                                        "--load_task", previous_task['Name'],
                                        "--load_sample_idx", str(load_sample_num),
                                        "--training_iter", str(iter_per_task),
                                        ] + gpu_args,
                                        env=subprocess_env,
                                        )


    def generate_reward_code(self, curriculum_idx, sample_num, failure_reasons):
        # Generate reward weight
        reward_code = None
        max_attempt = 5
        attempt = 0
        while reward_code is None and attempt < max_attempt:
            reward_code = self.gpt_api.generate_rewards(self.curriculum, curriculum_idx, sample_num,
                                                        reward_code_history=self.best_reward_code,
                                                        reward_samples=self.current_reward_code_list,
                                                        failure_reasons=failure_reasons)
            attempt += 1
            if reward_code is None:
                print("Failed to generate reward weight. Retrying...")

        # Update env code
        self.gpt_api.update_env_code(self.env_path, self.curriculum, curriculum_idx,
                                        reward_code=reward_code,
                                        version_number=sample_num)
        self.current_reward_code_list.append(reward_code)

    def refine_reward_code(self, curriculum_idx, sample_num, advices):
        # Refine reward weight
        reward_code = None
        max_attempt = 5
        attempt = 0
        while reward_code is None and attempt < max_attempt:
            reward_code = self.gpt_api.refine_reward(self.curriculum, curriculum_idx, sample_num,
                                                        failed_reward=self.current_reward_code_list[-1],
                                                        advice=advices[-1])
            attempt += 1
            if reward_code is None:
                print("Failed to refine reward weight. Retrying...")

        # Update env code
        self.gpt_api.update_env_code(self.env_path, self.curriculum, curriculum_idx,
                                        reward_code=reward_code,
                                        version_number=sample_num)
        self.current_reward_code_list.append(reward_code)

    def collect_evaluation_data(self, task, sample_num, rollout):
        print(f"Collecting evaluation data for task {task['Name']} sample {sample_num}")
        subprocess_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        # Save the trajectory analysis in the log path
        process = subprocess.run([sys.executable,
                                    "-m",
                                    "openrl_ws.curriculum_eval",
                                    "--task", self.task,
                                    "--run_date", self.experiment_time,
                                    "--curriculum_task", task['Name'],
                                    "--sample_idx", str(sample_num),
                                    "--seed", str(rollout),
                                    "--sim_device", f"cuda:{self.gpu_id}",
                                    "--graphics_device_id", str(self.gpu_id),
                                    ],
                                    env=subprocess_env,
                                    )

        # Load the trajectory and reward data
        save_path = os.path.join("logs", self.experiment_time, task['Name'], f"sample_{sample_num}", "model")
        with open(os.path.join(save_path, "traj_dict.pkl"), 'rb') as f:
            traj = pickle.load(f)
        with open(os.path.join(save_path, "rew_dict.pkl"), 'rb') as f:
            rew = pickle.load(f)

        # Read all snapshot images in order
        snapshot_pattern = os.path.join(save_path, 'images', "snapshot_*.png")
        snapshot_files = glob.glob(snapshot_pattern)

        # Sort by the numeric suffix
        snapshot_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Load images into a list
        snapshots = []
        for snapshot_file in snapshot_files:
            with open(snapshot_file, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                snapshots.append(encoded_string)

        return rew, traj, snapshots

    def evaluate_subtask(self, curriculum_idx, sample_num, snapshot, traj_rollout):
        # Ask LLM to evaluate the subtask: success or failure
        evaluation_decision = None
        trial = 0
        while evaluation_decision is None:
            evaluation_decision = self.gpt_api.evaluate_result(self.curriculum, curriculum_idx, sample_num,
                                                               snapshot, traj_rollout)
            trial += 1
            if trial == 5:
                evaluation_decision = -1
            if evaluation_decision == -1 or evaluation_decision is None:
                print("Evaluation error. Try again.")

        return evaluation_decision

    def get_advice(self, curriculum_idx, sample_num, failure_reasons):
        # Ask VLM for advice on how to improve the subtask
        advice = None
        trial = 0
        while advice is None:
            print("Advice error. Try again.")
            advice = self.gpt_api.get_advice(self.curriculum, curriculum_idx, sample_num,
                                             failed_reward=self.current_reward_code_list[-1],
                                             failure_reason=failure_reasons[-1])
            trial += 1
            if trial == 5:
                break

        return advice

    def evaluate_subtask_aligment(self, subtask, subtask_idx, traj_rollout, rew_rollout):
        # Asl LLM to choose the best model
        best_sample_idx = None
        trial = 0
        while best_sample_idx is None:
            best_sample_idx = self.gpt_api.feedback(self.curriculum, subtask_idx, traj_rollout, rew_rollout)
            trial += 1
            if trial == 5:
                best_sample_idx = 0

            if best_sample_idx is None and trial > 1:
                print("Statistics Analysis error. Try again.")

        self.best_model_idx_list.append(best_sample_idx)
        # Update best reward code list
        self.best_reward_code[subtask['Name']] = self.current_reward_code_list[best_sample_idx]

        # Save the best reward code list
        with open(self.logger_path + f"{subtask['Name']}/best_reward_code.txt", "w") as file:
            file.write(self.current_reward_code_list[best_sample_idx])

        self.current_reward_code_list = []
