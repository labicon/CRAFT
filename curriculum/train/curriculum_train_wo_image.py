import os
import traceback
import pickle
import shutil
import subprocess

from gpt.curriculum_api_wo_image import CurriculumAPI
from gpt.utils import *

MAX_ATTEMPT = 5

class Curriculum_Module:
    def __init__(self, env_path, logger_path, run_datetime, cfg, seed=0):
        self.env_path = env_path
        self.prompt_path = "./curriculum/gpt/prompts/go2seesaw"
        self.gpt_api = CurriculumAPI(self.prompt_path, logger_path, 
                                     line_num=cfg['line_num'])
        self.logger_path = logger_path
        self.best_reward_code = {}
        self.best_model_idx_list = []
        self.current_reward_code_list = []
        self.best_sim_reward = None
        self.cfg = cfg
        self.seed = seed
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

    def train(self):
        self.generate_curriculum()

        curriculum_idx = 0
        while curriculum_idx < self.curriculum_length:
            if self.terminate_training:
                break
            traj_rollout = []
            rew_rollout = []
            for sample_num in range(self.cfg["num_reward_samples"]):
                task = self.curriculum[curriculum_idx]
                
                # Make sure an executable reward function is generated and rollout is successful
                trial_num = 0
                while trial_num < MAX_ATTEMPT:
                    try:
                        self.generate_reward_code(curriculum_idx, sample_num)                        
                        self.train_single(curriculum_idx, task, sample_num)
                    
                    except Exception as e:
                        print(f"Error in training task {task['Name']} sample {sample_num}")
                        print(e)
                        traceback.print_exc()
                        
                        if e == "": # If error is due to NanAction, end training and move to collecting evaluation data
                            self.nan_reward = True
                            pass                            
                        else:
                            # Save error message in log path and discard current reward code & directory
                            os.makedirs(self.logger_path + f"{task['Name']}/sample_{sample_num}/", exist_ok=True)
                            with open(self.logger_path + f"{task['Name']}/sample_{sample_num}_training_error_{trial_num}.txt", "w") as file:
                                file.write(str(e))
                            self.current_reward_code_list.pop()
                            if os.path.exists(self.logger_path + f"{task['Name']}/sample_{sample_num}"):
                                shutil.rmtree(self.logger_path + f"{task['Name']}/sample_{sample_num}")
                            print("Discarded current reward code & directory. Retrying...")

                            trial_num += 1
                            continue # if generating reward code fails, skip evaluation and pass to the next iteration

                    try:
                        rew, traj = self.collect_evaluation_data(task, curriculum_idx, sample_num)
                        rew_rollout.append(rew)
                        traj_rollout.append(traj)

                        if self.nan_reward:
                            traj_rollout['nan_reward'] = True
                            self.nan_reward = False
                            
                        break # if training and rollout is successful, break the loop

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
                        
                        trial_num += 1

                if trial_num == MAX_ATTEMPT:
                    print(f"Failed to train task {task['Name']} sample {sample_num}. Terminating.")
                    with open(self.logger_path + f"{task['Name']}/failed_training.txt", "w") as file:
                        file.write(f"Failed to train task {task['Name']} sample {sample_num}. Terminating.")
                    self.terminate_training = True
                    break

            # Ask LLM to choose the best model
            self.evaluate_subtask_aligment(task, curriculum_idx, traj_rollout, rew_rollout)
            curriculum_idx += 1


    def train_single(self, curriculum_idx, task, sample_num, executable_test=False):
        iter_per_task = self.cfg["iter_per_task"]
        if curriculum_idx == 0:
            print(f"Training task {task['Name']} sample {sample_num} from scratch")
            process = subprocess.run(["python", 
                                        "/home/kang/MANavigation/skrl_ws/curriculum_train.py", 
                                        "--run_date", self.experiment_time,
                                        "--curriculum_task", task['Name'], 
                                        "--sample_idx", str(sample_num),
                                        "--training_iter", str(iter_per_task),
                                        ],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)
            print(process.stdout)
            print(process.stderr)
            # Save stdout and stderr to log file
            with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/training_log.txt", "w") as file:
                file.write(process.stdout)
                file.write(process.stderr)

        else:
            previous_task = self.curriculum[curriculum_idx - 1]
            load_sample_num = self.best_model_idx_list[curriculum_idx - 1]
            print(f"Training task {task['Name']} sample {sample_num} from previous task {previous_task['Name']} sample {load_sample_num}")
            process = subprocess.run(["python", 
                                        "/home/kang/MANavigation/skrl_ws/curriculum_train.py", 
                                        "--run_date", self.experiment_time,
                                        "--curriculum_task", task['Name'], 
                                        "--sample_idx", str(sample_num),
                                        "--load", "True",
                                        "--load_task", previous_task['Name'],
                                        "--load_sample_idx", str(load_sample_num),
                                        "--training_iter", str(iter_per_task),
                                        ],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)
            print(process.stdout)
            print(process.stderr)
            with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/training_log.txt", "w") as file:
                file.write(process.stdout)
                file.write(process.stderr)


    def generate_reward_code(self, curriculum_idx, sample_num):
        # Generate reward weight
        reward_code = None
        max_attempt = 5
        attempt = 0
        while reward_code is None and attempt < max_attempt:
            reward_code = self.gpt_api.generate_rewards(self.curriculum, curriculum_idx, sample_num,
                                                                reward_code_history=self.best_reward_code)
            attempt += 1
            if reward_code is None:
                print("Failed to generate reward weight. Retrying...")
        
        # Update env code
        self.gpt_api.update_env_code(self.env_path, self.curriculum, curriculum_idx, 
                                                    reward_code=reward_code, 
                                                    version_number=sample_num)
        self.current_reward_code_list.append(reward_code)
        

    def redesign_reward_weight(self, curriculum_idx, sample_num, previous_rew_rollout, previous_traj_rollout):
        # Generate reward code
        reward_code = None
        max_attempt = 5
        attempt = 0
        while reward_code is None and attempt < max_attempt:
            reward_code = self.gpt_api.redesign_weight(self.curriculum, curriculum_idx, sample_num,
                                                        reward_code_list=self.current_reward_code_list[-self.cfg["num_reward_feedback_samples"]:],
                                                        previous_rew_rollout=previous_rew_rollout,
                                                        previous_traj_rollout=previous_traj_rollout)
            attempt += 1
            if reward_code is None:
                print("Failed to generate reward code. Retrying...")
        
        # Update env code
        reward_code = self.gpt_api.update_env_code(self.env_path, self.curriculum, curriculum_idx,
                                    reward_code=reward_code,  
                                    version_number=sample_num)
        self.current_reward_code_list.append(reward_code)

    def collect_evaluation_data(self, task, curriculum_idx, sample_num):    
        print(f"Collecting evaluation data for task {task['Name']} sample {sample_num}")  
        # Save the trajectory analysis in the log path        
        process = subprocess.run(["python",
                                    "/home/kang/MANavigation/skrl_ws/curriculum_eval.py",
                                    "--run_date", self.experiment_time,
                                    "--curriculum_task", task['Name'],
                                    "--sample_idx", str(sample_num),
                                    ],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
        # Save stdout and stderr to log file
        with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/evaluation_log.txt", "w") as file:
            file.write(process.stdout)
            file.write(process.stderr)
        # Load the trajectory and reward data
        save_path = os.path.join("runs", self.experiment_time, task['Name'], f"sample_{sample_num}")
        with open(os.path.join(save_path, "traj_dict.pkl"), 'rb') as f:
            traj = pickle.load(f)
        with open(os.path.join(save_path, "rew_dict.pkl"), 'rb') as f:
            rew = pickle.load(f)
        
        return rew, traj

    def evaluate_subtask_aligment(self, subtask, subtask_idx, traj_rollout, rew_rollout):
        # Asl LLM to choose the best model
        best_sample_idx = None
        trial = 0
        while best_sample_idx is None:
            best_sample_idx = self.gpt_api.feedback(self.curriculum, subtask_idx, traj_rollout, rew_rollout)
            trial += 1
            if best_sample_idx is None:
                print("Statistics Analysis error. Try again.")
            if trial == 5:
                best_sample_idx = 0

        self.best_model_idx_list.append(best_sample_idx)
        # Update best reward code list
        self.best_reward_code[subtask['Name']] = self.current_reward_code_list[best_sample_idx]

        # Save the best reward code list
        with open(self.logger_path + f"{subtask['Name']}/best_reward_code.txt", "w") as file:
            file.write(self.current_reward_code_list[best_sample_idx])
            
        self.current_reward_code_list = []