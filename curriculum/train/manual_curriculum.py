import os
import traceback
import pickle
import shutil
import subprocess

from gpt.curriculum_api import CurriculumAPI, extract_task_details
from gpt.utils import *

MAX_ATTEMPT = 5

class Manual_Module:
    def __init__(self, env_path, logger_path, run_datetime, cfg, seed=0):
        self.env_path = env_path
        self.prompt_path = "/home/kang/multiagent-quadruped-environment/curriculum/gpt/prompts" + "/manual_curriculum"
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
        self.nan_reward = False
        self.experiment_time = run_datetime
        
    def load_curriculum(self):
        # Load curriculum
        with open(self.prompt_path + "/curriculum.txt", "r") as f:
            tasks_string = f.read()
        task_sections = re.split(r"\n\n(?=Task)", tasks_string)

        # Extract details for all tasks
        datetime = os.path.basename(os.path.normpath(self.logger_path))
        self.curriculum = [extract_task_details(section, datetime) for section in task_sections]
        self.curriculum_length = len(self.curriculum)

    def train(self):
        self.load_curriculum()

        curriculum_idx = 0
        while curriculum_idx < self.curriculum_length:
            traj_rollout = []
            rew_rollout = []

            task = self.curriculum[curriculum_idx]
                
            self.replace_reward_code(curriculum_idx, 0)                        
            self.train_single(curriculum_idx, task, 0)
                    
            rew, traj = self.collect_evaluation_data(task, curriculum_idx, 0)
            rew_rollout.append(rew)
            traj_rollout.append(traj)

            # Ask LLM to choose the best model
            self.evaluate_subtask_aligment(task, curriculum_idx, traj_rollout, rew_rollout)
            curriculum_idx += 1


    def train_single(self, curriculum_idx, task, sample_num, executable_test=False):
        iter_per_task = self.cfg["iter_per_task"]
        if curriculum_idx == 0:
            print(f"Training task {task['Name']} sample {sample_num} from scratch")
            process = subprocess.run(["python", 
                                        "/home/kang/multiagent-quadruped-environment/openrl_ws/curriculum_train.py", 
                                        "--run_date", self.experiment_time,
                                        "--curriculum_task", task['Name'], 
                                        "--sample_idx", str(sample_num),
                                        "--training_iter", str(iter_per_task),
                                        ])

        else:
            previous_task = self.curriculum[curriculum_idx - 1]
            load_sample_num = self.best_model_idx_list[curriculum_idx - 1]
            print(f"Training task {task['Name']} sample {sample_num} from previous task {previous_task['Name']} sample {load_sample_num}")
            process = subprocess.run(["python", 
                                        "/home/kang/multiagent-quadruped-environment/openrl_ws/curriculum_train.py", 
                                        "--run_date", self.experiment_time,
                                        "--curriculum_task", task['Name'], 
                                        "--sample_idx", str(sample_num),
                                        "--load", "True",
                                        "--load_task", previous_task['Name'],
                                        "--load_sample_idx", str(load_sample_num),
                                        "--training_iter", str(iter_per_task),
                                        ])


    def replace_reward_code(self, curriculum_idx, sample_num):
        reward_path = os.path.join(self.prompt_path, f'reward_{curriculum_idx}.txt')
        with open(reward_path, "r", encoding="utf-8") as f:
            reward_string = f.read()
        pattern = r"`python(.*?)`"
        reward_match = re.search(pattern, reward_string, re.DOTALL)
        if reward_match:
            reward_code = reward_match.group(1)
        print(reward_code)    
        
        # Update env code
        # self.gpt_api.update_env_code(self.env_path, self.curriculum, curriculum_idx, 
        #                                             reward_code=reward_code, 
        #                                             version_number=sample_num)
        self.current_reward_code_list.append(reward_code)
        

    def collect_evaluation_data(self, task, curriculum_idx, sample_num):    
        print(f"Collecting evaluation data for task {task['Name']} sample {sample_num}")  
        # Save the trajectory analysis in the log path        
        process = subprocess.run(["python",
                                    "/home/kang/multiagent-quadruped-environment/openrl_ws/curriculum_eval.py",
                                    "--run_date", self.experiment_time,
                                    "--curriculum_task", task['Name'],
                                    "--sample_idx", str(sample_num),
                                    ])
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
        if not os.path.exists(self.logger_path + f"{subtask['Name']}"):
            os.makedirs(self.logger_path + f"{subtask['Name']}")
        with open(self.logger_path + f"{subtask['Name']}/best_reward_code.txt", "w") as file:
            file.write(self.current_reward_code_list[best_sample_idx])
            
        self.current_reward_code_list = []