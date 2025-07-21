import os
import traceback
import pickle
import shutil
import subprocess
import glob
import base64
from io import BytesIO

from gpt.curriculum_api import CurriculumAPI
from gpt.utils import *

MAX_ATTEMPT = 10

class Curriculum_Module:
    def __init__(self, env_path, logger_path, run_datetime, cfg, seed=0):
        self.env_path = env_path
        self.prompt_path = "/home/kang/multiagent-quadruped-environment/curriculum/gpt/prompts/go1gate"
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

    def open_curriculum(self):
        # Open curriculum pkl file - not tested
        file_path = "logs/robosuite_two-arm-lift/06-14_04-19/curriculum.pkl"
        with open(file_path, 'rb') as file:
            self.curriculum = pickle.load(file)
            self.curriculum_length = len(self.curriculum)

            print(os.path.basename(os.path.normpath(self.logger_path)))
            datetime = os.path.basename(os.path.normpath(self.logger_path))
            for task_detail in self.curriculum:
                task_detail['Name'] = f'{task_detail["Task_num"]}_{task_detail["Subtask"]}({datetime})'
            
            print(f"Curriculum loaded from {file_path}")
            print(self.curriculum) 
        
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
        # self.open_curriculum()

        # curriculum_idx = 3
        # # stack best_reward_code
        # for i in range(curriculum_idx):
        #     subtask = self.curriculum[i]
        #     with open(os.path.join(self.logger_path, subtask['Name'], "best_reward_code.txt"), "r") as file:
        #         reward_code = file.read()
        #         self.best_reward_code[subtask['Name']] = reward_code
        #         print(f"Loaded best reward code for {subtask['Name']}")
        #         print(reward_code)
        # # stack best_model_idx_list
        # self.best_model_idx_list = [1, 2, 2]

        curriculum_idx = 0
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

                    try:
                        rew, traj, snapshots = self.collect_evaluation_data(task, sample_num)
                        rew_rollout.append(rew)
                        traj_rollout.append(traj)
                        snapshot_rollout.append(snapshots)

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

                # Check if succeeded in current task
                evaluation_decision = self.evaluate_subtask(curriculum_idx, sample_num, snapshot_rollout[-1], traj_rollout[-1])
                if evaluation_decision == -1:
                    print(f"Failed to evaluate task {task['Name']} sample {sample_num}. Terminating.")
                    self.terminate_training = True
                    break

                elif evaluation_decision['decision']: # Success
                    print(f"Successfully trained task {task['Name']} sample {sample_num}.")
        
                    # Update best reward 
                    self.best_model_idx_list.append(sample_num)
                    self.best_reward_code[task['Name']] = self.current_reward_code_list[sample_num]
                    with open(self.logger_path + f"{task['Name']}/best_reward_code.txt", "w") as file:
                        file.write(self.current_reward_code_list[sample_num])
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

            # Move on to the next curriculum task
            curriculum_idx += 1


    def train_single(self, curriculum_idx, task, sample_num):
        iter_per_task = self.cfg["iter_per_task"]
        if curriculum_idx == 0:
            print(f"Training task {task['Name']} sample {sample_num} from scratch")
            process = subprocess.run(["python", 
                                        "/home/kang/multiagent-quadruped-environment/openrl_ws/curriculum_train.py",
                                        "--run_date", self.experiment_time,
                                        "--curriculum_task", task['Name'], 
                                        "--sample_idx", str(sample_num),
                                        "--training_iter", str(iter_per_task),
                                        ],
                                        # stdout=subprocess.PIPE,
                                        # stderr=subprocess.PIPE,
                                        # text=True
                                        )
            # print(process.stdout)
            # print(process.stderr)
            # Save stdout and stderr to log file
            # with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/training_log.txt", "w") as file:
            #     file.write(process.stdout)
            #     file.write(process.stderr)

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
                                        ],
                                                # stdout=subprocess.PIPE,
                                                # stderr=subprocess.PIPE,
                                                # text=True
                                        )
            # print(process.stdout)
            # print(process.stderr)
            # with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/training_log.txt", "w") as file:
            #     file.write(process.stdout)
            #     file.write(process.stderr)


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

    def collect_evaluation_data(self, task, sample_num):    
        print(f"Collecting evaluation data for task {task['Name']} sample {sample_num}")  
        # Save the trajectory analysis in the log path        
        process = subprocess.run(["python",
                                    "/home/kang/multiagent-quadruped-environment/openrl_ws/curriculum_eval.py",
                                    "--run_date", self.experiment_time,
                                    "--curriculum_task", task['Name'],
                                    "--sample_idx", str(sample_num),
                                    ],
                                    # stdout=subprocess.PIPE,
                                    # stderr=subprocess.PIPE,
                                    # text=True
                                    )
        # # Save stdout and stderr to log file
        # with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/evaluation_log.txt", "w") as file:
        #     file.write(process.stdout)
        #     file.write(process.stderr)
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