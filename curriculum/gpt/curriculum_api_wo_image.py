import os
import re
import textwrap
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gpt.utils import *

GPT_MODEL = "gpt-4o" # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106 gpt-4-turbo-preview


class CurriculumAPI:
    def __init__(self, prompt_path, log_path, line_num):
        self.client = get_client()
        self.prompt_path = prompt_path
        self.log_path = log_path
        self.insert_line_num = line_num

    def generate_curriculum(self):
        initial_system = file_to_string(self.prompt_path + "/curriculum_system.txt")
        initial_user = file_to_string(self.prompt_path + "/curriculum_user.txt")

        # Add env description
        env_description = file_to_string(self.prompt_path + "/environment_description.txt")
        initial_user = initial_user.replace("<<Environment_Description>>", env_description)
        
        # Get the curriculum from GPT
        tasks_string = gpt_interaction(self.client, GPT_MODEL, initial_system, initial_user)

        # Ensure the directory exists and write the curriculum to a file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + "/curriculum.md", "w") as file:
            file.write(tasks_string)

        # Split the string into individual task sections
        task_sections = re.split(r"\n\n(?=Task)", tasks_string)

        # Extract details for all tasks
        datetime = os.path.basename(os.path.normpath(self.log_path))
        tasks_details = [extract_task_details(section, datetime) for section in task_sections]

        # Return list of dictionaries with task details
        return tasks_details
    
    def generate_rewards(self, curriculum, curriculum_idx, sample_num, reward_code_history):
        task_detail = curriculum[curriculum_idx]

        reward_system = file_to_string(self.prompt_path + "/reward_system.txt")
        reward_user = file_to_string(self.prompt_path + "/reward_user.txt")

        # Add env description
        env_description = file_to_string(self.prompt_path + "/environment_description.txt")
        reward_user = reward_user.replace("<<Environment_Description>>", env_description)

        # Concatenate the task details into the user strings
        reward_user = reward_user.replace("<<Task_Name>>", f'{task_detail["Task_num"]}_{task_detail["Subtask"]}')
        reward_user = reward_user.replace("<<Task_Description>>", task_detail["Description"])
        reward_user = reward_user.replace("<<Task_Reason>>", task_detail["Reason"])

        # Add previous task and reward information
        if curriculum_idx > 0:
            i = curriculum_idx - 1
            task_history_details = curriculum[i]
            reward_code = reward_code_history[task_history_details["Name"]]
            reward_history = file_to_string(self.prompt_path + "/reward_history.txt")
            reward_history = reward_history.replace("<<Task_Name>>", f'{task_history_details["Task_num"]}_{task_history_details["Subtask"]}')
            reward_history = reward_history.replace("<<Task_Description>>", task_history_details["Description"])
            reward_history = reward_history.replace("<<Task_Reason>>", task_history_details["Reason"])
            reward_history = reward_history.replace("<<Task_Code>>", reward_code)

            reward_user = reward_user + "\n" + reward_history

        # Get reward function from GPT
        reward_answer = gpt_interaction(self.client, GPT_MODEL, reward_system, reward_user)

        pattern = r"`python(.*?)`"
        reward_match = re.search(pattern, reward_answer, re.DOTALL)

        if reward_match:
            reward_code_block = reward_match.group(1)
            
            # Save prompt and answer
            os.makedirs(os.path.join(self.log_path, f'{task_detail["Name"]}'), exist_ok=True)
            with open(self.log_path + f'{task_detail["Name"]}/' + f"generate_reward_prompt_{sample_num}.md", "w") as file:
                file.write(reward_user)
            with open(self.log_path + f'{task_detail["Name"]}/' + f"generate_reward_answer_{sample_num}.md", "w") as file:
                file.write(reward_answer)

            return reward_code_block
        else:
            print("No code block found.")
        return None

    def redesign_rewards(self, curriculum, curriculum_idx, sample_num, reward_code_list, previous_rew_rollout, previous_traj_rollout):
        task_detail = curriculum[curriculum_idx]

        redesign_rew_system = file_to_string(self.prompt_path + "/redesign_reward_system.txt")
        redesign_rew_user = file_to_string(self.prompt_path + "/redesign_reward_user.txt")

        # Add env description
        env_description = file_to_string(self.prompt_path + "/environment_description.txt")
        redesign_rew_user = redesign_rew_user.replace("<<Environment_Description>>", env_description)

        # Concatenate the task details into the user strings
        redesign_rew_user = redesign_rew_user.replace("<<Task_Name>>", f'{task_detail["Task_num"]}_{task_detail["Subtask"]}')
        redesign_rew_user = redesign_rew_user.replace("<<Task_Description>>", task_detail["Description"])
        redesign_rew_user = redesign_rew_user.replace("<<Task_Reason>>", task_detail["Reason"])

        # Replace sample reward code
        redesign_rew_user = redesign_rew_user.replace("<<Reward_Code0>>", reward_code_list[0])
        redesign_rew_user = redesign_rew_user.replace("<<Reward_Code1>>", reward_code_list[1])

        # Add rollout statistics
        for sample_idx in range(len(previous_rew_rollout)):
            feedback_statistics = "\n"
            try:
                # Reward statistics
                feedback_statistics += f"Episode rewards:\n"
                for key in previous_rew_rollout[sample_idx].keys():
                    if key == 'total_reward': # write total_reward at the end
                        continue
                    feedback_statistics += f"{key}: {previous_rew_rollout[sample_idx][key]}\n"
                feedback_statistics += f"Total Reward: {previous_rew_rollout[sample_idx]['total_reward']}\n"
                
                # Trajectory statistics
                feedback_statistics += "\nTrajectory rollout:\n"
                if 'nan_reward' in previous_traj_rollout[sample_idx].keys():
                    feedback_statistics += "Experiment failed due to NaN reward value. \nThis is likely because the reward magnitude is too big.\n\n"
                else:
                    for key in previous_traj_rollout[sample_idx].keys():
                        if key == 'is_success': # skip is_success
                            continue
                        feedback_statistics += f"{key}:\n"
                        feedback_statistics += f"{previous_traj_rollout[sample_idx][key]}\n"
                    feedback_statistics += f"Package reached goal: "
                    feedback_statistics += "True" if previous_traj_rollout[sample_idx]['is_success'] == 1 else "False"
                    feedback_statistics += "\n"
            except:
                feedback_statistics += "No statistics available\n\n"
            redesign_rew_user = redesign_rew_user.replace(f"<<Statistics{sample_idx}>>", feedback_statistics)

        # Get reward function from GPT
        reward_answer = gpt_interaction(self.client, GPT_MODEL, redesign_rew_system, redesign_rew_user)

        pattern = r"`python(.*?)`"
        reward_match = re.search(pattern, reward_answer, re.DOTALL)

        if reward_match:
            reward_code_block = reward_match.group(1)
            os.makedirs(os.path.join(self.log_path, f'{task_detail["Name"]}'), exist_ok=True)
            with open(self.log_path + f'{task_detail["Name"]}/' + f"redesign_reward_prompt_{sample_num}.md", "w") as file:
                file.write(redesign_rew_system + "\n\n\n" + redesign_rew_user)
            with open(self.log_path + f'{task_detail["Name"]}/' + f"redesign_reward_answer_{sample_num}.md", "w") as file:
                file.write(reward_answer)
            
            return reward_code_block
        else:
            print("No code block found.")
        return None
    
    def update_env_code(self, reward_code_path, curriculum, curriculum_idx, reward_code, version_number=0):
        # Created environment with task and save as version = env_version
        # Save the reward code
        task = curriculum[curriculum_idx]
        save_string_to_file(self.log_path + f"{task['Name']}/sample_{version_number}/" + "reward_code.md", reward_code)

        # Save the updated reward code
        new_code_path = reward_code_path.replace("_source.py", ".py")

        def insert_line_in_file(source_file_path, new_file_path, string_to_insert, line_number):
            # Read the current content of the file
            with open(source_file_path, 'r') as file:
                lines = file.readlines()
            
            # Insert the string at the specified line number
            indented_text = textwrap.indent(string_to_insert, '    ')
            lines.insert(line_number - 1, indented_text + '\n')
            
            # Write the updated content back to the file
            with open(new_file_path, 'w') as file:
                file.writelines(lines)

        insert_line_in_file(reward_code_path, new_code_path, reward_code, self.insert_line_num)
        print(f"Updated reward code saved to {new_code_path}")

        return reward_code

    def feedback(self, curriculum, curriculum_idx, trajectories, rewards):
        feedback_system = file_to_string(self.prompt_path + "/feedback_system.txt")
        feedback_user = file_to_string(self.prompt_path + "/feedback_user.txt")

        task = curriculum[curriculum_idx]
        # Concatenate the task details into the user strings
        feedback_user = feedback_user.replace("<<Task_Name>>", f'{task["Task_num"]}_{task["Subtask"]}')
        feedback_user = feedback_user.replace("<<Task_Description>>", task["Description"])
        feedback_user = feedback_user.replace("<<Task_Reason>>", task["Reason"])

        # Add previous task information
        if curriculum_idx > 0:
            for i in range(curriculum_idx):
                task_history_details = curriculum[i]
                feedback_history = file_to_string(self.prompt_path + "/feedback_history.txt")
                feedback_history = feedback_history.replace("<<Task_Name>>", f'{task_history_details["Task_num"]}_{task_history_details["Subtask"]}')
                feedback_history = feedback_history.replace("<<Task_Description>>", task_history_details["Description"])
                feedback_history = feedback_history.replace("<<Task_Reason>>", task_history_details["Reason"])

                feedback_user = feedback_user + "\n" + feedback_history

        # Statistics to string
        feedback_statistics = "\nStatistics:\n"
        for experiment_idx in range(len(trajectories)):
            feedback_statistics += f"\nExperiment {experiment_idx}:\n"
            try:
                feedback_statistics += "Episode trajectories:\n"
                traj_dict = trajectories[experiment_idx]
                for key in traj_dict.keys():
                    feedback_statistics += f"{key}:\n"
                    feedback_statistics += f"{traj_dict[key]}\n"
                
                # feedback_statistics += "Episode rewards:\n"
                # rew_dict = rewards[experiment_idx]
                # for key in rew_dict.keys():
                #     if key == 'total_reward': # write total_reward at the end
                #         continue
                #     feedback_statistics += f"{key}: {rew_dict[key]}\n"
                # feedback_statistics += f"Total Reward: {rew_dict['total_reward']}\n"
            except:
                feedback_statistics += "No statistics available\n\n"

        feedback_user = feedback_user + "\n" + feedback_statistics

        gpt_answer = gpt_interaction(self.client, GPT_MODEL, feedback_system, feedback_user)

        # Ensure the directory exists and write the curriculum to a file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + task["Name"] + "_statistics.md", "w") as file:
            file.write(feedback_user)
        with open(self.log_path + task["Name"] + ".md", "w") as file:
            file.write(gpt_answer)

        decision = gpt_answer.split("\n")[0]
        numbers = re.findall(r"\d+", decision)
        if numbers:
            return int(numbers[0])
        else:
            print("No number found in the decision.")
            return None
        
# Function to extract details from each task section
def extract_task_details(task_section, datetime):

    details = {}
    lines = task_section.split("\n")
    for line in lines:
        if line.startswith("Task"):
            details["Task_num"] = line.split(" ")[1].strip()
        elif line.startswith("Name:"):
            details["Subtask"] = line.split(": ")[1].strip()
            details["Name"] = f'{details["Task_num"]}_{details["Subtask"]}({datetime})'
        elif line.startswith("Description:"):
            details["Description"] = ": ".join(line.split(": ")[1:])
        elif line.startswith("Reason:"):
            details["Reason"] = ": ".join(line.split(": ")[1:])
    return details