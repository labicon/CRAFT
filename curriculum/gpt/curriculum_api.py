import os
import re
import textwrap
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from curriculum.gpt.utils import *

GPT_LLM_MODEL = "gpt-4o-2024-08-06" # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106 gpt-4-turbo-preview
GPT_VLM_MODEL = "o4-mini-2025-04-16"

class CurriculumAPI:
    def __init__(self, prompt_path, log_path, line_num):
        self.client = get_client()
        self.prompt_path = prompt_path
        self.log_path = log_path
        self.insert_line_num = line_num
        self.task_name = os.path.basename(os.path.normpath(prompt_path))

    def generate_curriculum(self):
        initial_system = file_to_string(self.prompt_path + "/curriculum_system.txt")
        initial_user = file_to_string(self.prompt_path + "/curriculum_user.txt")

        # Add env description
        env_description = file_to_string(self.prompt_path + "/environment_description.txt")
        initial_user = initial_user.replace("<<Environment_Description>>", env_description)
        
        # Get three curriculums from GPT
        candidates = []
        for _ in range(3):
            tasks_string = gpt_interaction(self.client, GPT_LLM_MODEL, initial_system, initial_user)
            candidates.append(tasks_string)

        # Curriculum refinement
        curriculum_refine_system = file_to_string(self.prompt_path + "/curriculum_refine_system.txt")
        curriculum_refine_user = file_to_string(self.prompt_path + "/curriculum_refine_user.txt")
        curriculum_refine_user = curriculum_refine_user.replace("<<Environment_Description>>", env_description)
        for i, curriculum in enumerate(candidates):
            curriculum_refine_user = curriculum_refine_user.replace(f"<<Candidate_{i+1}>>", curriculum)

        # Save the user prompt for curriculum refinement
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + "/curriculum_refine_user.md", "w") as file:
            file.write(curriculum_refine_user)

        # Get the final curriculum from GPT
        tasks_string = gpt_interaction(self.client, GPT_LLM_MODEL, curriculum_refine_system, curriculum_refine_user)

        # Ensure the directory exists and write the curriculum to a file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + "/curriculum.md", "w") as file:
            file.write(tasks_string)

        # Split the string into individual task sections
        task_sections = re.split(r"\n\n(?=Task)", tasks_string)

        # Extract details for all tasks
        datetime = os.path.basename(os.path.normpath(self.log_path))
        tasks_details = [extract_task_details(section, datetime) for section in task_sections]
        tasks_details = [d for d in tasks_details if d] # filter out empty dictionaries

        # Return list of dictionaries with task details
        return tasks_details
    
    def evaluate_result(self, curriculum, curriculum_idx, sample_num, snapshots, traj_dict):
        # Evalute success/failure of the subtask
        # traj_dict is the last trajectory in traj_rollout

        evaluation_system = file_to_string(self.prompt_path + "/evaluation_system.txt")
        evaluation_user = file_to_string(self.prompt_path + "/evaluation_user.txt")

        # Add task details
        task_detail = file_to_string(self.prompt_path + "/feedback_history.txt")
        current_task = curriculum[curriculum_idx]
        task_detail = task_detail.replace("<<Task_Name>>", f'{current_task["Task_num"]}_{current_task["Subtask"]}')
        task_detail = task_detail.replace("<<Task_Description>>", current_task["Description"])
        task_detail = task_detail.replace("<<Task_Reason>>", current_task["Reason"])
        evaluation_user = evaluation_user.replace("<<Current task>>", task_detail)

        # Add previous task information
        former_task_string = ""
        if curriculum_idx > 0:
            for i in range(curriculum_idx):
                task_details = curriculum[i]
                evaluation_history = file_to_string(self.prompt_path + "/feedback_history.txt")
                evaluation_history = evaluation_history.replace("<<Task_Name>>", f'{task_details["Task_num"]}_{task_details["Subtask"]}')
                evaluation_history = evaluation_history.replace("<<Task_Description>>", task_details["Description"])
                evaluation_history = evaluation_history.replace("<<Task_Reason>>", task_details["Reason"])

                former_task_string = former_task_string + "\n" + evaluation_history
        else: 
            former_task_string = "No previous task learned."
        evaluation_user = evaluation_user.replace("<<Former tasks>>", former_task_string)

        # Add trajectory
        trajectory_string = ""
        try:
            for key in traj_dict.keys():
                if 'orientation' in key or 'distance' in key:
                    continue
                trajectory_string += f"{key}:\n"
                if type(traj_dict[key]) == str:  # if the value is a string, write it directly
                    trajectory_string += f"{traj_dict[key]}\n"
                else:
                    trajectory_string += np.array2string(
                        traj_dict[key],
                        formatter={'float_kind': lambda x: f"{x:.3f}"}) + "\n"
        except:
            trajectory_string += "No statistics available\n\n"
        evaluation_user = evaluation_user.replace("<<Trajectory>>", trajectory_string)
        
        if curriculum_idx != len(curriculum) - 1: # not the last task in the curriculum
            evaluation_user += "\nNote that this is not the final goal in the curriculum, don't be too strict on the decision."

        # Save the user prompt for evaluation
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + f"{current_task['Name']}/sample_{sample_num}/" + "evaluation_user.md", "w") as file:
            file.write(evaluation_user)

        encoded_image_list = snapshots # Snapshots are already base64 encoded

        gpt_answer = gpt_interaction_image(self.client, GPT_VLM_MODEL, evaluation_system, evaluation_user, encoded_image_list)

        # Save the GPT answer
        with open(self.log_path + f"{current_task['Name']}/sample_{sample_num}/" + "evaluation_answer.md", "w") as file:
            file.write(gpt_answer)

        # Parse the decision from GPT's response
        result = {"decision": None, "reason": ""}

        lines = gpt_answer.strip().splitlines()
        for idx, raw in enumerate(lines):
            line = raw.strip()

            # 1) If it starts with "Decision:", extract Success/Failure
            if line.startswith("Decision:"):
                # Try matching "[Success]" or "[Failure]" inside brackets first
                m = re.search(r"Decision:\s*\[(Success|Failure)\]", line)
                if m:
                    decision_text = m.group(1)
                else:
                    decision_text = line[len("Decision:"):].strip()

                if decision_text.lower() == "success":
                    result["decision"] = True
                elif decision_text.lower() == "failure":
                    result["decision"] = False
                else:
                    result["decision"] = None
                continue

            # 2) If it starts with "Reason:", capture that line’s remainder plus all following lines
            if line.startswith("Reason:"):
                after_colon = line[len("Reason:"):].strip()
                reason_lines = []
                if after_colon:
                    reason_lines.append(after_colon)

                # Append every line after the current one
                for j in range(idx + 1, len(lines)):
                    reason_lines.append(lines[j].rstrip())
                result["reason"] = "\n".join(reason_lines).strip()
                break

        return result
    

    def get_advice(self, curriculum, curriculum_idx, sample_num, failed_reward, failure_reason):
        advice_system = file_to_string(self.prompt_path + "/advice_system.txt")
        advice_user = file_to_string(self.prompt_path + "/advice_user.txt")

        # Add current task
        task = curriculum[curriculum_idx]
        advice_user = advice_user.replace("<<Task_Name>>", f'{task["Task_num"]}_{task["Subtask"]}')
        advice_user = advice_user.replace("<<Task_Description>>", task["Description"])
        advice_user = advice_user.replace("<<Task_Reason>>", task["Reason"])

        # Add current reward function
        advice_user = advice_user.replace("<<Reward_Function>>", failed_reward)

        # Add failure reason
        advice_user = advice_user.replace("<<Failure_Reason>>", failure_reason)

        # Find tensorboard event file
        pattern = os.path.join("./training-log", "MQE-Curriculum", self.task_name,
                               f"{task['Name']}_sample_{sample_num}/",
                               "tensorboard",
                               "events.out.tfevents.*")
        event_files = glob.glob(pattern)
        if not event_files:
            raise FileNotFoundError(f"No event files found in {pattern}")
        event_file = max(event_files, key=os.path.getmtime)
        print(f"Using event file: {event_file}")
        ea = EventAccumulator(event_file)
        ea.Reload()

        # Collect reward curve - save than reload (easier this way)
        curve_img_path = []
        for tag in ea.Tags()["scalars"]:
            if 'Reward' in tag:
                reward_curve = []
                for scalar in ea.Scalars(tag):
                    reward_curve.append([scalar.step, scalar.value])
                reward_curve = np.array(reward_curve)  
                # name = tag # .removeprefix('Reward / ')
                def remove_prefix(text, prefix):
                    if text.startswith(prefix):
                        return text[len(prefix):]
                    return text
                name = remove_prefix(tag, 'Reward/')

                plt.figure(figsize=(5, 4))
                plt.plot(reward_curve[:, 0], reward_curve[:, 1])
                plt.xlabel('Step', fontsize=16)
                plt.title(f"{name}", fontsize=20)
                ax = plt.gca()
                ax.set_xlim(left=0)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=False))
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}k"))
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=False))
                ax.tick_params(axis='both', which='major', labelsize=14)

                plt.tight_layout()
                img_path = self.log_path + f"{task['Name']}/sample_{sample_num}/{name}.png"
                # os.makedirs(os.path.join(self.log_path, f"{task['Name']}/sample_{sample_num}/Reward"), exist_ok=True)
                plt.savefig(img_path)
                curve_img_path.append(img_path)
        
        # Encode reward curve images
        encoded_image_list = []
        for img_path in curve_img_path:
            encoded_image_list.append(encode_image(img_path))
        
        # Get advice from GPT
        gpt_answer = gpt_interaction_image(self.client, GPT_VLM_MODEL, advice_system, advice_user, encoded_image_list)

        return gpt_answer

    def generate_rewards(self, curriculum, curriculum_idx, sample_num, reward_code_history, reward_samples, failure_reasons):
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

        # Add failure reward function samples and reasons
        if sample_num > 0:
            reward_user += "\nHere are some reward function samples tried for the current task, along with the reasons why the task failed.\n"
            for i in range(sample_num):
                reward_user += f"\nReward Sample {i+1}:"
                reward_user += reward_samples[i] + "\n"
                reward_user += f"Failure reason:\n"
                reward_user += failure_reasons[i] + "\n"

        # Get reward function from GPT
        reward_answer = gpt_interaction(self.client, GPT_LLM_MODEL, reward_system, reward_user)

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

    def refine_reward(self, curriculum, curriculum_idx, sample_num, failed_reward, advice):
        refine_system = file_to_string(self.prompt_path + "/refine_reward_system.txt")
        refine_user = file_to_string(self.prompt_path + "/refine_reward_user.txt")

        # Add current task
        task = curriculum[curriculum_idx]
        current_task_string = f'{task["Task_num"]}_{task["Subtask"]}\n{task["Description"]}\n{task["Reason"]}'
        refine_user = refine_user.replace("<<Current_Task>>", current_task_string)

        # Add former task
        former_task_string = ""
        if curriculum_idx > 0:
            for i in range(curriculum_idx):
                task_details = curriculum[i]
                former_task_string += f'{task_details["Task_num"]}_{task_details["Subtask"]}\n{task_details["Description"]}\n{task_details["Reason"]}\n'
        else:
            former_task_string = "No previous task learned."
        refine_user = refine_user.replace("<<Former_Tasks>>", former_task_string)

        # Add failed reward function
        refine_user = refine_user.replace("<<Reward_Function>>", failed_reward)

        # Add advice
        refine_user = refine_user.replace("<<Advice>>", advice)

        # Save the user prompt for reward refinement
        os.makedirs(self.log_path + f"{task['Name']}/sample_{sample_num}/", exist_ok=True)
        with open(self.log_path + f"{task['Name']}/sample_{sample_num}/" + "refine_user.md", "w") as file:
            file.write(refine_user)
        
        # Get refined reward function from GPT
        reward_answer = gpt_interaction(self.client, GPT_LLM_MODEL, refine_system, refine_user)
        
        # Save GPT answer
        with open(self.log_path + f"{task['Name']}/sample_{sample_num}/" + "refine_answer.md", "w") as file:
            file.write(reward_answer)

        pattern = r"`python(.*?)`"
        reward_match = re.search(pattern, reward_answer, re.DOTALL)
        if reward_match:
            reward_code_block = reward_match.group(1)
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
                    if type(traj_dict[key]) == str:  # if the value is a string, write it directly
                        feedback_statistics += f"{traj_dict[key]}\n"
                    else:
                        feedback_statistics += np.array2string(
                            traj_dict[key],
                            formatter={'float_kind': lambda x: f"{x:.3f}"}) + "\n"
                
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

        gpt_answer = gpt_interaction(self.client, GPT_LLM_MODEL, feedback_system, feedback_user)

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
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith("Task"):
            details["Task_num"] = line.split(" ")[1].strip()
        elif line.startswith("Name:"):
            details["Subtask"] = line.split(": ")[1].strip()
            details["Name"] = f'{details["Task_num"]}_{details["Subtask"]}({datetime})'
        elif line.startswith("Description:"):
            # Handle multi-line description
            description_lines = []
            if ":" in line:
                # Get text after the colon on the same line
                after_colon = ": ".join(line.split(": ")[1:]).strip()
                if after_colon:
                    description_lines.append(after_colon)
            
            # Continue reading subsequent lines until we hit "Reason:" or another section
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith("Reason:") or next_line.startswith("Task") or next_line.startswith("Name:"):
                    i -= 1  # Step back so the outer loop can process this line
                    break
                if next_line:  # Only add non-empty lines
                    description_lines.append(next_line)
                i += 1
            
            details["Description"] = "\n".join(description_lines).strip()
            
        elif line.startswith("Reason:"):
            # Handle multi-line reason
            reason_lines = []
            if ":" in line:
                # Get text after the colon on the same line
                after_colon = ": ".join(line.split(": ")[1:]).strip()
                if after_colon:
                    reason_lines.append(after_colon)
            
            # Continue reading subsequent lines until we hit the next section or end
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith("Task") or next_line.startswith("Name:") or next_line.startswith("Description:"):
                    i -= 1  # Step back so the outer loop can process this line
                    break
                if next_line:  # Only add non-empty lines
                    reason_lines.append(next_line)
                i += 1
            
            details["Reason"] = "\n".join(reason_lines).strip()
        
        i += 1
    
    return details
