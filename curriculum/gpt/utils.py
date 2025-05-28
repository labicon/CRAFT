from openai import OpenAI
import yaml
import os
import re
import sys

GPT_KEY_PATH = "/home/kang/MANavigation/curriculum/gpt/key.yaml"


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def get_client():
    with open(GPT_KEY_PATH, "r") as stream:
        config = yaml.safe_load(stream)

    client = OpenAI(api_key=config["OPENAI_API_KEY"])

    return client


def gpt_interaction(client, gpt_model, system_string, user_string):
    trial = 0
    completion = None

    while completion is None and trial < 5:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[{"role": "system", "content": system_string}, {"role": "user", "content": user_string}],
        )
        trial += 1

    # print("GPT System Input: ", system_string)
    # print("GPT User Input: ", user_string)
    print(completion.choices[0].message.content)

    return completion.choices[0].message.content


def save_string_to_file(save_path, string_file):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        file.write(string_file)

def retrive_curriculum(log_path):
    with open(log_path + "/curriculum.md", "r") as file:
        return file.read()
    
    # Split the string into individual task sections
    task_sections = re.split(r"\n\n(?=Task)", tasks_string)

    # Function to extract details from each task section
    def extract_task_details(task_section):

        details = {}
        lines = task_section.split("\n")
        for line in lines:
            if line.startswith("Task"):
                details["Task"] = line.split(" ")[1]
            elif line.startswith("Name:"):
                details["Name"] = line.split(": ")[1]
            elif line.startswith("Description:"):
                details["Description"] = ": ".join(line.split(": ")[1:])
            elif line.startswith("Reason:"):
                details["Reason"] = ": ".join(line.split(": ")[1:])
        return details
    
    # Extract details for all tasks
    tasks_details = [extract_task_details(section) for section in task_sections]

    return tasks_details

def retrive_subtask_rewards(log_path, curriculum_idx):
    curriculum = retrive_curriculum(log_path)
    reward_code_history = []

    for i in range(curriculum_idx):
        subtask = curriculum[i]
        reward_code = file_to_string(os.path.join(log_path, subtask['Name'], f"best_reward_code.txt"))
        reward_code_history.append(reward_code)

    return reward_code_history

def retrive_evaluation_decision(log_path, curriculum_idx):
    curriculum = retrive_curriculum(log_path)
    evaluation_decision = []

    for i in range(curriculum_idx):
        subtask = curriculum[i]
        evaluation_string = file_to_string(os.path.join(log_path, f"{subtask['Name']}.md"))
        decision = evaluation_string.split("\n")[0]
        number = re.findall(r'\d+', decision)
        if number:
            evaluation_decision.append(int(number[0]))
        else:
            evaluation_decision.append(0)

    return evaluation_decision