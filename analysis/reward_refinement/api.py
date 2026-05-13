import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from curriculum.gpt.curriculum_api import CurriculumAPI
from curriculum.gpt.utils import encode_image, file_to_string, gpt_interaction_image

GPT_VLM_MODEL = "gpt-o4-mini-2025-04-16"


class VLMRefinementAPI(CurriculumAPI):
    """CurriculumAPI subclass for VLM-guided refinement experiments.

    The only difference from the parent is that get_advice looks for TensorBoard
    events under MQE-Eureka (used by eureka_train) rather than MQE-Curriculum.
    The caller sets the exact event directory via set_event_dir() before calling
    get_advice(), which avoids any glob ambiguity across multiple runs.
    """

    def __init__(self, prompt_path, log_path, line_num):
        super().__init__(prompt_path, log_path, line_num)
        self._event_dir = None

    def set_event_dir(self, event_dir: str):
        self._event_dir = event_dir

    def get_advice(self, curriculum, curriculum_idx, sample_num, failed_reward, failure_reason):
        advice_system = file_to_string(self.prompt_path + "/advice_system.txt")
        advice_user = file_to_string(self.prompt_path + "/advice_user.txt")

        task = curriculum[curriculum_idx]
        advice_user = advice_user.replace("<<Task_Name>>", f'{task["Task_num"]}_{task["Subtask"]}')
        advice_user = advice_user.replace("<<Task_Description>>", task["Description"])
        advice_user = advice_user.replace("<<Task_Reason>>", task["Reason"])
        advice_user = advice_user.replace("<<Reward_Function>>", failed_reward)
        advice_user = advice_user.replace("<<Failure_Reason>>", failure_reason)

        encoded_image_list = []

        if self._event_dir and os.path.isdir(self._event_dir):
            pattern = os.path.join(self._event_dir, "events.out.tfevents.*")
            event_files = glob.glob(pattern)
            if event_files:
                event_file = max(event_files, key=os.path.getmtime)
                ea = EventAccumulator(event_file)
                ea.Reload()

                sample_log_dir = os.path.join(
                    self.log_path, task["Name"], f"sample_{sample_num}"
                )
                os.makedirs(sample_log_dir, exist_ok=True)

                for tag in ea.Tags()["scalars"]:
                    if "Reward" not in tag:
                        continue
                    reward_curve = np.array(
                        [[s.step, s.value] for s in ea.Scalars(tag)]
                    )
                    name = tag[len("Reward/"):] if tag.startswith("Reward/") else tag

                    plt.figure(figsize=(5, 4))
                    plt.plot(reward_curve[:, 0], reward_curve[:, 1])
                    plt.xlabel("Step", fontsize=16)
                    plt.title(name, fontsize=20)
                    ax = plt.gca()
                    ax.set_xlim(left=0)
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=False))
                    ax.xaxis.set_major_formatter(
                        ticker.FuncFormatter(lambda x, pos: f"{int(x / 1000)}k")
                    )
                    ax.set_ylim(bottom=0)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=False))
                    ax.tick_params(axis="both", which="major", labelsize=14)
                    plt.tight_layout()

                    img_path = os.path.join(sample_log_dir, f"{name}.png")
                    plt.savefig(img_path)
                    plt.close()
                    encoded_image_list.append(encode_image(img_path))
            else:
                print(f"Warning: no TensorBoard event files found in {self._event_dir}")
        else:
            print(f"Warning: event_dir not set or missing — skipping reward curves")

        gpt_answer = gpt_interaction_image(
            self.client, GPT_VLM_MODEL, advice_system, advice_user, encoded_image_list
        )

        sample_log_dir = os.path.join(self.log_path, task["Name"], f"sample_{sample_num}")
        os.makedirs(sample_log_dir, exist_ok=True)
        with open(os.path.join(sample_log_dir, "advice_answer.md"), "w") as f:
            f.write(gpt_answer)

        return gpt_answer
