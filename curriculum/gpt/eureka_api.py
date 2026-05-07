import os
import re
import textwrap
from typing import Dict, List, Optional

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from curriculum.gpt.utils import file_to_string, get_client, save_string_to_file


GPT_LLM_MODEL = "gpt-4o-2024-08-06"


class EurekaAPI:
    def __init__(self, prompt_path, log_path, line_num, model=GPT_LLM_MODEL):
        self.client = get_client()
        self.prompt_path = prompt_path
        self.eureka_prompt_path = os.path.join(prompt_path, "eureka")
        self.log_path = log_path
        self.insert_line_num = line_num
        self.model = model

    def generate_candidate(
        self,
        task_name: str,
        iteration: int,
        candidate_idx: int,
        reflection: Optional[str] = None,
        best_reward_code: Optional[str] = None,
    ) -> Dict[str, str]:
        system_prompt = file_to_string(os.path.join(self.eureka_prompt_path, "reward_system.txt"))
        user_prompt = file_to_string(os.path.join(self.eureka_prompt_path, "reward_user.txt"))
        env_description = file_to_string(os.path.join(self.prompt_path, "environment_description.txt"))
        reward_context = file_to_string(os.path.join(self.prompt_path, "reward_user.txt"))
        reward_context = reward_context.replace("<<Environment_Description>>", env_description)
        user_prompt = user_prompt.replace("<<Environment_Description>>", env_description)
        user_prompt = user_prompt.replace("<<Reward_Context>>", reward_context)
        user_prompt = user_prompt.replace("<<Task_Name>>", task_name)
        user_prompt = user_prompt.replace("<<Iteration>>", str(iteration))
        user_prompt = user_prompt.replace("<<Candidate_Index>>", str(candidate_idx))

        if best_reward_code and reflection:
            reflection_prompt = file_to_string(os.path.join(self.eureka_prompt_path, "reflection_user.txt"))
            reflection_prompt = reflection_prompt.replace("<<Best_Reward_Code>>", best_reward_code)
            reflection_prompt = reflection_prompt.replace("<<Reward_Reflection>>", reflection)
            user_prompt = user_prompt + "\n\n" + reflection_prompt

        answer = self._chat(system_prompt, user_prompt)
        reward_code = self.extract_reward_code(answer)
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "answer": answer,
            "reward_code": reward_code,
        }

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        completion = None
        for _ in range(5):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if completion is not None:
                break
        if completion is None:
            raise RuntimeError("OpenAI completion failed after 5 attempts.")
        content = completion.choices[0].message.content
        print(content)
        return content

    @staticmethod
    def extract_reward_code(answer: str) -> str:
        patterns = [
            r"```python\s*(.*?)```",
            r"```\s*(.*?)```",
            r"`python\s*(.*?)`",
        ]
        for pattern in patterns:
            match = re.search(pattern, answer, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if "def gpt_reward" in code:
                    return code
        if "def gpt_reward" in answer:
            return answer[answer.index("def gpt_reward"):].strip()
        raise ValueError("No reward code block containing gpt_reward was found.")

    def update_env_code(self, reward_code_path, reward_code, save_path=None):
        if save_path:
            save_string_to_file(save_path, reward_code)

        new_code_path = reward_code_path.replace("_source.py", ".py")
        with open(reward_code_path, "r") as file:
            lines = file.readlines()

        indented_text = textwrap.indent(reward_code, "    ")
        lines.insert(self.insert_line_num - 1, indented_text + "\n")

        with open(new_code_path, "w") as file:
            file.writelines(lines)

        print(f"Updated reward code saved to {new_code_path}")
        return new_code_path

    @staticmethod
    def load_tensorboard_scalars(event_dir: str) -> Dict[str, List[List[float]]]:
        if not os.path.isdir(event_dir):
            return {}

        event_files = []
        for root, _, files in os.walk(event_dir):
            for filename in files:
                if filename.startswith("events.out.tfevents."):
                    event_files.append(os.path.join(root, filename))
        if not event_files:
            return {}

        event_file = max(event_files, key=os.path.getmtime)
        accumulator = EventAccumulator(event_file)
        accumulator.Reload()

        scalars = {}
        for tag in accumulator.Tags().get("scalars", []):
            values = []
            for scalar in accumulator.Scalars(tag):
                values.append([scalar.step, scalar.value])
            scalars[tag] = values
        return scalars

    @staticmethod
    def summarize_scalars(scalars: Dict[str, List[List[float]]]) -> Dict[str, Dict[str, float]]:
        summary = {}
        for tag, points in scalars.items():
            if not points:
                continue
            values = np.array([point[1] for point in points], dtype=np.float32)
            summary[tag] = {
                "first": float(values[0]),
                "last": float(values[-1]),
                "max": float(np.max(values)),
                "min": float(np.min(values)),
                "mean": float(np.mean(values)),
            }
        return summary

    @staticmethod
    def build_reflection(best_candidate: Dict, candidates: List[Dict]) -> str:
        lines = [
            "Use the following training feedback to improve the reward function.",
            f"Best candidate: iteration {best_candidate['iteration']}, candidate {best_candidate['candidate_idx']}, score {best_candidate['score']:.6f}.",
            "",
            "Candidate summaries:",
        ]
        for candidate in candidates:
            status = "failed" if candidate.get("error") else "trained"
            lines.append(
                f"- iteration {candidate['iteration']} candidate {candidate['candidate_idx']}: "
                f"{status}, score {candidate.get('score', float('-inf')):.6f}"
            )
            if candidate.get("error"):
                lines.append(f"  error: {candidate['error'][:800]}")
                continue
            scalar_summary = candidate.get("scalar_summary", {})
            for tag in sorted(scalar_summary):
                if tag.startswith("Eval/") or tag.startswith("Reward/") or tag == "average step reward":
                    stats = scalar_summary[tag]
                    lines.append(
                        f"  {tag}: first={stats['first']:.4f}, last={stats['last']:.4f}, "
                        f"max={stats['max']:.4f}, min={stats['min']:.4f}, mean={stats['mean']:.4f}"
                    )
        lines.extend([
            "",
            "Revise the reward to improve the objective metrics, remove terms that create high reward without task progress, and keep useful shaping terms from the best candidate.",
        ])
        return "\n".join(lines)
