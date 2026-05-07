# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CRAFT is an automated curriculum learning system for multi-agent quadruped reinforcement learning. It uses LLMs (GPT-4o) to generate task curricula and reward functions, and VLMs (o4-mini) to evaluate training progress via visual feedback. Robots are Unitree Go1/Go2 quadrupeds simulated in NVIDIA Isaac Gym.

## Setup & Environment

```bash
# Install MQE in editable mode
cd mqe && pip install -e . && cd ..

# Install additional dependencies
pip install -r requires.txt

# Required runtime env var (adjust path to your conda env)
export LD_LIBRARY_PATH=/path/to/conda/envs/mqe/lib

# OpenAI API key must be in curriculum/gpt/key.yaml
# Format: OPENAI_API_KEY: "sk-..."
```

**Python 3.8** on Ubuntu 20.04. Requires Isaac Gym Preview 4 and git-lfs for pretrained locomotion policies.

## Running Experiments

```bash
# Set PYTHONPATH (required when running from repo root)
export PYTHONPATH=$PWD

# Automated curriculum (5 runs)
./run_experiments.sh

# Manual/baseline (5 runs)
./run_experiments.sh -m

# Single run with specific task
python curriculum/main.py --task go2pushbox --module curriculum --seed 0
python curriculum/main.py --task go2seesaw --module manual --seed 1
```

**Arguments:** `--task` (go2gate|go2seesaw|go2pushbox), `--module` (curriculum|manual), `--seed`, `--logdir` (default: logs/).

## Architecture

### Training Pipeline

`curriculum/main.py` is the entry point. It loads a task config from `curriculum/configs/<task>.yaml` and instantiates either `Curriculum_Module` or `Manual_Module`.

**Automated curriculum flow** (`curriculum/train/curriculum_train.py`):
1. `CurriculumAPI.generate_curriculum()` — LLM generates 3 candidate curricula, then refines to 1
2. For each subtask: generate reward function → train via OpenRL (PPO) → evaluate
3. On failure: VLM provides visual feedback + advice → LLM refines reward (up to 10 attempts, 3 reward samples + 2 feedback samples per attempt)
4. On success: save best model, load it as initialization for next subtask

### Key Modules

- **`curriculum/gpt/curriculum_api.py`** — `CurriculumAPI` class: all LLM/VLM interactions (curriculum generation, reward code generation/refinement, visual evaluation). Uses models defined at top: `GPT_LLM_MODEL` and `GPT_VLM_MODEL`.
- **`curriculum/gpt/utils.py`** — OpenAI client setup, GPT interaction helpers, image encoding. API key loaded from `curriculum/gpt/key.yaml`.
- **`curriculum/gpt/prompts/<task>/`** — Task-specific prompt templates with `<<Placeholder>>` substitution.
- **`mqe/envs/wrappers/`** — Environment wrappers defining observations, rewards, and step logic. Files ending in `_source.py` are templates; the system injects GPT-generated reward code at a specific line number (configured via `line_num` in the task YAML).
- **`openrl_ws/curriculum_train.py`** — PPO training loop using OpenRL. 500 parallel envs, checkpoints every 5000 steps, WandB + TensorBoard logging.
- **`openrl_ws/curriculum_eval.py`** — Runs trained policy, records video frames and trajectory data (`.pkl` files) for VLM evaluation.

### Reward Code Injection

The `CurriculumAPI.update_env_code()` method takes a `_source.py` wrapper template, generates a reward function via GPT, and injects it at the configured line number. The generated code defines a `gpt_reward(global_state, action)` function inside the wrapper. The line number is set per-task in `curriculum/configs/<task>.yaml`.

### Task Configuration

Each task YAML in `curriculum/configs/` specifies:
- `env_path`: path to the `_source.py` wrapper template
- `num_reward_samples` / `num_reward_feedback_samples`: how many reward variants to try
- `iter_per_task`: training iterations per subtask
- `line_num`: insertion point for generated reward code

### Output Structure

Logs go to `logs/<timestamp>/` containing the generated curriculum, per-subtask reward code samples, trajectory data, evaluation images, and training checkpoints.

## Branch Info

Active development is on the `quadruped_navigation` branch. Main branch is `main`.
