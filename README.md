# MQE Curriculum: Automated Curriculum Learning for Multi-Agent Tasks

This repository implements automated curriculum learning for multi-agent quadruped tasks in the **Multi-agent Quadruped Environment (MQE)**.

## Prerequisites & Installation

This project is built on top of [Isaac Gym](https://developer.nvidia.com/isaac-gym) and [MQE](https://ziyanx02.github.io/multiagent-quadruped-environment/).

### 1. Basic Requirements
*   **OS:** ubuntu 20.04 (Recommended)
*   **Python:** 3.8

### 2. Install Isaac Gym
Download the Isaac Gym Preview 4 package from the [developer portal](https://developer.nvidia.com/isaac-gym) and follow the instructions:

```bash
tar -xf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
```

### 3. Clone and install MQE

First, to clone this repository, you will need git-lfs installed. Follow instructions from [here](https://git-lfs.com/). This is required to pretrained low-level locomotion policies for quadruped robots.

Then, clone this repository and navigate into it:

```bash
git clone --branch quadruped_navigation https://github.com/labicon/CRAFT.git
cd CRAFT
```

This repository assumes you have the MQE dependencies installed. You can install MQE in editable mode.

```bash
# Clone the MQE repo provided within this workspace. Original MQE repo does not support Unitree Go2.
# This curriculum assumes it runs alongside MQE structure.
cd mqe
pip install -e .
cd ..
```

### 4. Install Dependencies
Install additional dependencies required for this curriculum module:

```bash
pip install -r requires.txt
```

## Repository Structure

*   `curriculum/`: Main source code for the curriculum learning logic.
    *   `main.py`: Entry point for running experiments.
    *   `train/`: Contains the `Curriculum_Module` and `Manual_Module`.
    *   `configs/`: YAML configuration files for different tasks (`go2gate`, `go2seesaw`).
    *   `gpt/`: Prompts and OpenAI API integration for automated curriculum generation.
*   `mqe/`: The Multi-agent Quadruped Environment core.
*   `openrl_ws/`: Workstation scripts for OpenRL integration (evaluators, utils).
*   `utils/`: Utility scripts (plotting, etc.).

## Running Experiments

### Setup API keys

Set up your OpenAI API key as key.yaml file in the `curriculum/gpt` directory.

```yaml
OPENAI_API_KEY: "your_openai_api_key_here"
```

### Scripts

Set `LD_LIBRARY_PATH` to include your conda environment's library directory.
For the `mqe` environment:

```bash
# Replace with your actual path (e.g., ~/miniconda3/envs/mqe/lib)
export LD_LIBRARY_PATH=/path/to/conda/envs/mqe/lib
```

We provide a unified bash script `run_experiments.sh` to facilitate running multiple trials of experiments efficiently.
This `run_experiments.sh` script supports running both the automated curriculum and manual baselines.

**1. Run Curriculum Learning (Default):**
Runs the automated curriculum logic.
```bash
./run_experiments.sh
```

**2. Run Manual/Baseline:**
Runs the manual curriculum or standard training baseline.
```bash
./run_experiments.sh -m
```

### Running Manually via Python

You can also run the python scripts directly for more granular control over arguments like tasks (`go2gate`, `go2seesaw`) or modules.

**Command Structure:**
```bash
python curriculum/main.py --task [TASK_NAME] --module [MODULE_TYPE]
```

**Arguments:**
*   `--task`: Select the environment task. Choices: `go2gate` (default), `go2seesaw`.
*   `--module`: Select the training module. Choices: `curriculum` (default), `manual`.
*   `--seed`: Random seed (default: 0).
*   `--logdir`: Directory to save logs (default: `logs`).

**Examples:**

*   **Train Go2Gate with Automated Curriculum:**
    ```bash
    python curriculum/main.py --task go2gate --module curriculum
    ```

*   **Train Go2Seesaw with Manual Curriculum Baseline:**
    ```bash
    python curriculum/main.py --task go2seesaw --module manual
    ```

