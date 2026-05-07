# Repository Guidelines

## Project Structure & Module Organization
- `curriculum/`: Core curriculum-learning code. Entry point `curriculum/main.py`, training modules in `curriculum/train/`, task configs in `curriculum/configs/`, and prompt templates in `curriculum/gpt/prompts/<task>/`.
- `mqe/`: Multi-agent Quadruped Environment (editable install required).
- `openrl_ws/`: OpenRL training/evaluation scripts and utilities.
- `resources/`: Robot and object assets (URDFs/meshes).
- `logs/`: Run outputs (created at runtime, typically `logs/<timestamp>/`).
- `run_experiments.sh`: Convenience runner for multi-trial experiments.

## Build, Test, and Development Commands
- `pip install -e mqe`: Install MQE in editable mode for local development.
- `pip install -r requires.txt`: Install additional Python dependencies.
- `export LD_LIBRARY_PATH=/path/to/conda/envs/mqe/lib`: Required for Isaac Gym runtime.
- `export PYTHONPATH=$PWD`: Required when running from repo root.
- `./run_experiments.sh`: Run automated curriculum (default, 5 runs).
- `./run_experiments.sh -m`: Run manual/baseline curriculum.
- `python curriculum/main.py --task go2gate --module curriculum --seed 0`: Single run with explicit task/module.

## Coding Style & Naming Conventions
- Python 3.8 codebase; use 4-space indentation and snake_case for functions/variables.
- Task configs live in `curriculum/configs/<task>.yaml` and should align with prompt folders in `curriculum/gpt/prompts/<task>/`.
- Generated reward code is injected into wrapper templates ending in `_source.py` under `mqe/envs/wrappers/`.

## Testing Guidelines
- No dedicated test suite is present. Use evaluation scripts and training runs to validate changes.
- Typical validation flow: train via `curriculum/main.py`, then evaluate with `openrl_ws/curriculum_eval.py` (produces frames and `.pkl` data for VLM evaluation).

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and lowercase (e.g., `add pushbox with go2`, `fix typo`).
- Keep PRs focused. Include: a concise summary, affected tasks/configs, and any relevant logs or plots from `logs/<timestamp>/`.

## Configuration & Secrets
- Set the OpenAI API key in `curriculum/gpt/key.yaml` as:
  `OPENAI_API_KEY: "sk-..."`.
- Do not commit API keys or generated logs.
