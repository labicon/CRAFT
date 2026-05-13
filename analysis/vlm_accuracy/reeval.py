import argparse
import base64
import glob
import os
import pickle
import re
import subprocess
import sys
import traceback

# Ensure repo root is on the path when the script is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from curriculum.gpt.curriculum_api import CurriculumAPI

CRAFT_REEVAL_ROOT = "CRAFT_re-eval"


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_samples(task: str, run_filter: str = None):
    """
    Yield (run_ts, run_dir, task_dir_name, sample_dir) for every sample that
    has model/module.pt inside CRAFT_re-eval/{task}/.
    """
    task_root = os.path.join(CRAFT_REEVAL_ROOT, task)
    if not os.path.isdir(task_root):
        print(f"Warning: {task_root} does not exist")
        return

    timestamp_re = re.compile(r"^\d{2}-\d{2}_\d{2}-\d{2}")

    for run_ts in sorted(os.listdir(task_root)):
        run_dir = os.path.join(task_root, run_ts)
        if not os.path.isdir(run_dir):
            continue
        if not timestamp_re.match(run_ts):
            continue
        if run_filter and run_filter not in run_ts:
            continue

        for task_dir_name in sorted(os.listdir(run_dir)):
            task_dir = os.path.join(run_dir, task_dir_name)
            if not os.path.isdir(task_dir) or not task_dir_name[0].isdigit():
                continue

            for entry in sorted(os.listdir(task_dir)):
                if not entry.startswith("sample_"):
                    continue
                sample_dir = os.path.join(task_dir, entry)
                if not os.path.isdir(sample_dir):
                    continue
                if os.path.isfile(os.path.join(sample_dir, "model", "module.pt")):
                    yield run_ts, run_dir, task_dir_name, sample_dir


# ── rollout ───────────────────────────────────────────────────────────────────

def run_rollout_subprocess(task: str, model_dir: str, seed: int):
    subprocess.run(
        [
            sys.executable, "-m", "openrl_ws.curriculum_eval",
            "--task", task,
            "--load_dir", model_dir,
            "--seed", str(seed),
        ],
        check=True,
    )


# ── snapshot loading ──────────────────────────────────────────────────────────

def load_snapshots(model_dir: str):
    pattern = os.path.join(model_dir, "images", "snapshot_*.png")
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    snapshots = []
    for path in files:
        with open(path, "rb") as f:
            snapshots.append(base64.b64encode(f.read()).decode("utf-8"))
    return snapshots


# ── VLM evaluation ────────────────────────────────────────────────────────────

def run_vlm_eval(
    task: str,
    run_dir: str,
    curriculum: list,
    curriculum_idx: int,
    sample_num: int,
    snapshots: list,
    traj_dict: dict,
):
    prompt_path = f"./curriculum/gpt/prompts/{task}"
    api = CurriculumAPI(prompt_path, log_path=run_dir + "/", line_num=0)
    result = api.evaluate_result(curriculum, curriculum_idx, sample_num, snapshots, traj_dict)
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Re-run rollout and VLM evaluation for CRAFT_re-eval")
    parser.add_argument("--task", required=True, choices=["go2gate", "go2seesaw"])
    parser.add_argument("--run", default=None, help="Substring filter on run timestamp")
    parser.add_argument("--seed", type=int, default=0, help="Rollout seed")
    parser.add_argument("--dry_run", action="store_true", help="Print what would run without executing")
    args = parser.parse_args()

    samples = list(discover_samples(args.task, run_filter=args.run))
    if not samples:
        print(f"No samples found in {CRAFT_REEVAL_ROOT}/{args.task}/")
        return

    print(f"Found {len(samples)} sample(s) to re-evaluate (task={args.task})\n")

    ok = 0
    errors = []

    for run_ts, run_dir, task_dir_name, sample_dir in samples:
        sample_name = os.path.basename(sample_dir)
        label = f"[{run_ts}] {task_dir_name} / {sample_name}"

        if args.dry_run:
            print(f"  {label}")
            continue

        model_dir = os.path.join(sample_dir, "model")
        curriculum_pkl = os.path.join(run_dir, "curriculum.pkl")

        try:
            curriculum = pickle.load(open(curriculum_pkl, "rb"))
        except Exception as exc:
            msg = f"Failed to load curriculum.pkl: {exc}"
            print(f"  ERROR {label}: {msg}")
            errors.append((label, msg))
            continue

        curriculum_idx = int(task_dir_name.split("_")[0]) - 1
        sample_num = int(sample_name.split("_")[1])

        try:
            # 1. Run rollout via subprocess — saves traj/rew/video/images to model_dir
            print(f"  {label} — running rollout...", flush=True)
            run_rollout_subprocess(args.task, model_dir, seed=args.seed)

            # 2. Load fresh trajectory and snapshots
            with open(os.path.join(model_dir, "traj_dict.pkl"), "rb") as f:
                traj_dict = pickle.load(f)
            snapshots = load_snapshots(model_dir)
            if not snapshots:
                raise RuntimeError("No snapshot images found after rollout.")

            # 3. VLM evaluation
            print(f"  {label} — running VLM eval...", flush=True)
            result = run_vlm_eval(
                args.task, run_dir, curriculum,
                curriculum_idx, sample_num, snapshots, traj_dict,
            )

            decision = "Success" if result.get("decision") else "Failure"
            print(f"  {label} — done  [{decision}]")
            ok += 1

        except Exception as exc:
            msg = traceback.format_exc()
            print(f"  ERROR {label}:\n{msg}")
            errors.append((label, str(exc)))
            error_path = os.path.join(sample_dir, "reeval_error.txt")
            with open(error_path, "w") as f:
                f.write(msg)

    if not args.dry_run:
        print(f"\nDone: {ok}/{len(samples)} succeeded, {len(errors)} error(s).")
        if errors:
            print("Errors:")
            for label, msg in errors:
                print(f"  {label}: {msg.splitlines()[0]}")


if __name__ == "__main__":
    main()
