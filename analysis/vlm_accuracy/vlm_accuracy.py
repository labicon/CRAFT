import argparse
import os
import pickle
import re

CRAFT_RUNS_ROOT = "/home/kanghyun/mqe-curriculum/logs/CRAFT_updated"

# ── success rules ─────────────────────────────────────────────────────────────

def manual_success(traj: dict, task: str) -> bool:
    if task == "go2gate":
        gate_x = float(traj["gate_pos"][0])
        return (
            float(traj["agent_0_pos"][-1, 0]) > gate_x
            and float(traj["agent_1_pos"][-1, 0]) > gate_x
        )
    if task == "go2seesaw":
        def agent_ok(pos):
            return bool(((pos[:, 0] >= 7.0) & (pos[:, 2] >= 1.3)).any())
        return agent_ok(traj["agent_0_pos"]) or agent_ok(traj["agent_1_pos"])
    raise ValueError(f"Unknown task: {task}")


# ── file loading ──────────────────────────────────────────────────────────────

def parse_vlm_decision(sample_dir: str) -> bool:
    path = os.path.join(sample_dir, "evaluation_answer.md")
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.lower().startswith("decision:"):
                return "success" in stripped.lower()
    return False


def load_traj(sample_dir: str) -> dict:
    path = os.path.join(sample_dir, "model", "traj_dict.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_samples(task: str, run_filter: str = None, curriculum_only: bool = False):
    """
    Yield (run_ts, task_dir_name, sample_dir) for every sample that has both
    evaluation_answer.md and model/traj_dict.pkl.
    """
    task_root = os.path.join(CRAFT_RUNS_ROOT, task)
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
        if curriculum_only and "no_refine" in run_ts:
            continue
        if run_filter and run_filter not in run_ts:
            continue

        # Collect numbered task dirs and keep only the final (highest-numbered) one.
        task_dirs = [
            name for name in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, name)) and name[0].isdigit()
        ]
        if not task_dirs:
            continue
        final_task_dir_name = max(task_dirs, key=lambda n: int(n.split("_")[0]))
        task_dir_name = final_task_dir_name
        task_dir = os.path.join(run_dir, task_dir_name)

        for entry in sorted(os.listdir(task_dir)):
            if not entry.startswith("sample_"):
                continue
            sample_dir = os.path.join(task_dir, entry)
            if not os.path.isdir(sample_dir):
                continue
            has_eval = os.path.isfile(os.path.join(sample_dir, "evaluation_answer.md"))
            has_traj = os.path.isfile(os.path.join(sample_dir, "model", "traj_dict.pkl"))
            if has_eval and has_traj:
                yield run_ts, task_dir_name, sample_dir


# ── reporting ─────────────────────────────────────────────────────────────────

def _label(flag: bool) -> str:
    return "Success" if flag else "Failure"


def print_table(records: list):
    col_run  = max(len(r["run"])      for r in records)
    col_task = max(len(r["task_dir"]) for r in records)
    col_run  = max(col_run,  3)
    col_task = max(col_task, 7)

    header = (
        f"  {'Run':<{col_run}}  {'Task':<{col_task}}  "
        f"{'Samp':<5}  {'Manual':<8}  {'VLM':<8}  Match"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in records:
        match_str = "   ok" if r["match"] else " ***"
        print(
            f"  {r['run']:<{col_run}}  {r['task_dir']:<{col_task}}  "
            f"{r['sample']:<5}  {_label(r['manual']):<8}  "
            f"{_label(r['vlm']):<8}  {match_str}"
        )


def print_summary(records: list):
    total = len(records)
    if total == 0:
        print("No records.")
        return

    tp = sum(1 for r in records if     r["vlm"] and     r["manual"])
    tn = sum(1 for r in records if not r["vlm"] and not r["manual"])
    fp = sum(1 for r in records if     r["vlm"] and not r["manual"])
    fn = sum(1 for r in records if not r["vlm"] and     r["manual"])
    accuracy = (tp + tn) / total * 100

    print(f"\n{'─'*50}")
    print(f"  Total samples : {total}")
    print(f"  Accuracy      : {accuracy:.1f}%  ({tp + tn}/{total} correct)")
    print(f"\n  Confusion matrix (VLM rows × Manual cols):")
    print(f"                  Manual-S  Manual-F")
    print(f"    VLM-Success :  {tp:>6}    {fp:>6}   (FP={fp})")
    print(f"    VLM-Failure :  {fn:>6}    {tn:>6}   (FN={fn})")

    # per-run breakdown
    runs = sorted({r["run"] for r in records})
    if len(runs) > 1:
        print(f"\n  Per-run accuracy:")
        for run in runs:
            run_recs = [r for r in records if r["run"] == run]
            n = len(run_recs)
            correct = sum(1 for r in run_recs if r["match"])
            print(f"    {run:<22}  {correct}/{n}  ({correct/n*100:.0f}%)")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare VLM decisions against manual success metrics")
    parser.add_argument("--task", required=True, choices=["go2gate", "go2seesaw"])
    parser.add_argument("--run", default=None,
                        help="Substring filter on run timestamp (e.g. '08-02_12-13')")
    parser.add_argument("--curriculum_only", action="store_true",
                        help="Exclude no_refine runs")
    args = parser.parse_args()

    records = []
    errors = []

    for run_ts, task_dir_name, sample_dir in discover_samples(
        args.task, run_filter=args.run, curriculum_only=args.curriculum_only
    ):
        sample_name = os.path.basename(sample_dir)
        try:
            traj = load_traj(sample_dir)
            m_success = manual_success(traj, args.task)
            v_success = parse_vlm_decision(sample_dir)
            records.append({
                "run": run_ts,
                "task_dir": task_dir_name,
                "sample": sample_name,
                "manual": m_success,
                "vlm": v_success,
                "match": m_success == v_success,
            })
        except Exception as exc:
            errors.append(f"  {sample_dir}: {exc}")

    if not records:
        print("No samples found.")
        return

    print(f"\nVLM accuracy check — task: {args.task}  ({len(records)} samples)\n")
    print_table(records)
    print_summary(records)

    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(e)


if __name__ == "__main__":
    main()
