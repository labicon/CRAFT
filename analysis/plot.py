import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import re
import argparse
import matplotlib.ticker as ticker

# ==========================================
# Configuration and Directories
# ==========================================

CONFIGS = {
    "go2gate": {
        "curriculum_directories": [
            "CRAFT_runs/go2gate/08-02_12-13",
            "CRAFT_runs/go2gate/08-02_19-44",
            "CRAFT_runs/go2gate/08-04_17-21",
        ],
        "no_refine_directories": [
            "CRAFT_runs/go2gate/08-16_05-14_no_refine_2",
            "CRAFT_runs/go2gate/08-16_17-09_no_refine_3",
            "CRAFT_runs/go2gate/08-17_15-09_no_refine_5",
        ],
        "scratch_directories": [
            "baselines/checkpoints/go2gate/go2gate_08-02_12-13_scratch",
            "baselines/checkpoints/go2gate/go2gate_08-02_19-44_scratch",
            "baselines/checkpoints/go2gate/go2gate_08-04_17-21_scratch",
        ],
        "example_directories": [
            "baselines/checkpoints/go2gate/go2gate_example_reward_1",
            "baselines/checkpoints/go2gate/go2gate_example_reward_2",
            "baselines/checkpoints/go2gate/go2gate_example_reward_3",
            "baselines/checkpoints/go2gate/go2gate_example_reward_4",
            "baselines/checkpoints/go2gate/go2gate_example_reward_5",
        ],
        "mqe_directories": [
            "baselines/checkpoints/go2gate/go2gate_mqe_reward_1",
            "baselines/checkpoints/go2gate/go2gate_mqe_reward_2",
            "baselines/checkpoints/go2gate/go2gate_mqe_reward_3",
            "baselines/checkpoints/go2gate/go2gate_mqe_reward_4",
            "baselines/checkpoints/go2gate/go2gate_mqe_reward_5",
        ],
        "x_scale": 2500000,
        "metrics": {
            "avg_min_dist": "average_minimum_distance",
            "std_min_dist": "std_minimum_distance",
            "avg_x_traversed": "average_x_traversed",
            "std_x_traversed": "std_x_traversed"
        },
        "plot_titles": {
            "avg_min_dist": "Average Minimum Distance to Goal",
            "avg_x_traversed": "Average X Traversed"
        },
        "y_labels": {
            "avg_min_dist": "Distance (m)",
            "avg_x_traversed": "Distance (m)"
        }
    },
    "go2seesaw": {
        "curriculum_directories": [
            "CRAFT_runs/go2seesaw/08-17_10-44",
            "CRAFT_runs/go2seesaw/08-18_09-25",
            "CRAFT_runs/go2seesaw/08-19_02-44",
        ],
        "no_refine_directories": [
            "CRAFT_runs/go2seesaw/09-01_03-39_no_refine",
            "CRAFT_runs/go2seesaw/09-02_17-02_no_refine",
            "CRAFT_runs/go2seesaw/09-03_11-53_no_refine",
        ],
        "scratch_directories": [
            "baselines/checkpoints/go2seesaw/go2seesaw_08-17_10-44_scratch",
            "baselines/checkpoints/go2seesaw/go2seesaw_08-18_09-25_scratch",
            "baselines/checkpoints/go2seesaw/go2seesaw_08-19_02-44_scratch",
        ],
        "example_directories": [
            "baselines/checkpoints/go2seesaw/go2seesaw_example_reward_0",
            "baselines/checkpoints/go2seesaw/go2seesaw_example_reward_1",
            "baselines/checkpoints/go2seesaw/go2seesaw_example_reward_2",
            "baselines/checkpoints/go2seesaw/go2seesaw_example_reward_3",
            "baselines/checkpoints/go2seesaw/go2seesaw_example_reward_4",
        ],
        "mqe_directories": [
            "baselines/checkpoints/go2seesaw/go2seesaw_mqe_reward_0",
            "baselines/checkpoints/go2seesaw/go2seesaw_mqe_reward_1",
            "baselines/checkpoints/go2seesaw/go2seesaw_mqe_reward_2",
            "baselines/checkpoints/go2seesaw/go2seesaw_mqe_reward_3",
            "baselines/checkpoints/go2seesaw/go2seesaw_mqe_reward_4",
        ],
        "x_scale": 2500000,
         "metrics": {
            "avg_max_height": "average_maximum_height",
            "std_max_height": "std_maximum_height",
            "avg_target_dist": "average_target_distance",
            "std_target_dist": "std_target_distance"
        },
        "plot_titles": {
            "avg_max_height": "Average Maximum Height",
            "avg_target_dist": "Average Target Distance"
        },
        "y_labels": {
            "avg_max_height": "Height (m)",
            "avg_target_dist": "Distance (m)"
        }
    },
    "lift": {
        "x_scale": 19200, 
        # Lift uses summary.pkl, so directories are placeholders or handled differently
        "from_pickle": True,
        "pickle_path": "summary.pkl", # Expect this file in current directory or updated path
         "metrics": {}, # Metrics are loaded from pickle
        "plot_titles": {},
        "y_labels": {}
    }
}

COLORS = {
    "curriculum": "#1f77b4",  # Blue
    "scratch": "#ff7f0e",     # Orange
    "no_refine": "#2ca02c",   # Green
    "example": "#d62728",     # Red
    "mqe": "#9467bd"          # Purple
}

LABELS = {
    "curriculum": "CRAFT",
    "scratch": "No Curriculum",
    "no_refine": "No Refine",
    "example": "Example Reward",
    "mqe": "Environment Reward"
}


# ==========================================
# Helper Functions
# ==========================================

def get_model_directories(base_dir):
    """
    Get all checkpoint directories with format rl_model_{stepsize}_steps from the given base directory.
    """
    model_dirs = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist")
        return model_dirs
    
    pattern = re.compile(r'^rl_model_(\d+)_steps$')
    
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                match = pattern.match(item)
                if match:
                    step_size = int(match.group(1))
                    model_dirs.append((step_size, item_path))
        
        model_dirs.sort(key=lambda x: x[0])
        return [path for _, path in model_dirs]
        
    except OSError as e:
        print(f"Error reading directory {base_dir}: {e}")
        return []

def lookup_optimal_dir(experiment_dir):
    """
    Look for optimal model directories in an experiment directory.
    """
    optimal_dirs = []
    
    if not os.path.exists(experiment_dir):
        print(f"Warning: Experiment directory {experiment_dir} does not exist")
        return optimal_dirs
    
    task_dirs = []
    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)
        if os.path.isdir(item_path) and item[0].isdigit():
            task_dirs.append((item, item_path))
    
    task_dirs.sort(key=lambda x: int(x[0].split('_')[0]))
    
    for task_name, task_path in task_dirs:
        decision_file = os.path.join(experiment_dir, f"{task_name}.md")
        
        if os.path.exists(decision_file):
            try:
                with open(decision_file, 'r') as f:
                    content = f.read()
                    
                decision_match = re.search(r'Decision:\s*Experiment\s*(\d+)', content, re.IGNORECASE)
                if decision_match:
                    experiment_num = int(decision_match.group(1))
                    optimal_sample_dir = os.path.join(task_path, f"sample_{experiment_num}")
                    
                    if os.path.exists(optimal_sample_dir):
                        model_dir = os.path.join(optimal_sample_dir, "model")
                        if os.path.exists(model_dir):
                            optimal_dirs.append(model_dir)
                        else:
                            print(f"Warning: Model directory not found in {optimal_sample_dir}")
                    else:
                        print(f"Warning: Sample directory {optimal_sample_dir} not found")
                else:
                    print(f"Warning: Could not parse decision from {decision_file}")
                    
            except Exception as e:
                print(f"Error reading decision file {decision_file}: {e}")
        
        if not any(task_name in path for path in optimal_dirs):
            sample_dirs = []
            try:
                for item in os.listdir(task_path):
                    item_path = os.path.join(task_path, item)
                    if os.path.isdir(item_path) and item.startswith("sample_"):
                        try:
                            sample_num = int(item.split("_")[1])
                            sample_dirs.append((sample_num, item_path))
                        except (IndexError, ValueError):
                            continue
                
                if sample_dirs:
                    sample_dirs.sort(key=lambda x: x[0])
                    highest_sample_num, highest_sample_dir = sample_dirs[-1]
                    
                    model_dir = os.path.join(highest_sample_dir, "model")
                    if os.path.exists(model_dir):
                        optimal_dirs.append(model_dir)
                    else:
                        print(f"Warning: Model directory not found in {highest_sample_dir}")
                else:
                    print(f"Warning: No sample directories found in {task_path}")
                    
            except OSError as e:
                print(f"Error listing directories in {task_path}: {e}")
    
    return optimal_dirs

def pad_curve_to_length(curve, target_length):
    if len(curve) == 0:
        return [0] * target_length
    if len(curve) >= target_length:
        return curve[:target_length]
    final_value = curve[-1]
    padded_curve = curve + [final_value] * (target_length - len(curve))
    return padded_curve

def load_eval(model_path):
    eval_results_path = os.path.join(model_path, "eval_results.pkl")
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'rb') as f:
            eval_results = pickle.load(f)
            return eval_results
    return None

def process_evaluation(directories, method_type="curriculum", extra_metrics=None):
    if extra_metrics is None:
        extra_metrics = {}
        
    summary = {
        "total_success": [],
        "partial_success": [],
        "average_reward": [],
        "std_reward": []
    }
    for m in extra_metrics.keys():
        summary[m] = []

    for path in directories:
        if method_type == "curriculum" or method_type == "no_refine":
             # Traverse optimal directories for curriculum
             base_directories = lookup_optimal_dir(path)
        else:
             # Direct path for scratch/baseline
             base_directories = [path]
             
        # Lists for this single run
        run_curves = {k: [] for k in summary.keys()}
        
        for base_dir in base_directories:
            model_dirs = get_model_directories(base_dir)
            if not model_dirs:
                # print(f"No model directories found in {base_dir}")
                continue
            
            for model_path in model_dirs:
                eval_results = load_eval(model_path)
                if eval_results is None:
                    continue
                
                run_curves["total_success"].append(eval_results['success_runs'] / eval_results['total_runs'] * 100 if eval_results['total_runs'] > 0 else 0)
                run_curves["partial_success"].append(eval_results['partial_success_runs'] / eval_results['total_runs'] * 100 if eval_results['total_runs'] > 0 else 0)
                run_curves["average_reward"].append(eval_results['average_reward'])
                run_curves["std_reward"].append(eval_results['std_reward'])
                
                for metric_key, eval_key in extra_metrics.items():
                    if eval_key in eval_results:
                        run_curves[metric_key].append(eval_results[eval_key])
                        
        # Append this run to summary
        if len(run_curves["total_success"]) > 0:
            for k in summary.keys():
                summary[k].append(run_curves[k])

    # Pad and convert to numpy
    result = {}
    
    # helper for finding max length across ALL metrics for consistency
    max_length = 0
    for key in summary:
        for curve in summary[key]:
            max_length = max(max_length, len(curve))
            
    print(f"processed {method_type}, max_length: {max_length}")

    for key in summary:
        padded_list = []
        for curve in summary[key]:
            padded_list.append(pad_curve_to_length(curve, max_length))
        result[key] = np.array(padded_list)
        
    return result

# ==========================================
# Plotting
# ==========================================

def plot_with_std(ax, x, mean_curve, std_curve, label, color):
    # Ensure x matches mean_curve length
    if len(x) != len(mean_curve):
        # Truncate or extend x
        x_plot = np.arange(len(mean_curve)) * (x[1] - x[0] if len(x) > 1 else 1)
        # Using x[1]-x[0] assumes step size from passed x. 
        # Better: resize x to match mean_curve
        step = x[1] - x[0] if len(x) > 1 else 1
        x_plot = np.arange(len(mean_curve)) * step
    else:
        x_plot = x

    ax.plot(x_plot, mean_curve, label=label, color=color, linewidth=3)
    ax.fill_between(x_plot, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.1)

def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--task", type=str, default="go2gate", choices=["go2gate", "go2seesaw", "lift"], help="Task to plot")
    parser.add_argument("--output_dir", type=str, default="figure", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    config = CONFIGS[args.task]
    
    # Setup plot style
    plt.rcParams["font.family"] = "DejaVu Sans" # Fallback
    try:
        plt.rcParams["font.family"] = "P052" # Original request
    except:
        pass
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Gather Data
    results = {}
    
    if args.task == "lift":
        print("Loading Lift data from pickle...")
        try:
             with open(config["pickle_path"], 'rb') as f:
                data = pickle.load(f)
                # Map keys
                results["curriculum"] = data.get('Ours')
                results["scratch"] = data.get('no_curriculum')
                results["no_refine"] = data.get('no_refinement')
                results["example"] = data.get('example_reward')
                results["mqe"] = data.get('env_reward')
                # Lift pickle has 'mean' and 'std' directly in it for success rate?
                # Based on notebook: curriculum_results['mean'], curriculum_results['std']
                # But generic structure usually keys are metrics. 
                # Let's assume the pickle structure matches what plot_with_std expects or contains 'mean'/'std' keys 
                # wrapper is needed if structure differs.
        except Exception as e:
            print(f"Failed to load pickle: {e}")
            return
    else:
        extra_metrics = config.get("metrics", {})
        
        print("Processing Curriculum...")
        results["curriculum"] = process_evaluation(config["curriculum_directories"], "curriculum", extra_metrics)
        print("Processing No Refine...")
        results["no_refine"] = process_evaluation(config["no_refine_directories"], "no_refine", extra_metrics)
        print("Processing Scratch...")
        results["scratch"] = process_evaluation(config["scratch_directories"], "scratch", extra_metrics)
        print("Processing Example...")
        results["example"] = process_evaluation(config["example_directories"], "example", extra_metrics)
        print("Processing MQE...")
        results["mqe"] = process_evaluation(config["mqe_directories"], "mqe", extra_metrics)
    
    # Define plotting helper
    def plot_metric(metric_key, title, ylabel, filename_suffix, use_percent=False):
        plt.figure(figsize=(12, 8))
        
        # Determine max length for X axis definition
        max_len = 0
        for method_key, data in results.items():
            if data is None: continue
            
            # Lift data might be formatted differently (dict with 'mean' code)
            if args.task == "lift":
                 if 'mean' in data:
                     max_len = max(max_len, len(data['mean']))
            else:
                # Regular data is dict of arrays of shape (n_runs, n_steps)
                if metric_key in data:
                    max_len = max(max_len, data[metric_key].shape[1])

        x = np.arange(max_len) * config["x_scale"]
        
        for method_key, label in LABELS.items():
            data = results.get(method_key)
            if data is None: continue
            color = COLORS[method_key]
            
            mean_curve = None
            std_curve = None
            
            if args.task == "lift":
                # Lift dict structure from notebook: data['mean'], data['std']
                # But is it for ONE metric (success rate)? 
                # Lift notebook code: plot_with_std(..., curriculum_results['mean'], ...)
                # It seems 'Ours' maps to success rate dict directly. 
                # If we want other metrics, lift pickle might not have them or have different keys.
                # Assuming lift pickle is ONLY success rate for now based on notebook code provided.
                if metric_key == "total_success":
                    mean_curve = data['mean']
                    std_curve = data['std']
            else:
                if metric_key in data and len(data[metric_key]) > 0:
                    arr = data[metric_key]
                    mean_curve = np.mean(arr, axis=0)
                    std_curve = np.std(arr, axis=0)
            
            if mean_curve is not None:
                # Pad/Fix 0 at start if needed (Lift notebook does this inside plot_with_std)
                # But here we standardize. 
                plot_with_std(plt.gca(), x, mean_curve, std_curve, label, color)

        plt.title(title, fontsize=36)
        plt.xlabel("Training Steps", fontsize=28)
        plt.ylabel(ylabel, fontsize=28)
        
        if use_percent:
            plt.ylim(-5, 100) # Or dynamic
            
        plt.legend(frameon=False, loc="upper left", fontsize=22)
        plt.grid(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        
        # Formatting
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        plt.gca().tick_params(axis='both', which='major', labelsize=18)
        plt.gca().xaxis.offsetText.set_fontsize(18)
        
        save_path = os.path.join(args.output_dir, f"{args.task}_{filename_suffix}.pdf")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved {save_path}")
        plt.close()

    # 1. Total Success Rate
    plot_metric("total_success", "Success Rate", "Success Rate (%)", "total_success_rate", use_percent=True)
    
    # 2. Partial Success Rate (Not for Lift?)
    if args.task != "lift":
        plot_metric("partial_success", "Partial Success Rate", "Success Rate (%)", "partial_success_rate", use_percent=True)

    # 3. Other Metrics
    if args.task != "lift":
        extra_metrics = config.get("metrics", {})
        titles = config.get("plot_titles", {})
        ylabels = config.get("y_labels", {})
        
        for key, name in extra_metrics.items():
            t = titles.get(key, name)
            y = ylabels.get(key, "Value")
            plot_metric(key, t, y, key)

if __name__ == "__main__":
    main()
