import numpy as np
import os
import subprocess
import pickle
import matplotlib.pyplot as plt
import re

def get_model_directories(base_dir):
    """
    Get all checkpoint directories with format rl_model_{stepsize}_steps from the given base directory.
    
    Args:
        base_dir (str): Parent directory to search for checkpoint folders
        
    Returns:
        list: List of absolute paths to rl_model directories, sorted by step size
    """
    model_dirs = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist")
        return model_dirs
    
    # Pattern to match rl_model_{stepsize}_steps directories
    pattern = re.compile(r'^rl_model_(\d+)_steps$')
    
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                match = pattern.match(item)
                if match:
                    step_size = int(match.group(1))
                    model_dirs.append((step_size, item_path))
        
        # Sort by step size and return only the paths
        model_dirs.sort(key=lambda x: x[0])
        return [path for _, path in model_dirs]
        
    except OSError as e:
        print(f"Error reading directory {base_dir}: {e}")
        return []
    
def run_eval(model_path):
    process = subprocess.run(["python",
                            "./openrl_ws/go2gate_eval.py",
                            "--task", "go2gate",
                            "--algo", "ppo",
                            "--sim_device", "cuda:0",
                            "--rl_device", "cuda:0",
                            "--num_envs", "1",
                            "--checkpoint", str(model_path),
                            "--headless"
                            ],
                            )
    
    # Load the evaluation results
    eval_results_path = os.path.join(model_path, "eval_results.pkl")
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'rb') as f:
            eval_results = pickle.load(f)
        print(f"Evaluation results for {model_path}:")
        print(f"Total runs: {eval_results['total_runs']}, "
              f"Successful runs: {eval_results['success_runs']}, "
              f"Partial success runs: {eval_results['partial_success_runs']}, "
              f"Average reward: {eval_results['average_reward']:.2f}, "
              f"Std reward: {eval_results['std_reward']:.2f}")
        
    return eval_results

def plot_results(save_dir, total_success_rate_curve, partial_success_rate_curve, average_reward_curve, std_reward_curve):  
    # Use palatino for plotting
    plt.rcParams.update({'font.family': 'Palatino'})
    
    # Plot total success and partial success rates in two different plots
    plt.figure(figsize=(12, 6))
    plt.plot(total_success_rate_curve, label='Total Success Rate')
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Success Rate (%)')
    plt.title('Total Success Rate Curve')
    plt.savefig(os.path.join(save_dir, 'total_success_rate_curve.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(partial_success_rate_curve, label='Partial Success Rate', color='orange')
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Success Rate (%)')
    plt.title('Partial Success Rate Curve')
    plt.savefig(os.path.join(save_dir, 'partial_success_rate_curve.png'))
    plt.show()

    # Plot reward curve with using std as shadow
    plt.figure(figsize=(12, 6))
    plt.plot(average_reward_curve, label='Average Reward')
    plt.fill_between(range(len(average_reward_curve)),
                     np.array(average_reward_curve) - np.array(std_reward_curve),
                     np.array(average_reward_curve) + np.array(std_reward_curve),
                     color='gray', alpha=0.5, label='Std Dev')
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Curve with Std Dev')
    plt.savefig(os.path.join(save_dir, 'average_reward_curve.png'))
    plt.show()

def plot_multiple_results(save_dir, total_success_summary, partial_success_summary, average_reward_summary, std_reward_summary):  
    # Use palatino for plotting
    plt.rcParams.update({'font.family': 'Palatino'})
    
    # Plot total success and fill the std
    plt.figure(figsize=(12, 6))
    total_success_mean = np.mean(total_success_summary, axis=0)
    total_success_std = np.std(total_success_summary, axis=0)
    plt.plot(total_success_mean, label='Total Success Rate')
    plt.fill_between(range(len(total_success_mean)),
                     total_success_mean - total_success_std,
                     total_success_mean + total_success_std,
                     color='gray', alpha=0.5, label='Std Dev')
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Success Rate (%)')
    plt.title('Total Success Rate Curve')
    plt.savefig(os.path.join(save_dir, 'total_success_rate_curve.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    partial_success_mean = np.mean(partial_success_summary, axis=0)
    partial_success_std = np.std(partial_success_summary, axis=0)
    plt.plot(partial_success_mean, label='Partial Success Rate', color='orange')
    plt.fill_between(range(len(partial_success_mean)),
                     partial_success_mean - partial_success_std,
                     partial_success_mean + partial_success_std,
                     color='gray', alpha=0.5, label='Std Dev')
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Success Rate (%)')
    plt.title('Partial Success Rate Curve')
    plt.savefig(os.path.join(save_dir, 'partial_success_rate_curve.png'))
    plt.show()

    # Plot reward curve with using std as shadow
    plt.figure(figsize=(12, 6))
    average_reward_mean = np.mean(average_reward_summary, axis=0)
    average_reward_std = np.std(average_reward_summary, axis=0)
    plt.plot(average_reward_mean, label='Average Reward')
    plt.fill_between(range(len(average_reward_mean)),
                     average_reward_mean - average_reward_std,
                     average_reward_mean + average_reward_std,
                     color='gray', alpha=0.5, label='Std Dev')
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Curve with Std Dev')
    plt.savefig(os.path.join(save_dir, 'average_reward_curve.png'))
    plt.show()


def lookup_optimal_dir(experiment_dir):
    """
    Look for optimal model directories in an experiment directory.
    
    Args:
        experiment_dir (str): Path to experiment directory
        
    Returns:
        list: List of optimal model directories
    """
    optimal_dirs = []
    
    if not os.path.exists(experiment_dir):
        print(f"Warning: Experiment directory {experiment_dir} does not exist")
        return optimal_dirs
    
    # Get all task directories (numbered folders)
    task_dirs = []
    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)
        if os.path.isdir(item_path) and item[0].isdigit():
            task_dirs.append((item, item_path))
    
    # Sort task directories by number
    task_dirs.sort(key=lambda x: int(x[0].split('_')[0]))
    
    for task_name, task_path in task_dirs:
        # Check if there's a decision file for this task
        decision_file = os.path.join(experiment_dir, f"{task_name}.md")
        
        if os.path.exists(decision_file):
            # Parse decision file to extract experiment number
            try:
                with open(decision_file, 'r') as f:
                    content = f.read()
                    
                # Look for "Decision: Experiment X" pattern
                decision_match = re.search(r'Decision:\s*Experiment\s*(\d+)', content, re.IGNORECASE)
                if decision_match:
                    experiment_num = int(decision_match.group(1))
                    optimal_sample_dir = os.path.join(task_path, f"sample_{experiment_num}")
                    
                    if os.path.exists(optimal_sample_dir):
                        model_dir = os.path.join(optimal_sample_dir, "model")
                        if os.path.exists(model_dir):
                            optimal_dirs.append(model_dir)
                            print(f"Found decision-based optimal dir for {task_name}: sample_{experiment_num}")
                        else:
                            print(f"Warning: Model directory not found in {optimal_sample_dir}")
                    else:
                        print(f"Warning: Sample directory {optimal_sample_dir} not found")
                else:
                    print(f"Warning: Could not parse decision from {decision_file}")
                    
            except Exception as e:
                print(f"Error reading decision file {decision_file}: {e}")
        
        if not any(task_name in path for path in optimal_dirs):
            # No decision file found or parsing failed, use highest numbered sample
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
                    # Sort by sample number and get the highest one
                    sample_dirs.sort(key=lambda x: x[0])
                    highest_sample_num, highest_sample_dir = sample_dirs[-1]
                    
                    model_dir = os.path.join(highest_sample_dir, "model")
                    if os.path.exists(model_dir):
                        optimal_dirs.append(model_dir)
                        print(f"Using highest sample_{highest_sample_num} for {task_name}")
                    else:
                        print(f"Warning: Model directory not found in {highest_sample_dir}")
                else:
                    print(f"Warning: No sample directories found in {task_path}")
                    
            except OSError as e:
                print(f"Error listing directories in {task_path}: {e}")
    
    return optimal_dirs

# Pad all curves to max_length by extending with the final value
def pad_curve_to_length(curve, target_length):
    if len(curve) == 0:
        return [0] * target_length
    if len(curve) >= target_length:
        return curve[:target_length]
    # Extend with the final value
    final_value = curve[-1]
    padded_curve = curve + [final_value] * (target_length - len(curve))
    return padded_curve

def main():
    # Check if wrapper file is configured correctly
    input("Have you configured the wrapper file to output mqe reward? Press any key to continue...")
    print("Starting evaluation...")

    experiment_directories = [
        "/home/kang/mqe-curriculum/logs/go2gate/08-02_12-13",
        "/home/kang/mqe-curriculum/logs/go2gate/08-02_19-44",
    ]

    total_success_summary = []
    partial_success_summary = []
    average_reward_summary = []
    std_reward_summary = []

    for experiment in experiment_directories:
        base_directories = lookup_optimal_dir(experiment)

        total_success_rate_curve = []
        partial_success_rate_curve = []
        average_reward_curve = []
        std_reward_curve = []

        for base_dir in base_directories:
            model_dirs = get_model_directories(base_dir)
            if not model_dirs:
                print(f"No model directories found in {base_dir}")
                break
            
            print(f"Found {len(model_dirs)} model directories in {base_dir}")
            
            for model_path in model_dirs:
                print(f"Evaluating model: {model_path}")
                eval_results = run_eval(model_path)
                
                total_success_rate_curve.append(eval_results['success_runs'] / eval_results['total_runs'] * 100 if eval_results['total_runs'] > 0 else 0)
                partial_success_rate_curve.append(eval_results['partial_success_runs'] / eval_results['total_runs'] * 100 if eval_results['total_runs'] > 0 else 0)
                average_reward_curve.append(eval_results['average_reward'])
                std_reward_curve.append(eval_results['std_reward'])

        # Plot the results
        plot_results(experiment, total_success_rate_curve, partial_success_rate_curve, average_reward_curve, std_reward_curve)
        total_success_summary.append(total_success_rate_curve)
        partial_success_summary.append(partial_success_rate_curve)
        average_reward_summary.append(average_reward_curve)
        std_reward_summary.append(std_reward_curve)

    # convert to numpy arrays for easier handling
    # Find the maximum length among all curves
    max_length = 0
    for curve_list in [total_success_summary, partial_success_summary, average_reward_summary, std_reward_summary]:
        for curve in curve_list:
            max_length = max(max_length, len(curve))
    
    print(f"Maximum curve length: {max_length}")
    
    # Pad and convert to numpy arrays
    total_success_summary_padded = []
    for curve in total_success_summary:
        padded_curve = pad_curve_to_length(curve, max_length)
        total_success_summary_padded.append(padded_curve)
    total_success_summary = np.array(total_success_summary_padded)
    
    partial_success_summary_padded = []
    for curve in partial_success_summary:
        padded_curve = pad_curve_to_length(curve, max_length)
        partial_success_summary_padded.append(padded_curve)
    partial_success_summary = np.array(partial_success_summary_padded)
    
    average_reward_summary_padded = []
    for curve in average_reward_summary:
        padded_curve = pad_curve_to_length(curve, max_length)
        average_reward_summary_padded.append(padded_curve)
    average_reward_summary = np.array(average_reward_summary_padded)
    
    std_reward_summary_padded = []
    for curve in std_reward_summary:
        padded_curve = pad_curve_to_length(curve, max_length)
        std_reward_summary_padded.append(padded_curve)
    std_reward_summary = np.array(std_reward_summary_padded)
    
    # Plot the summary curves
    plot_multiple_results("/home/kang/mqe-curriculum/logs/go2gate", 
                          total_success_summary, 
                          partial_success_summary, 
                          average_reward_summary, 
                          std_reward_summary)

if __name__ == "__main__":
    main()