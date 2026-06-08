import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.logging_utils import setup_logger

logger = setup_logger("dashboard_plots")

# Set premium styling defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "savefig.dpi": 150
})

def generate_all_plots(save_dir="results", analytics_dir="results/analytics"):
    """
    Reads analytics histories and generates comparison and performance charts.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(curr_dir, ".."))
    
    analytics_path = os.path.join(project_root, analytics_dir)
    output_path = os.path.join(project_root, save_dir)
    os.makedirs(output_path, exist_ok=True)
    
    history_json = os.path.join(analytics_path, "evaluation_history.json")
    if not os.path.exists(history_json):
        logger.warning(f"Could not generate training plots. History file not found at {history_json}")
        return
        
    try:
        with open(history_json, "r") as f:
            data = json.load(f)
            # Keys are timesteps as strings, convert to ints
            history = {int(k): v for k, v in data.items()}
    except Exception as e:
        logger.error(f"Error reading evaluation history: {e}")
        return
        
    if not history:
        logger.warning("History file is empty.")
        return
        
    timesteps = sorted(history.keys())
    goals = list(history[timesteps[0]].keys())
    
    # 1. Success Rate Curve
    plt.figure(figsize=(8, 5))
    for goal in goals:
        rates = [history[t][goal]["success_rate"] for t in timesteps]
        plt.plot(timesteps, rates, label=goal, alpha=0.8, linewidth=1.5)
    # Global average success rate
    global_avg = [np.mean([history[t][g]["success_rate"] for g in goals]) for t in timesteps]
    plt.plot(timesteps, global_avg, label="GLOBAL AVG", color="black", linestyle="--", linewidth=2.5)
    plt.title("Goal Success Rates over Training Steps")
    plt.xlabel("Timesteps")
    plt.ylabel("Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "success_rate_curve.png"))
    plt.close()
    
    # 2. Reward Progression Curve
    plt.figure(figsize=(8, 5))
    for goal in goals:
        rewards = [history[t][goal]["average_reward"] for t in timesteps]
        plt.plot(timesteps, rewards, label=goal, alpha=0.8)
    global_rew = [np.mean([history[t][g]["average_reward"] for g in goals]) for t in timesteps]
    plt.plot(timesteps, global_rew, label="GLOBAL AVG", color="black", linestyle="--", linewidth=2.5)
    plt.title("Average Reward over Training Steps")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "reward_curve.png"))
    plt.close()
    
    # 3. Energy Consumption Curve
    plt.figure(figsize=(8, 5))
    for goal in goals:
        energies = [history[t][goal]["average_energy"] for t in timesteps]
        plt.plot(timesteps, energies, label=goal, alpha=0.8)
    plt.title("Action Effort (Energy) over Training Steps")
    plt.xlabel("Timesteps")
    plt.ylabel("Effort (Average Control Force Squared)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "energy_curve.png"))
    plt.close()

    # 4. Max Cart Displacement
    plt.figure(figsize=(8, 5))
    for goal in goals:
        disp = [history[t][goal]["average_max_cart_displacement"] for t in timesteps]
        plt.plot(timesteps, disp, label=goal, alpha=0.8)
    plt.axhline(y=1.0, color='r', linestyle=':', label='Track Max Bound')
    plt.title("Max Cart Displacement over Training Steps")
    plt.xlabel("Timesteps")
    plt.ylabel("Displacement (meters)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "track_position_curve.png"))
    plt.close()
    
    # 5. Final Step Goal Success Rates Comparison (Bar Chart)
    latest_step = timesteps[-1]
    latest_data = history[latest_step]
    
    plt.figure(figsize=(8, 5))
    success_rates = [latest_data[g]["success_rate"] for g in goals]
    colors = sns.color_palette("viridis", len(goals))
    bars = plt.bar(goals, success_rates, color=colors, edgecolor='grey', alpha=0.9)
    plt.title(f"Goal Success Rates at Step {latest_step}")
    plt.xlabel("Goal Config")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    
    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "goal_success_bar.png"))
    plt.close()
    
    # 6. Try loading Transition Matrix from CSV and save heatmap if matches
    csv_path = os.path.join(output_path, "transition_matrix.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col=0)
            plt.figure(figsize=(9, 7))
            sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1%", cbar=True, vmin=0.0, vmax=1.0)
            plt.title("Goal-to-Goal Transition Success Heatmap (8x8)")
            plt.xlabel("Target Goal")
            plt.ylabel("Initial Goal")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "transition_matrix_heatmap.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate transition heatmap from CSV: {e}")

    logger.info("Successfully refreshed all training analysis charts in results/")

if __name__ == "__main__":
    generate_all_plots()
