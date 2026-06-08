import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logging_utils import setup_logger

logger = setup_logger("live_panel")

class LivePanel:
    """
    Renders dynamic training dashboards inside Google Colab cells.
    """
    def __init__(self, analytics_dir="results/analytics"):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(curr_dir, ".."))
        self.analytics_dir = os.path.join(self.project_root, analytics_dir)
        self.history_path = os.path.join(self.analytics_dir, "evaluation_history.json")
        self.curriculum_path = os.path.join(self.analytics_dir, "curriculum_history.json")
        
    def _load_history(self):
        if not os.path.exists(self.history_path):
            return None
        try:
            with open(self.history_path, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except Exception:
            return None
            
    def _load_curriculum(self):
        if not os.path.exists(self.curriculum_path):
            return []
        try:
            with open(self.curriculum_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def update(self):
        """
        Clears the current cell and renders an updated graphical dashboard and textual progress summary.
        """
        history = self._load_history()
        if not history:
            print("No training analytics data available yet. Waiting for first evaluation...")
            return
            
        curriculum_history = self._load_curriculum()
        
        # Determine dynamic cell updates (IPython only works inside notebooks)
        try:
            from IPython.display import clear_output, display
            clear_output(wait=True)
        except ImportError:
            pass
            
        timesteps = sorted(history.keys())
        latest_step = timesteps[-1]
        latest_data = history[latest_step]
        goals = list(latest_data.keys())
        
        # Calculate summary statistics
        success_rates = [latest_data[g]["success_rate"] for g in goals]
        avg_rewards = [latest_data[g]["average_reward"] for g in goals]
        avg_energies = [latest_data[g]["average_energy"] for g in goals]
        avg_collisions = [latest_data[g]["track_collision_rate"] for g in goals]
        avg_overspins = [latest_data[g]["overspin_rate"] for g in goals]
        
        current_stage = 1
        if curriculum_history:
            current_stage = curriculum_history[-1]["to_stage"]
            
        # 1. Plot Graphical Dashboard
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Subplot 1: Success Rate Curve
        for goal in goals:
            rates = [history[t][goal]["success_rate"] for t in timesteps]
            axs[0, 0].plot(timesteps, rates, label=goal, alpha=0.7)
        global_avg_success = [np.mean([history[t][g]["success_rate"] for g in goals]) for t in timesteps]
        axs[0, 0].plot(timesteps, global_avg_success, color="black", linestyle="--", label="Global Avg", linewidth=2)
        axs[0, 0].set_title("Success Rates by Goal")
        axs[0, 0].set_xlabel("Timesteps")
        axs[0, 0].set_ylabel("Success Rate")
        axs[0, 0].set_ylim(-0.05, 1.05)
        axs[0, 0].legend(loc="lower right")
        
        # Subplot 2: Reward Curve
        for goal in goals:
            rewards = [history[t][goal]["average_reward"] for t in timesteps]
            axs[0, 1].plot(timesteps, rewards, label=goal, alpha=0.7)
        global_avg_reward = [np.mean([history[t][g]["average_reward"] for g in goals]) for t in timesteps]
        axs[0, 1].plot(timesteps, global_avg_reward, color="black", linestyle="--", label="Global Avg", linewidth=2)
        axs[0, 1].set_title("Average Reward by Goal")
        axs[0, 1].set_xlabel("Timesteps")
        axs[0, 1].set_ylabel("Reward")
        axs[0, 1].legend(loc="lower right")
        
        # Subplot 3: Goal Success Rates Bar Chart
        colors = sns.color_palette("viridis", len(goals))
        bars = axs[1, 0].bar(goals, success_rates, color=colors, edgecolor='grey')
        axs[1, 0].set_title(f"Success Rates at Step {latest_step}")
        axs[1, 0].set_xlabel("Goal")
        axs[1, 0].set_ylabel("Success Rate")
        axs[1, 0].set_ylim(0, 1.05)
        for bar in bars:
            h = bar.get_height()
            axs[1, 0].text(bar.get_x() + bar.get_width()/2.0, h + 0.01, f"{h:.1%}", ha="center", va="bottom", fontsize=8)
            
        # Subplot 4: Energy (Effort) Curve
        for goal in goals:
            energies = [history[t][goal]["average_energy"] for t in timesteps]
            axs[1, 1].plot(timesteps, energies, label=goal, alpha=0.7)
        axs[1, 1].set_title("Action Effort by Goal")
        axs[1, 1].set_xlabel("Timesteps")
        axs[1, 1].set_ylabel("Energy (Action^2)")
        axs[1, 1].legend(loc="upper right")
        
        plt.suptitle(f"TriPendulum-8 Training Dashboard (Curriculum Stage {current_stage} | Step {latest_step})", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        try:
            display(plt.gcf())
        except NameError:
            plt.show()
        plt.close()
        
        # 2. Print Text-based Dashboard
        worst_idx = np.argmin(success_rates)
        best_idx = np.argmax(success_rates)
        
        print("=" * 60)
        print("                  TRIPENDULUM-8 STATUS PANEL                  ")
        print("=" * 60)
        print(f"Current Training Step: {latest_step}")
        print(f"Curriculum Stage     : {current_stage}")
        print(f"Global Success Rate  : {np.mean(success_rates):.2%}")
        print(f"Global Avg Reward    : {np.mean(avg_rewards):.2f}")
        print(f"Global Avg Energy    : {np.mean(avg_energies):.4f}")
        print(f"Global Collision Rate: {np.mean(avg_collisions):.2%}")
        print(f"Global Overspin Rate : {np.mean(avg_overspins):.2%}")
        print("-" * 60)
        print(f"Best Performing Goal : {goals[best_idx]} ({success_rates[best_idx]:.1%} Success)")
        print(f"Worst Performing Goal: {goals[worst_idx]} ({success_rates[worst_idx]:.1%} Success)")
        
        # Load anomaly report if exists
        anomaly_path = os.path.join(self.analytics_dir, "anomaly_report.json")
        if os.path.exists(anomaly_path):
            try:
                with open(anomaly_path, "r") as f:
                    anom = json.load(f)
                    if anom.get("reward_hacking_suspected", False):
                        print("\n[!] WARNING: Potential Reward Hacking Detected!")
                        for r in anom.get("reasons", []):
                            print(f"    - {r}")
            except Exception:
                pass
                
        # Load forgetting report if exists
        forget_path = os.path.join(self.analytics_dir, "forgetting_report.json")
        if os.path.exists(forget_path):
            try:
                with open(forget_path, "r") as f:
                    forg = json.load(f)
                    if forg.get("catastrophic_forgetting_detected", False):
                        print("\n[!] WARNING: Catastrophic Forgetting Detected!")
                        for g in forg.get("forgotten_goals", []):
                            print(f"    - Goal {g['goal']} fell to {g['current_success_rate']:.1%} (Peak: {g['peak_success_rate']:.1%})")
            except Exception:
                pass
                
        print("=" * 60)
