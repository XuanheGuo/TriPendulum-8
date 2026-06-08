import os
import json
import numpy as np
from utils.logging_utils import setup_logger

logger = setup_logger("report_generator")

class ReportGenerator:
    """
    Analyzes historical and current metrics to check for anomalies (reward hacking, forgetting)
    and formats reports (Markdown and JSON).
    """
    def __init__(self, save_dir="results/analytics"):
        self.save_dir = save_dir
        self.history_path = os.path.join(save_dir, "evaluation_history.json")
        self.history = {}
        self.load_history()
        
    def load_history(self):
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    # JSON keys are strings, convert them back to integers (timesteps)
                    data = json.load(f)
                    self.history = {int(k): v for k, v in data.items()}
            except Exception as e:
                logger.error(f"Failed to load evaluation history: {e}")
                
    def save_history(self):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(self.history_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save evaluation history: {e}")

    def update_history(self, timestep, goal_metrics):
        """
        Stores metrics for the current timestep in history.
        """
        self.history[int(timestep)] = goal_metrics
        self.save_history()

    def detect_anomalies(self, timestep, current_metrics):
        """
        Checks for reward hacking and catastrophic forgetting by comparing the current step
        with the previous evaluation step in history.
        """
        anomaly_report = {
            "timestep": int(timestep),
            "reward_hacking_suspected": False,
            "reasons": []
        }
        forgetting_report = {
            "timestep": int(timestep),
            "catastrophic_forgetting_detected": False,
            "forgotten_goals": []
        }
        
        # Get previous step from history
        timesteps = sorted([t for t in self.history.keys() if t < timestep])
        if not timesteps:
            # First evaluation, cannot detect relative anomalies
            return anomaly_report, forgetting_report
            
        prev_step = timesteps[-1]
        prev_metrics = self.history[prev_step]
        
        # 1. Reward Hacking Checks
        avg_reward_curr = np.mean([m["average_reward"] for m in current_metrics.values()])
        avg_reward_prev = np.mean([m["average_reward"] for m in prev_metrics.values()])
        avg_success_curr = np.mean([m["success_rate"] for m in current_metrics.values()])
        avg_success_prev = np.mean([m["success_rate"] for m in prev_metrics.values()])
        
        avg_energy_curr = np.mean([m["average_energy"] for m in current_metrics.values()])
        avg_energy_prev = np.mean([m["average_energy"] for m in prev_metrics.values()])
        avg_pose_err_curr = np.mean([m["average_pose_error"] for m in current_metrics.values()])
        avg_pose_err_prev = np.mean([m["average_pose_error"] for m in prev_metrics.values()])
        
        avg_collision_curr = np.mean([m["track_collision_rate"] for m in current_metrics.values()])
        avg_collision_prev = np.mean([m["track_collision_rate"] for m in prev_metrics.values()])
        avg_spin_curr = np.mean([m["overspin_rate"] for m in current_metrics.values()])
        avg_spin_prev = np.mean([m["overspin_rate"] for m in prev_metrics.values()])
        
        # Case A: Reward goes up, but success rate drops
        if avg_reward_curr > avg_reward_prev and avg_success_curr < avg_success_prev - 0.05:
            anomaly_report["reward_hacking_suspected"] = True
            anomaly_report["reasons"].append(
                f"Global reward increased from {avg_reward_prev:.2f} to {avg_reward_curr:.2f}, "
                f"but success rate dropped from {avg_success_prev:.2%} to {avg_success_curr:.2%}"
            )
            
        # Case B: Energy increases massively, but pose error does not improve
        if avg_energy_curr > avg_energy_prev * 1.5 and avg_pose_err_curr > avg_pose_err_prev * 0.95:
            anomaly_report["reward_hacking_suspected"] = True
            anomaly_report["reasons"].append(
                f"Energy consumption spiked by {(avg_energy_curr - avg_energy_prev)/avg_energy_prev:.1%}, "
                f"but pose error did not improve significantly (prev: {avg_pose_err_prev:.3f}, curr: {avg_pose_err_curr:.3f})"
            )
            
        # Case C: Collision rate goes up while reward increases
        if avg_collision_curr > avg_collision_prev + 0.1 and avg_reward_curr > avg_reward_prev:
            anomaly_report["reward_hacking_suspected"] = True
            anomaly_report["reasons"].append(
                f"Track collision rate increased from {avg_collision_prev:.2%} to {avg_collision_curr:.2%} "
                f"even though training reward increased"
            )
            
        # Case D: Overspin rate increases
        if avg_spin_curr > avg_spin_prev + 0.1 and avg_reward_curr > avg_reward_prev:
            anomaly_report["reward_hacking_suspected"] = True
            anomaly_report["reasons"].append(
                f"Joint overspin rate increased from {avg_spin_prev:.2%} to {avg_spin_curr:.2%} "
                f"indicating unstable spinning loops"
            )

        # 2. Catastrophic Forgetting Checks
        # Compare each goal's historical peak success rate with current success rate
        for goal_name in current_metrics.keys():
            historical_success_rates = [
                self.history[t][goal_name]["success_rate"]
                for t in sorted(self.history.keys())
                if goal_name in self.history[t] and t < timestep
            ]
            if not historical_success_rates:
                continue
                
            peak_success = max(historical_success_rates)
            current_success = current_metrics[goal_name]["success_rate"]
            
            # If a goal once had >80% success and now fell to <50%
            if peak_success > 0.8 and current_success < 0.5:
                forgetting_report["catastrophic_forgetting_detected"] = True
                forgetting_report["forgotten_goals"].append({
                    "goal": goal_name,
                    "peak_success_rate": peak_success,
                    "current_success_rate": current_success
                })
                
        # Write reports to JSON
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "anomaly_report.json"), "w") as f:
                json.dump(anomaly_report, f, indent=2)
            with open(os.path.join(self.save_dir, "forgetting_report.json"), "w") as f:
                json.dump(forgetting_report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write anomaly/forgetting reports: {e}")
            
        return anomaly_report, forgetting_report

    def generate_step_report(
        self,
        timestep,
        algorithm,
        stage_id,
        stage_goals,
        goal_metrics,
        worst_goals,
        best_goals,
        anomaly_report,
        forgetting_report,
        video_metadata
    ):
        """
        Creates report_step_xxxxxx.md inside results/analytics directory.
        """
        report_dir = os.path.join(self.save_dir, f"step_{timestep:07d}")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"report_step_{timestep}.md")
        
        md = []
        md.append(f"# Performance Report - Step {timestep}")
        md.append(f"**Algorithm**: {algorithm}")
        md.append(f"**Current Curriculum Stage**: {stage_id} (Active Goals: {', '.join(stage_goals)})")
        md.append("")
        
        md.append("## Goal Rankings")
        md.append("| Goal | Success Rate | Avg Reward | Avg Pose Error | Avg Energy | Avg Max X | Collision Rate | Overspin Rate |")
        md.append("|---|---|---|---|---|---|---|---|")
        
        # Sort goals by success rate descending
        sorted_goals = sorted(goal_metrics.keys(), key=lambda g: goal_metrics[g]["success_rate"], reverse=True)
        for goal in sorted_goals:
            m = goal_metrics[goal]
            md.append(
                f"| **{goal}** | {m['success_rate']:.1%} | {m['average_reward']:.2f} | {m['average_pose_error']:.3f} | "
                f"{m['average_energy']:.3f} | {m['average_max_cart_displacement']:.2f} | {m['track_collision_rate']:.1%} | "
                f"{m['overspin_rate']:.1%} |"
            )
        md.append("")
        
        md.append(f"**Best Performing Goal(s)**: {', '.join(best_goals)}")
        md.append(f"**Worst Performing Goal(s)**: {', '.join(worst_goals)}")
        md.append("")
        
        md.append("## Anomaly & Risk Reports")
        if anomaly_report["reward_hacking_suspected"]:
            md.append("> [!WARNING]")
            md.append("> **Potential Reward Hacking Detected!**")
            for reason in anomaly_report["reasons"]:
                md.append(f"> - {reason}")
        else:
            md.append("> [!NOTE]")
            md.append("> No signs of reward hacking detected.")
        md.append("")
        
        if forgetting_report["catastrophic_forgetting_detected"]:
            md.append("> [!CAUTION]")
            md.append("> **Catastrophic Forgetting Detected!**")
            for fg in forgetting_report["forgotten_goals"]:
                md.append(
                    f"> - Goal **{fg['goal']}** plummeted to {fg['current_success_rate']:.1%} success rate "
                    f"(previous peak: {fg['peak_success_rate']:.1%})"
                )
        else:
            md.append("> [!NOTE]")
            md.append("> No catastrophic forgetting detected.")
        md.append("")
        
        md.append("## Diagnostic Videos")
        if video_metadata:
            md.append("| Goal | Selection Reason | Episode Reward | Length | Success | Max X | Collided | Overspun |")
            md.append("|---|---|---|---|---|---|---|---|")
            for vm in video_metadata:
                md.append(
                    f"| **{vm['goal']}** | {vm['selection_reason']} | {vm['episode_reward']:.2f} | {vm['episode_length']} | "
                    f"{vm['success']} | {vm['max_abs_x']:.2f} | {vm['track_collision']} | {vm['overspin']} |"
                )
            md.append("")
            md.append("Videos have been exported to the diagnostics folder.")
        else:
            md.append("No videos generated in this step.")
        md.append("")
        
        # Recommendations
        md.append("## Recommended Actions")
        if forgetting_report["catastrophic_forgetting_detected"]:
            md.append("- **Action**: Add forgotten goals back to training buffers or adjust curriculum to prevent goal omission.")
        if anomaly_report["reward_hacking_suspected"]:
            md.append("- **Action**: Increase boundary collision penalties (`collision_penalty`) or tune weight coefficients `w_track`/`w_spin` to suppress exploits.")
        if not forgetting_report["catastrophic_forgetting_detected"] and not anomaly_report["reward_hacking_suspected"]:
            md.append("- **Action**: Policy is training normally. Continue monitoring.")
            
        try:
            with open(report_path, "w") as f:
                f.write("\n".join(md))
            logger.info(f"Generated Markdown report at {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate Markdown report: {e}")
            
        return report_path

    def generate_final_report(
        self,
        algorithm,
        config,
        curriculum_history,
        final_goal_success,
        worst_goals,
        reasons_summary
    ):
        """
        Creates final_report.md inside the project results/ folder.
        """
        report_path = os.path.join(self.save_dir, "..", "final_report.md")
        report_path = os.path.abspath(report_path)
        
        md = []
        md.append("# Final Training & Analysis Report - TriPendulum-8")
        md.append(f"**Algorithm**: {algorithm}")
        md.append("")
        
        md.append("## 1. Environment & Training Configuration")
        md.append("```yaml")
        md.append(json.dumps(config, indent=2))
        md.append("```")
        md.append("")
        
        md.append("## 2. Curriculum Progression History")
        if curriculum_history:
            md.append("| Step | From Stage | To Stage | Description |")
            md.append("|---|---|---|---|")
            for ch in curriculum_history:
                stage_name = f"Stage {ch['to_stage']}"
                md.append(f"| {ch['timestep']} | Stage {ch['from_stage']} | {stage_name} | Transitioned successfully |")
        else:
            md.append("Curriculum training was disabled or did not advance past Stage 1.")
        md.append("")
        
        md.append("## 3. Final Goal-Conditioned Success Rates")
        md.append("| Goal Pose | Success Rate |")
        md.append("|---|---|")
        for goal, rate in final_goal_success.items():
            md.append(f"| **{goal}** | {rate:.1%} |")
        md.append("")
        
        md.append("## 4. Diagnostics & Failures Summary")
        md.append(f"**Worst goals**: {', '.join(worst_goals)}")
        md.append("")
        md.append("### Failure Analysis")
        md.append(reasons_summary)
        md.append("")
        
        md.append("## 5. Potential Future Enhancements")
        md.append("1. **HER (Hindsight Experience Replay)**: Map failed trajectories to their achieved endpoints as virtual successful goals.")
        md.append("2. **MPC / iLQR Fine-Tuning**: Wrap the RL action space inside an MPC controller for local stability.")
        md.append("3. **Domain Randomization**: Randomize pole masses and lengths during training to enable zero-shot transfer.")
        md.append("4. **Sim-to-Real**: Apply delay compensation and actuator dynamics filters.")
        
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                f.write("\n".join(md))
            logger.info(f"Generated final Markdown report at {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
            
        return report_path
