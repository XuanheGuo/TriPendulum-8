import os
import yaml
import json
import numpy as np

from analytics.evaluator import Evaluator
from analytics.video_recorder import VideoRecorder
from analytics.report_generator import ReportGenerator
from analytics.curriculum_monitor import CurriculumMonitor
from analytics.metrics import aggregate_episode_metrics
from utils.logging_utils import setup_logger
from envs.goals import GOAL_NAMES

logger = setup_logger("analytics_manager")

class AnalyticsManager:
    """
    Coordinates evaluation, metrics aggregation, anomaly detection,
    diagnostics video rendering, and markdown report generation.
    """
    def __init__(self, model, config_path=None, tensorboard_log_dir=None):
        self.model = model
        self.config_path = config_path
        
        # Load configuration
        if config_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(curr_dir, "..", "configs", "default.yaml")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Set up absolute paths for workspace stability
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(curr_dir, ".."))
        
        analytics_cfg = self.config.get("analytics", {})
        self.save_dir = os.path.join(project_root, analytics_cfg.get("save_dir", "results/analytics"))
        self.video_dir = os.path.join(project_root, analytics_cfg.get("video_dir", "videos/diagnostics"))
        self.n_eval_episodes = analytics_cfg.get("n_eval_episodes", 5)
        self.video_top_k = analytics_cfg.get("video_top_k", 3)
        self.mode = analytics_cfg.get("mode", "curriculum_worst")
        self.algorithm = self.config.get("algorithm", "SAC") # will fallback or read from config
        
        # Initialize modules
        self.evaluator = Evaluator(config_path=self.config_path)
        self.video_recorder = VideoRecorder(
            config_path=self.config_path,
            video_dir=self.video_dir,
            algorithm=self.algorithm
        )
        self.report_generator = ReportGenerator(save_dir=self.save_dir)
        self.curriculum_monitor = CurriculumMonitor(save_dir=self.save_dir)
        
        # dt for stability time conversion (frame_skip * mujoco_timestep)
        env_cfg = self.config.get("env", {})
        self.dt = env_cfg.get("frame_skip", 5) * 0.002
        
    def run_evaluation(self, timestep, stage_id, stage_goals):
        """
        Runs the full evaluation suite for the current training step.
        """
        logger.info(f"Running evaluation at step {timestep} (Curriculum Stage {stage_id})...")
        
        # 1. Evaluate all 8 goals to have a complete metric history
        all_metrics = {}
        for goal in GOAL_NAMES:
            episode_logs = self.evaluator.evaluate_goal(
                model=self.model,
                goal_name=goal,
                n_episodes=self.n_eval_episodes,
                curriculum_stage=stage_id
            )
            # Aggregate metrics for the goal
            goal_m = aggregate_episode_metrics(episode_logs, dt=self.dt)
            all_metrics[goal] = goal_m
            
        # 2. Update metric history
        self.report_generator.update_history(timestep, all_metrics)
        
        # 3. Detect anomalies (reward hacking, forgetting)
        anomaly_rep, forgetting_rep = self.report_generator.detect_anomalies(timestep, all_metrics)
        
        # 4. Compute curriculum progression success rate (average success of current stage goals)
        stage_successes = [all_metrics[g]["success_rate"] for g in stage_goals]
        stage_success_rate = float(np.mean(stage_successes)) if stage_successes else 0.0
        
        # Save goal metrics to step folder
        step_dir = os.path.join(self.save_dir, f"step_{timestep:07d}")
        os.makedirs(step_dir, exist_ok=True)
        metrics_json_path = os.path.join(step_dir, "goal_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
            
        # 5. Determine which goals to record videos for based on mode
        goals_to_record = self._select_goals_for_video(all_metrics, stage_goals)
        
        # 6. Render and save videos
        video_metadata = []
        for goal_name, reason in goals_to_record:
            meta = self.video_recorder.record_goal_episode(
                model=self.model,
                goal_name=goal_name,
                timestep=timestep,
                stage_id=stage_id,
                stage_goals=stage_goals,
                selection_reason=reason
            )
            video_metadata.append(meta)
            
        # 7. Generate Step Markdown Report
        best_goals = self._get_best_goals(all_metrics)
        worst_goals = [g for g, _ in goals_to_record if "worst" in _ or "failure" in _]
        if not worst_goals:
            # Fallback to absolute lowest success rate
            sorted_worst = sorted(all_metrics.keys(), key=lambda g: all_metrics[g]["success_rate"])
            worst_goals = sorted_worst[:self.video_top_k]
            
        self.report_generator.generate_step_report(
            timestep=timestep,
            algorithm=self.algorithm,
            stage_id=stage_id,
            stage_goals=stage_goals,
            goal_metrics=all_metrics,
            worst_goals=worst_goals,
            best_goals=best_goals,
            anomaly_report=anomaly_rep,
            forgetting_report=forgetting_rep,
            video_metadata=video_metadata
        )
        
        return {
            "stage_success_rate": stage_success_rate,
            "all_metrics": all_metrics
        }
        
    def _select_goals_for_video(self, all_metrics, stage_goals):
        """
        Applies selection logic to decide which goals need diagnostic videos.
        Returns list of (goal_name, reason) tuples.
        """
        goals_to_record = []
        
        # Mode: fixed
        if self.mode == "fixed":
            # Record first 2 stage goals
            for g in stage_goals[:self.video_top_k]:
                goals_to_record.append((g, "Config fixed goal selection"))
                
        # Mode: all
        elif self.mode == "all":
            for g in GOAL_NAMES:
                goals_to_record.append((g, "All goals video mode"))
                
        # Mode: worst (across all 8 goals)
        elif self.mode == "worst":
            sorted_worst = sorted(all_metrics.keys(), key=lambda g: all_metrics[g]["success_rate"])
            for g in sorted_worst[:self.video_top_k]:
                goals_to_record.append((g, f"Worst performing goal (success rate: {all_metrics[g]['success_rate']:.1%})"))
                
        # Mode: curriculum (all goals in current stage)
        elif self.mode == "curriculum":
            for g in stage_goals:
                goals_to_record.append((g, f"Curriculum stage goal"))
                
        # Mode: curriculum_worst (worst goals within current curriculum stage)
        elif self.mode == "curriculum_worst" or not stage_goals:
            # Filter all_metrics to only stage goals
            stage_metrics = {g: all_metrics[g] for g in stage_goals if g in all_metrics}
            if not stage_metrics:
                stage_metrics = all_metrics # Fallback
            sorted_worst = sorted(stage_metrics.keys(), key=lambda g: stage_metrics[g]["success_rate"])
            for g in sorted_worst[:self.video_top_k]:
                goals_to_record.append((g, f"Curriculum worst goal (success rate: {stage_metrics[g]['success_rate']:.1%})"))
                
        return goals_to_record

    def _get_best_goals(self, all_metrics):
        max_rate = max(m["success_rate"] for m in all_metrics.values())
        return [g for g, m in all_metrics.items() if m["success_rate"] == max_rate]
        
    def close(self):
        self.evaluator.close()
        self.video_recorder.close()
