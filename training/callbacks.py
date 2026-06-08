"""Stable-Baselines3 callbacks."""

from __future__ import annotations

import json
import os
from copy import deepcopy

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from envs.goals import GOAL_NAMES
from envs.tripendulum_env import TriPendulumGoalEnv
from evaluation.evaluate import evaluate_goal_metrics
from training.curriculum import get_current_goals, get_current_stage
from utils.video_utils import save_mp4


def make_checkpoint_callback(save_freq: int, checkpoint_dir: str, name_prefix: str, save_replay_buffer: bool = False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix=name_prefix,
        save_replay_buffer=save_replay_buffer,
        save_vecnormalize=True,
    )


class EpisodeInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "success" in info:
                self.logger.record("rollout/success", float(info["success"]))
            if "track_collision" in info:
                self.logger.record("rollout/track_collision", float(info["track_collision"]))
        return True


class DiagnosticVideoCallback(BaseCallback):
    """Evaluate current policy and render the most useful diagnostic goals."""

    def __init__(
        self,
        *,
        algorithm: str,
        env_config: dict | None = None,
        reward_config: dict | None = None,
        curriculum_config: dict | None = None,
        eval_freq: int = 50000,
        max_steps: int = 1000,
        n_eval_episodes: int = 3,
        mode: str = "curriculum_worst",
        worst_k: int = 2,
        goals: list[str] | tuple[str, ...] = ("DDD", "DUD", "UUU"),
        fallback_goals: list[str] | tuple[str, ...] = ("DDD", "UUU"),
        save_dir: str = "videos/diagnostics",
        fps: int = 30,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.algorithm = algorithm.lower()
        self.env_config = deepcopy(env_config or {})
        self.reward_config = deepcopy(reward_config or {})
        self.curriculum_config = deepcopy(curriculum_config or {})
        self.eval_freq = int(eval_freq)
        self.max_steps = int(max_steps)
        self.n_eval_episodes = int(n_eval_episodes)
        self.mode = mode
        self.worst_k = int(worst_k)
        self.goals = tuple(goals)
        self.fallback_goals = tuple(fallback_goals)
        self.save_dir = save_dir
        self.fps = int(fps)
        self.last_eval_timestep = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True

        self.last_eval_timestep = self.num_timesteps
        stage_id = get_current_stage(self.curriculum_config)
        stage_goals = get_current_goals(self.curriculum_config)
        curriculum_enabled = bool(self.curriculum_config.get("enabled", False))
        eval_goals = self._select_eval_goals(curriculum_enabled, stage_goals)
        goal_metrics = evaluate_goal_metrics(
            self.model,
            goals=eval_goals,
            episodes_per_goal=self.n_eval_episodes,
            deterministic=True,
            env_config=self.env_config,
            reward_config=self.reward_config,
            max_steps=self.max_steps,
        )
        video_goals, selection_reason = self._select_video_goals(
            goal_metrics=goal_metrics,
            curriculum_enabled=curriculum_enabled,
        )
        self._save_report(curriculum_enabled, stage_id, stage_goals, eval_goals, video_goals, goal_metrics)

        for goal in video_goals:
            self._render_goal(
                goal=goal,
                stage_id=stage_id,
                stage_goals=stage_goals,
                curriculum_enabled=curriculum_enabled,
                selection_reason=selection_reason,
            )
        return True

    def _select_eval_goals(self, curriculum_enabled: bool, stage_goals: list[str]) -> list[str]:
        if self.mode == "fixed":
            return list(self.goals or self.fallback_goals)
        if self.mode == "curriculum":
            return list(stage_goals)
        if self.mode == "all":
            return list(GOAL_NAMES)
        if self.mode == "worst":
            return list(GOAL_NAMES)
        if self.mode == "curriculum_worst":
            return list(stage_goals if curriculum_enabled else GOAL_NAMES)
        raise ValueError(f"Unknown diagnostic video mode {self.mode!r}")

    def _select_video_goals(self, goal_metrics: dict, curriculum_enabled: bool) -> tuple[list[str], str]:
        if self.mode == "fixed":
            return list(self.goals or self.fallback_goals), "fixed_config_goal"
        if self.mode == "curriculum":
            return list(goal_metrics.keys()), "current_curriculum_goal"
        if self.mode == "all":
            return list(goal_metrics.keys()), "all_goals"
        if self.mode == "worst":
            return self._worst_goals(goal_metrics), "worst_goal_over_all"
        if self.mode == "curriculum_worst":
            reason = "worst_goal_in_current_stage" if curriculum_enabled else "worst_goal_over_all"
            return self._worst_goals(goal_metrics), reason
        raise ValueError(f"Unknown diagnostic video mode {self.mode!r}")

    def _worst_goals(self, goal_metrics: dict) -> list[str]:
        ranked = sorted(
            goal_metrics,
            key=lambda goal: (
                goal_metrics[goal]["success_rate"],
                -goal_metrics[goal]["avg_pose_error"],
                -goal_metrics[goal]["track_collision_rate"],
                goal_metrics[goal]["avg_reward"],
            ),
        )
        return ranked[: max(1, min(self.worst_k, len(ranked)))]

    def _save_report(
        self,
        curriculum_enabled: bool,
        stage_id: int,
        stage_goals: list[str],
        eval_goals: list[str],
        video_goals: list[str],
        goal_metrics: dict,
    ) -> None:
        report = {
            "timestep": int(self.num_timesteps),
            "mode": self.mode,
            "curriculum_enabled": bool(curriculum_enabled),
            "stage_id": int(stage_id),
            "stage_goals": list(stage_goals),
            "evaluated_goals": list(eval_goals),
            "video_goals": list(video_goals),
            "goal_metrics": goal_metrics,
        }
        report_path = os.path.join(self.save_dir, f"diagnostic_step_{self.num_timesteps}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _render_goal(
        self,
        *,
        goal: str,
        stage_id: int,
        stage_goals: list[str],
        curriculum_enabled: bool,
        selection_reason: str,
    ) -> None:
        env_cfg = deepcopy(self.env_config)
        env_cfg["render_mode"] = "rgb_array"
        env_cfg["reward"] = deepcopy(self.reward_config)
        env = TriPendulumGoalEnv(env_cfg, render_mode="rgb_array")

        frames = []
        obs, _ = env.reset(goal=goal)
        episode_reward = 0.0
        max_abs_x = 0.0
        final_pose_error = None
        track_collision = False
        success = False
        overspin = False
        episode_length = 0

        try:
            for step in range(self.max_steps):
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += float(reward)
                episode_length = step + 1
                max_abs_x = max(max_abs_x, abs(float(info.get("x", 0.0))))
                final_pose_error = float(info.get("r_pose", 0.0))
                track_collision = track_collision or bool(info.get("track_collision", False))
                success = success or bool(info.get("success", False))
                overspin = overspin or bool(info.get("omega_over_limit", False))

                if terminated or truncated:
                    break
        finally:
            env.close()

        base_name = f"{self.algorithm}_step_{self.num_timesteps}_stage_{stage_id}_goal_{goal}"
        video_path = os.path.join(self.save_dir, f"{base_name}.mp4")
        json_path = os.path.join(self.save_dir, f"{base_name}.json")

        if frames:
            save_mp4(frames, video_path, fps=self.fps)

        diagnostics = {
            "algorithm": self.algorithm,
            "timestep": int(self.num_timesteps),
            "curriculum_enabled": bool(curriculum_enabled),
            "stage_id": int(stage_id),
            "stage_goals": list(stage_goals),
            "goal": goal,
            "selection_reason": selection_reason,
            "success": bool(success),
            "episode_reward": episode_reward,
            "episode_length": int(episode_length),
            "final_pose_error": final_pose_error,
            "max_abs_x": max_abs_x,
            "track_collision": bool(track_collision),
            "overspin": bool(overspin),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

        if self.verbose:
            print(f"Saved diagnostic video: {video_path}")
