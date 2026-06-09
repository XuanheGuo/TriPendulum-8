"""Stable-Baselines3 callbacks."""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np
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
    def __init__(self, window_size=100, verbose=0):
        super().__init__(verbose)
        self.window_size = int(window_size)
        self.success_history = deque(maxlen=self.window_size)
        self.collision_history = deque(maxlen=self.window_size)
        self.goal_success_history = defaultdict(lambda: deque(maxlen=self.window_size))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            # Monitor adds "episode" only on the terminal step. Logging here
            # avoids treating every non-terminal environment step as a failure.
            if "episode" not in info:
                continue

            success = float(info.get("success", False))
            collision = float(info.get("track_collision", False))
            goal_name = str(info.get("goal_name", "unknown"))

            self.success_history.append(success)
            self.collision_history.append(collision)
            self.goal_success_history[goal_name].append(success)

            self.logger.record_mean("rollout/episode_success", success)
            self.logger.record_mean("rollout/episode_track_collision", collision)
            self.logger.record_mean("rollout/terminal_abs_x", abs(float(info.get("x", 0.0))))
            self.logger.record_mean("rollout/max_abs_x_mean", float(info.get("max_abs_x", 0.0)))
            self.logger.record(
                f"rollout/success_rate_{self.window_size}",
                sum(self.success_history) / len(self.success_history),
            )
            self.logger.record(
                f"rollout/track_collision_rate_{self.window_size}",
                sum(self.collision_history) / len(self.collision_history),
            )
            goal_history = self.goal_success_history[goal_name]
            self.logger.record(
                f"goal_success/{goal_name}",
                sum(goal_history) / len(goal_history),
            )
        return True


class AutoCurriculumCallback(BaseCallback):
    """Advance through ordered goals after deterministic evaluation passes."""

    def __init__(
        self,
        *,
        curriculum_state,
        env_config: dict,
        reward_config: dict,
        eval_freq: int = 25000,
        n_eval_episodes: int = 5,
        max_steps: int = 1000,
        success_threshold: float = 0.8,
        max_pose_error: float = 0.25,
        require_pose_error: bool = False,
        pose_error_key: str = "avg_final_pose_error",
        max_collision_rate: float = 0.1,
        consecutive_passes_required: int = 2,
        min_steps_per_goal: int = 25000,
        eval_seed: int | None = 10000,
        checkpoint_dir: str = "checkpoints",
        algorithm: str = "sac",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.state = curriculum_state
        self.env_config = deepcopy(env_config)
        self.reward_config = deepcopy(reward_config)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.max_steps = int(max_steps)
        self.success_threshold = float(success_threshold)
        self.max_pose_error = float(max_pose_error)
        self.require_pose_error = bool(require_pose_error)
        self.pose_error_key = pose_error_key
        self.max_collision_rate = float(max_collision_rate)
        self.consecutive_passes_required = int(consecutive_passes_required)
        self.min_steps_per_goal = int(min_steps_per_goal)
        self.eval_seed = None if eval_seed is None else int(eval_seed)
        self.checkpoint_dir = checkpoint_dir
        self.algorithm = algorithm.lower()
        self.last_eval_timestep = 0

    def _on_training_start(self) -> None:
        self._update_training_goals()
        if self.state.goal_start_timestep == 0:
            self.state.goal_start_timestep = int(self.num_timesteps)
            self.state.save()

    def _on_step(self) -> bool:
        if self.state.completed or self.eval_freq <= 0:
            return True
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        if self.num_timesteps - self.state.goal_start_timestep < self.min_steps_per_goal:
            return True

        self.last_eval_timestep = self.num_timesteps
        goal = self.state.current_goal
        evaluation_env_config = deepcopy(self.env_config)
        evaluation_env_config["curriculum_initial_pose_fraction"] = self.state.current_initial_pose_fraction
        metrics = evaluate_goal_metrics(
            self.model,
            goals=[goal],
            episodes_per_goal=self.n_eval_episodes,
            deterministic=True,
            env_config=evaluation_env_config,
            reward_config=self.reward_config,
            max_steps=self.max_steps,
            base_seed=self.eval_seed,
        )[goal]
        pass_checks = {
            "success_rate": metrics["success_rate"] >= self.success_threshold,
            "track_collision_rate": metrics["track_collision_rate"] <= self.max_collision_rate,
        }
        if self.require_pose_error:
            pass_checks[self.pose_error_key] = metrics.get(self.pose_error_key, float("inf")) <= self.max_pose_error
        passed = bool(all(pass_checks.values()))
        metrics["pass_checks"] = pass_checks
        metrics["pass_thresholds"] = {
            "success_threshold": self.success_threshold,
            "max_collision_rate": self.max_collision_rate,
            "require_pose_error": self.require_pose_error,
            "pose_error_key": self.pose_error_key,
            "max_pose_error": self.max_pose_error,
        }
        self.state.register_evaluation(self.num_timesteps, metrics, passed)

        self.logger.record("curriculum/stage_id", self.state.stage_id)
        self.logger.record("curriculum/current_goal_index", self.state.current_index)
        self.logger.record("curriculum/difficulty_stage", self.state.difficulty_stage)
        self.logger.record("curriculum/initial_pose_fraction", self.state.current_initial_pose_fraction)
        self.logger.record("curriculum/eval_success_rate", metrics["success_rate"])
        self.logger.record("curriculum/eval_pose_error", metrics["avg_pose_error"])
        self.logger.record("curriculum/eval_final_pose_error", metrics.get("avg_final_pose_error", metrics["avg_pose_error"]))
        self.logger.record("curriculum/eval_collision_rate", metrics["track_collision_rate"])
        self.logger.record("curriculum/consecutive_passes", self.state.consecutive_passes)
        self.logger.record("curriculum/passed", float(passed))

        if self.verbose:
            print(
                f"Curriculum goal {goal}: success={metrics['success_rate']:.2f}, "
                f"pose={metrics['avg_pose_error']:.3f}, final_pose={metrics.get('avg_final_pose_error', 0):.3f}, "
                f"collision={metrics['track_collision_rate']:.2f}, passed={passed}, checks={pass_checks}, "
                f"passes={self.state.consecutive_passes}/{self.consecutive_passes_required}"
            )

        if self.state.consecutive_passes >= self.consecutive_passes_required:
            completed_goal = goal
            self.model.save(
                os.path.join(
                    self.checkpoint_dir,
                    f"{self.algorithm}_curriculum_goal_{completed_goal}_step_{self.num_timesteps}.zip",
                )
            )
            advance_type = self.state.advance(self.num_timesteps)
            self._update_training_goals()
            if self.verbose:
                if advance_type == "difficulty":
                    print(
                        f"Curriculum difficulty advanced for {self.state.current_goal}: "
                        f"{self.state.difficulty_label}, initial_pose_fraction="
                        f"{self.state.current_initial_pose_fraction:.2f}"
                    )
                elif advance_type == "goal":
                    print(
                        f"Curriculum advanced: {completed_goal} -> {self.state.current_goal}; "
                        f"training goals={self.state.training_goals}"
                    )
                else:
                    print("Curriculum complete: all ordered goals passed.")
        return True

    def _update_training_goals(self) -> None:
        self.training_env.env_method(
            "set_curriculum_goals",
            self.state.training_goals,
            self.state.current_goal,
            self.state.current_initial_pose_fraction,
        )


class DiagnosticVideoCallback(BaseCallback):
    """Evaluate current policy and render the most useful diagnostic goals."""

    def __init__(
        self,
        *,
        algorithm: str,
        env_config: dict | None = None,
        reward_config: dict | None = None,
        curriculum_config: dict | None = None,
        curriculum_state=None,
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
        self.curriculum_state = curriculum_state
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
        stage_id = get_current_stage(self.curriculum_config, self.curriculum_state)
        stage_goals = get_current_goals(self.curriculum_config, self.curriculum_state)
        curriculum_enabled = bool(self.curriculum_config.get("enabled", False))
        eval_goals = self._select_eval_goals(curriculum_enabled, stage_goals)
        evaluation_env_config = deepcopy(self.env_config)
        if self.curriculum_state is not None:
            evaluation_env_config["curriculum_initial_pose_fraction"] = (
                self.curriculum_state.current_initial_pose_fraction
            )
        goal_metrics = evaluate_goal_metrics(
            self.model,
            goals=eval_goals,
            episodes_per_goal=self.n_eval_episodes,
            deterministic=True,
            env_config=evaluation_env_config,
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
            if self.curriculum_state is not None:
                return [self.curriculum_state.current_goal]
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
            "current_goal": self.curriculum_state.current_goal if self.curriculum_state is not None else None,
            "difficulty_label": self.curriculum_state.difficulty_label if self.curriculum_state is not None else None,
            "initial_pose_fraction": (
                self.curriculum_state.current_initial_pose_fraction if self.curriculum_state is not None else 0.0
            ),
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
        if self.curriculum_state is not None and goal == self.curriculum_state.current_goal:
            env_cfg["curriculum_initial_pose_fraction"] = self.curriculum_state.current_initial_pose_fraction
        env = TriPendulumGoalEnv(env_cfg, render_mode="rgb_array")

        frames = []
        obs, _ = env.reset(goal=goal)
        episode_reward = 0.0
        max_abs_x = 0.0
        final_pose_error = None
        track_collision = False
        success = False
        overspin = False
        max_boundary_penalty = 0.0
        max_outward_penalty = 0.0
        absolute_actions = []
        signed_actions = []
        max_abs_omega = 0.0
        swing_energy_rewards = []
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
                max_boundary_penalty = max(max_boundary_penalty, float(info.get("r_boundary", 0.0)))
                max_outward_penalty = max(max_outward_penalty, float(info.get("r_outward", 0.0)))
                absolute_actions.append(float(abs(action[0])))
                signed_actions.append(float(action[0]))
                omega_abs = info.get("omega_abs", [])
                if len(omega_abs):
                    max_abs_omega = max(max_abs_omega, max(abs(float(value)) for value in omega_abs))
                swing_energy_rewards.append(float(info.get("r_swing_energy", 0.0)))

                if terminated or truncated:
                    break
        finally:
            env.close()

        base_name = f"{self.algorithm}_step_{self.num_timesteps}_stage_{stage_id}_goal_{goal}"
        video_path = os.path.join(self.save_dir, f"{base_name}.mp4")
        json_path = os.path.join(self.save_dir, f"{base_name}.json")

        if frames:
            save_mp4(frames, video_path, fps=self.fps)

        action_reversals = sum(
            1
            for previous, current in zip(signed_actions, signed_actions[1:])
            if abs(previous) > 1.0 and abs(current) > 1.0 and previous * current < 0.0
        )
        diagnostics = {
            "algorithm": self.algorithm,
            "timestep": int(self.num_timesteps),
            "curriculum_enabled": bool(curriculum_enabled),
            "stage_id": int(stage_id),
            "current_goal": self.curriculum_state.current_goal if self.curriculum_state is not None else goal,
            "difficulty_label": self.curriculum_state.difficulty_label if self.curriculum_state is not None else None,
            "initial_pose_fraction": (
                self.curriculum_state.current_initial_pose_fraction if self.curriculum_state is not None else 0.0
            ),
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
            "max_boundary_penalty": max_boundary_penalty,
            "max_outward_penalty": max_outward_penalty,
            "mean_abs_action": sum(absolute_actions) / len(absolute_actions) if absolute_actions else 0.0,
            "max_abs_action": max(absolute_actions) if absolute_actions else 0.0,
            "std_action": float(np.std(signed_actions)) if signed_actions else 0.0,
            "action_reversal_rate": (
                action_reversals / max(len(signed_actions) - 1, 1) if signed_actions else 0.0
            ),
            "max_abs_omega": max_abs_omega,
            "mean_swing_energy_reward": (
                sum(swing_energy_rewards) / len(swing_energy_rewards) if swing_energy_rewards else 0.0
            ),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

        if self.verbose:
            print(f"Saved diagnostic video: {video_path}")
