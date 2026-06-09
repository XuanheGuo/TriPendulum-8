"""Goal-conditioned single-actuator triple pendulum cart environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from envs.goals import GOAL_NAMES, get_goal, sample_goal
from utils.angle_utils import absolute_to_relative, relative_to_absolute, relative_velocity_to_absolute
from utils.reward import RewardConfig, compute_reward


@dataclass
class TriPendulumConfig:
    max_episode_steps: int = 1000
    frame_skip: int = 4
    render_mode: str | None = None
    seed: int | None = None
    x_max: float = 4.8
    f_max: float = 40.0
    omega_limit: float = 30.0
    stable_x_threshold: float = 3.2
    pose_threshold: float = 0.08
    omega_threshold: float = 1.0
    stable_steps_required: int = 25
    collision_penalty: float = 250.0
    random_initial_state: bool = True
    initial_cart_position_noise: float = 0.05
    initial_cart_velocity_noise: float = 0.05
    initial_angle_noise: float = 0.05
    initial_velocity_noise: float = 0.05
    primary_goal_probability: float = 0.8
    primary_goal_near_target_probability: float = 0.15
    near_target_angle_noise: float = 0.12
    near_target_velocity_noise: float = 0.1
    allowed_goals: tuple[str, ...] = field(default_factory=lambda: GOAL_NAMES)
    reward: RewardConfig = field(default_factory=RewardConfig)


class TriPendulumGoalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config: TriPendulumConfig | dict | None = None, xml_path: str | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = self._coerce_config(config)
        if render_mode is not None:
            self.config.render_mode = render_mode

        xml_path = xml_path or os.path.join(os.path.dirname(__file__), "mujoco_model.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.actuator_ctrlrange[0] = np.array([-self.config.f_max, self.config.f_max], dtype=np.float64)

        self.action_space = spaces.Box(
            low=np.array([-self.config.f_max], dtype=np.float32),
            high=np.array([self.config.f_max], dtype=np.float32),
            dtype=np.float32,
        )
        high = np.full(20, np.inf, dtype=np.float32)
        high[2:14] = 1.0
        high[-3:] = 1.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.goal_name = "DDD"
        self.goal_binary = np.zeros(3, dtype=np.float32)
        self.goal_abs_angles = np.zeros(3, dtype=np.float64)
        self.primary_goal = None
        self.stable_steps = 0
        self.step_count = 0
        self.prev_action = np.zeros(1, dtype=np.float64)
        self.prev_pose_error = 0.0
        self.max_abs_x = 0.0
        self.viewer = None
        self.renderer = None

    @staticmethod
    def _coerce_config(config):
        if config is None:
            return TriPendulumConfig()
        if isinstance(config, TriPendulumConfig):
            return config
        env_cfg = dict(config)
        reward_cfg = env_cfg.pop("reward", None) or {}
        cfg = TriPendulumConfig(**env_cfg)
        cfg.reward = RewardConfig(**reward_cfg)
        cfg.reward.pose_threshold = cfg.pose_threshold
        cfg.reward.omega_threshold = cfg.omega_threshold
        cfg.reward.stable_x_threshold = cfg.stable_x_threshold
        cfg.reward.stable_steps_required = cfg.stable_steps_required
        return cfg

    def reset(self, *, seed=None, options=None, goal=None):
        super().reset(seed=seed)
        options = options or {}
        goal = goal or options.get("goal")
        if goal is None:
            if self.primary_goal is not None and self.np_random.random() < self.config.primary_goal_probability:
                selected = get_goal(self.primary_goal)
            else:
                fallback_goals = tuple(goal for goal in self.config.allowed_goals if goal != self.primary_goal)
                selected = sample_goal(
                    self.np_random,
                    fallback_goals or self.config.allowed_goals,
                )
        else:
            selected = get_goal(goal)
        self.goal_name = selected.name
        self.goal_binary = selected.binary
        self.goal_abs_angles = selected.abs_angles

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if self.config.random_initial_state:
            self.data.qpos[0] = self.np_random.uniform(
                -self.config.initial_cart_position_noise,
                self.config.initial_cart_position_noise,
            )
            self.data.qpos[1:4] = self.np_random.uniform(
                -self.config.initial_angle_noise,
                self.config.initial_angle_noise,
                size=3,
            )
            self.data.qvel[0] = self.np_random.uniform(
                -self.config.initial_cart_velocity_noise,
                self.config.initial_cart_velocity_noise,
            )
            self.data.qvel[1:4] = self.np_random.uniform(
                -self.config.initial_velocity_noise,
                self.config.initial_velocity_noise,
                size=3,
            )
        if (
            self.primary_goal is not None
            and self.goal_name == self.primary_goal
            and self.goal_name != "DDD"
            and self.np_random.random() < self.config.primary_goal_near_target_probability
        ):
            target_q = absolute_to_relative(self.goal_abs_angles)
            self.data.qpos[1:4] = target_q + self.np_random.uniform(
                -self.config.near_target_angle_noise,
                self.config.near_target_angle_noise,
                size=3,
            )
            self.data.qvel[1:4] = self.np_random.uniform(
                -self.config.near_target_velocity_noise,
                self.config.near_target_velocity_noise,
                size=3,
            )
        mujoco.mj_forward(self.model, self.data)
        q_relative = np.array(self.data.qpos[1:4], dtype=np.float64)
        theta_abs = relative_to_absolute(q_relative)

        self.stable_steps = 0
        self.step_count = 0
        self.prev_action = np.zeros(1, dtype=np.float64)
        self.prev_pose_error = float(np.sum(1.0 - np.cos(theta_abs - self.goal_abs_angles)))
        self.max_abs_x = abs(float(self.data.qpos[0]))
        return self._get_obs(), self._base_info(track_collision=False, success=False)

    def set_allowed_goals(self, goals):
        goals = tuple(str(goal).upper() for goal in goals)
        unknown = [goal for goal in goals if goal not in GOAL_NAMES]
        if not goals or unknown:
            raise ValueError(f"Invalid allowed goals: {goals}")
        self.config.allowed_goals = goals

    def set_curriculum_goals(self, goals, primary_goal=None):
        self.set_allowed_goals(goals)
        if primary_goal is not None and primary_goal not in self.config.allowed_goals:
            raise ValueError(f"Primary goal {primary_goal} is not in allowed goals")
        self.primary_goal = primary_goal

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(1)
        clipped_action = np.clip(action, -self.config.f_max, self.config.f_max)
        self.data.ctrl[0] = clipped_action[0]

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        x = float(self.data.qpos[0])
        x_dot = float(self.data.qvel[0])
        self.max_abs_x = max(self.max_abs_x, abs(x))
        q_relative = np.array(self.data.qpos[1:4], dtype=np.float64)
        q_dot_relative = np.array(self.data.qvel[1:4], dtype=np.float64)
        omega = relative_velocity_to_absolute(q_dot_relative)
        theta_abs = relative_to_absolute(q_relative)

        track_collision = abs(x) > self.config.x_max
        reward, success, self.stable_steps, info = compute_reward(
            goal_name=self.goal_name,
            theta_abs=theta_abs,
            theta_goal_abs=self.goal_abs_angles,
            q_relative=q_relative,
            omega=omega,
            action=clipped_action,
            prev_action=self.prev_action,
            x=x,
            x_dot=x_dot,
            x_max=self.config.x_max,
            stable_steps=self.stable_steps,
            track_collision=track_collision,
            prev_pose_error=self.prev_pose_error,
            cfg=self.config.reward,
        )
        if track_collision:
            reward -= self.config.collision_penalty

        omega_over_limit = bool(np.max(np.abs(omega)) > self.config.omega_limit)
        terminated = bool(track_collision or omega_over_limit or success)
        truncated = bool(self.step_count >= self.config.max_episode_steps and not terminated)

        info.update(
            {
                "x": x,
                "x_max": self.config.x_max,
                "max_abs_x": self.max_abs_x,
                "action": clipped_action.copy(),
                "omega_abs": omega.copy(),
                "q_dot_relative": q_dot_relative.copy(),
                "omega_over_limit": omega_over_limit,
                "time_limit": truncated,
            }
        )
        self.prev_action = clipped_action.copy()
        self.prev_pose_error = float(info["r_pose"])
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        x = float(self.data.qpos[0])
        x_dot = float(self.data.qvel[0])
        q = np.array(self.data.qpos[1:4], dtype=np.float64)
        theta_abs = relative_to_absolute(q)
        omega = relative_velocity_to_absolute(np.array(self.data.qvel[1:4], dtype=np.float64))
        obs = np.array(
            [
                x,
                x_dot,
                np.sin(q[0]),
                np.cos(q[0]),
                np.sin(q[1]),
                np.cos(q[1]),
                np.sin(q[2]),
                np.cos(q[2]),
                np.sin(theta_abs[0]),
                np.cos(theta_abs[0]),
                np.sin(theta_abs[1]),
                np.cos(theta_abs[1]),
                np.sin(theta_abs[2]),
                np.cos(theta_abs[2]),
                omega[0],
                omega[1],
                omega[2],
                self.goal_binary[0],
                self.goal_binary[1],
                self.goal_binary[2],
            ],
            dtype=np.float32,
        )
        return obs

    def _base_info(self, track_collision, success):
        q_relative = np.array(self.data.qpos[1:4], dtype=np.float64)
        theta_abs = relative_to_absolute(q_relative)
        return {
            "goal_name": self.goal_name,
            "theta_abs": theta_abs,
            "theta_goal_abs": self.goal_abs_angles.copy(),
            "q_relative": q_relative,
            "success": bool(success),
            "stable_steps": self.stable_steps,
            "track_collision": bool(track_collision),
            "x": float(self.data.qpos[0]),
            "x_max": self.config.x_max,
            "max_abs_x": self.max_abs_x,
        }

    def render(self):
        if self.config.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera="fixed")
            return self.renderer.render()
        if self.config.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer

                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        return None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
