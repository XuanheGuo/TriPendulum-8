"""Reward and success logic based on absolute pendulum poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.angle_utils import angle_error


@dataclass
class RewardConfig:
    w_pose: float = 4.0
    w_vel: float = 0.05
    w_act: float = 0.001
    w_track: float = 1.0
    w_spin: float = 0.02
    w_delta_a: float = 0.01
    lambda_x: float = 1.0
    lambda_omega: float = 1.0
    lambda_delta_a: float = 1.0
    success_bonus: float = 50.0
    pose_threshold: float = 0.08
    omega_threshold: float = 1.0
    stable_x_threshold: float = 1.8
    stable_steps_required: int = 25


def pose_error_components(theta_abs, theta_goal_abs):
    err = angle_error(theta_abs, theta_goal_abs)
    return 1.0 - np.cos(err)


def compute_reward(
    *,
    goal_name,
    theta_abs,
    theta_goal_abs,
    q_relative,
    omega,
    action,
    prev_action,
    x,
    x_max,
    stable_steps,
    track_collision,
    cfg: RewardConfig,
):
    theta_abs = np.asarray(theta_abs, dtype=np.float64)
    theta_goal_abs = np.asarray(theta_goal_abs, dtype=np.float64)
    q_relative = np.asarray(q_relative, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    prev_action = np.asarray(prev_action, dtype=np.float64)

    pose_terms = pose_error_components(theta_abs, theta_goal_abs)
    r_pose = float(np.sum(pose_terms))
    r_vel = float(np.sum(np.square(omega)))
    r_act = float(np.sum(np.square(action)))
    r_track = float(cfg.lambda_x * (abs(float(x)) / float(x_max)) ** 2)
    r_spin = float(cfg.lambda_omega * np.sum(np.square(omega)))
    r_delta_a = float(cfg.lambda_delta_a * np.sum(np.square(action - prev_action)))

    pose_ready = r_pose < cfg.pose_threshold
    velocity_ready = float(np.max(np.abs(omega))) < cfg.omega_threshold
    cart_ready = abs(float(x)) < cfg.stable_x_threshold
    stable_now = bool(pose_ready and velocity_ready and cart_ready and not track_collision)
    next_stable_steps = stable_steps + 1 if stable_now else 0
    success = next_stable_steps >= cfg.stable_steps_required
    success_bonus = cfg.success_bonus if success else 0.0

    reward = (
        -cfg.w_pose * r_pose
        - cfg.w_vel * r_vel
        - cfg.w_act * r_act
        - cfg.w_track * r_track
        - cfg.w_spin * r_spin
        - cfg.w_delta_a * r_delta_a
        + success_bonus
    )

    info = {
        "goal_name": goal_name,
        "theta_abs": theta_abs.copy(),
        "theta_goal_abs": theta_goal_abs.copy(),
        "q_relative": q_relative.copy(),
        "r_pose": r_pose,
        "r_vel": r_vel,
        "r_act": r_act,
        "r_track": r_track,
        "r_spin": r_spin,
        "r_delta_a": r_delta_a,
        "success": success,
        "stable_steps": next_stable_steps,
        "track_collision": bool(track_collision),
    }
    return float(reward), success, next_stable_steps, info
