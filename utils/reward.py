"""Reward and success logic based on absolute pendulum poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.angle_utils import angle_error


@dataclass
class RewardConfig:
    w_pose: float = 4.0
    w_vel: float = 0.005
    w_act: float = 0.0002
    w_track: float = 6.0
    w_boundary: float = 25.0
    w_outward: float = 2.0
    w_pose_progress: float = 8.0
    w_spin: float = 0.002
    w_delta_a: float = 0.001
    lambda_x: float = 1.0
    lambda_omega: float = 1.0
    lambda_delta_a: float = 1.0
    soft_boundary_ratio: float = 0.7
    success_bonus: float = 50.0
    pose_threshold: float = 0.08
    omega_threshold: float = 1.0
    stable_x_threshold: float = 3.2
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
    x_dot,
    x_max,
    stable_steps,
    track_collision,
    prev_pose_error,
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
    boundary_progress = max(
        0.0,
        (abs(float(x)) / float(x_max) - cfg.soft_boundary_ratio) / (1.0 - cfg.soft_boundary_ratio),
    )
    r_boundary = float(boundary_progress**2)
    outward_speed = max(0.0, float(x) * float(x_dot) / float(x_max))
    r_outward = float(outward_speed**2)
    r_pose_progress = float(np.clip(float(prev_pose_error) - r_pose, -1.0, 1.0))
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
        - cfg.w_boundary * r_boundary
        - cfg.w_outward * r_outward
        - cfg.w_spin * r_spin
        - cfg.w_delta_a * r_delta_a
        + cfg.w_pose_progress * r_pose_progress
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
        "r_boundary": r_boundary,
        "r_outward": r_outward,
        "r_pose_progress": r_pose_progress,
        "r_spin": r_spin,
        "r_delta_a": r_delta_a,
        "success": success,
        "stable_steps": next_stable_steps,
        "track_collision": bool(track_collision),
    }
    return float(reward), success, next_stable_steps, info
