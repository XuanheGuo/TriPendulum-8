"""Native MuJoCo MJX environment for goal-conditioned TriPendulum-8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dataclasses

import jax
import jax.numpy as jnp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from envs.goals import GOAL_NAMES as _GOAL_NAMES
from envs.goals import GOAL_BINARY as _GOAL_BINARY_NP
from utils.reward import RewardConfig as _RewardConfig

GOAL_BINARY = jnp.asarray(
    [_GOAL_BINARY_NP[name] for name in _GOAL_NAMES], dtype=jnp.float32
)

_RC_DEFAULTS = {f.name: f.default for f in dataclasses.fields(_RewardConfig)}


def wrap_angle(angle: jax.Array) -> jax.Array:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def relative_to_absolute(q: jax.Array) -> jax.Array:
    return wrap_angle(jnp.cumsum(q, axis=-1))


def absolute_to_relative(theta: jax.Array) -> jax.Array:
    return wrap_angle(jnp.asarray([theta[0], theta[1] - theta[0], theta[2] - theta[1]]))


class TriPendulumMJXEnv(mjx_env.MjxEnv):
    """Single-cart-actuator triple pendulum, vectorized externally by Brax."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        self.config = config or {}
        self._xml_path = str(Path(__file__).resolve().parents[1] / "envs" / "mujoco_model.xml")
        self.frame_skip = int(self.config.get("frame_skip", 4))
        env_config = config_dict.create(
            ctrl_dt=0.005 * self.frame_skip,
            sim_dt=0.005,
            episode_length=int(self.config.get("max_episode_steps", 1000)),
            action_repeat=1,
            impl="jax",
            naconmax=0,
            njmax=8,
        )
        super().__init__(env_config, config_overrides=kwargs or None)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        self.x_max = float(self.config.get("x_max", 4.8))
        self.f_max = float(self.config.get("f_max", 40.0))
        self.omega_limit = float(self.config.get("omega_limit", 30.0))
        self.stable_x_threshold = float(self.config.get("stable_x_threshold", 3.2))
        self.pose_threshold = float(self.config.get("pose_threshold", 0.08))
        self.omega_threshold = float(self.config.get("omega_threshold", 1.0))
        self.stable_steps_required = int(self.config.get("stable_steps_required", 25))
        self.post_success_steps = int(self.config.get("post_success_steps", 100))
        self.collision_penalty = float(self.config.get("collision_penalty", 250.0))
        self.initial_pose_fractions = jnp.asarray(
            self.config.get("initial_pose_fractions", [1.0, 0.5, 0.25, 0.0]), dtype=jnp.float32
        )
        self.initial_pose_probabilities = jnp.asarray(
            self.config.get("initial_pose_probabilities", [0.2, 0.2, 0.2, 0.4]), dtype=jnp.float32
        )
        self.initial_pose_probabilities /= jnp.sum(self.initial_pose_probabilities)
        self.reward_cfg = self.config.get("reward", {})
        _all = list(_GOAL_NAMES)
        allowed = self.config.get("allowed_goals", _all)
        self._allowed_indices = jnp.asarray([_all.index(g) for g in allowed], dtype=jnp.int32)

    @property
    def action_size(self) -> int:
        return 1

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng_goal, rng_fraction, rng_q, rng_qd, rng_cart = jax.random.split(rng, 5)
        slot = jax.random.randint(rng_goal, (), 0, self._allowed_indices.shape[0])
        goal_index = self._allowed_indices[slot]
        goal_binary = GOAL_BINARY[goal_index]
        goal_abs = goal_binary * jnp.pi
        fraction_index = jax.random.choice(
            rng_fraction,
            self.initial_pose_fractions.shape[0],
            p=self.initial_pose_probabilities,
        )
        pose_fraction = self.initial_pose_fractions[fraction_index]
        initial_abs = goal_abs * pose_fraction
        target_q = absolute_to_relative(initial_abs)

        difficulty = 1.0 - pose_fraction
        angle_noise = 0.02 + difficulty * 0.10
        velocity_noise = 0.02 + difficulty * 0.08
        q = jnp.asarray(self._mj_model.qpos0)
        q = q.at[0].set(jax.random.uniform(rng_cart, (), minval=-0.05, maxval=0.05))
        q = q.at[1:4].set(target_q + jax.random.uniform(rng_q, (3,), minval=-angle_noise, maxval=angle_noise))
        qd = jax.random.uniform(rng_qd, (self._mj_model.nv,), minval=-velocity_noise, maxval=velocity_noise)
        qd = qd.at[0].set(jnp.clip(qd[0], -0.05, 0.05))
        data = mjx_env.make_data(
            self.mj_model,
            qpos=q,
            qvel=qd,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)
        theta_abs = relative_to_absolute(q[1:4])
        pose_error = jnp.sum(1.0 - jnp.cos(theta_abs - goal_abs))
        info = {
            "goal_index": goal_index,
            "goal_binary": goal_binary,
            "goal_abs": goal_abs,
            "initial_pose_fraction": pose_fraction,
            "prev_action": jnp.zeros((1,), dtype=jnp.float32),
            "prev_pose_error": pose_error,
            "stable_steps": jnp.asarray(0, dtype=jnp.int32),
            "episode_success": jnp.asarray(False),
            "steps_after_success": jnp.asarray(0, dtype=jnp.int32),
        }
        metrics = self._empty_metrics(pose_error)
        obs = self._get_obs(data, goal_binary)
        return mjx_env.State(data, obs, jnp.asarray(0.0), jnp.asarray(0.0), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        normalized_action = jnp.clip(action, -1.0, 1.0)
        force = normalized_action * self.f_max
        data = mjx_env.step(self.mjx_model, state.data, force, self.n_substeps)
        q = data.qpos
        qd = data.qvel
        x, x_dot = q[0], qd[0]
        theta_abs = relative_to_absolute(q[1:4])
        omega_abs = jnp.cumsum(qd[1:4])
        goal_abs = state.info["goal_abs"]
        pose_terms = 1.0 - jnp.cos(theta_abs - goal_abs)
        r_pose = jnp.sum(pose_terms)
        r_vel = jnp.sum(jnp.square(omega_abs))
        r_act = jnp.sum(jnp.square(force))
        r_track = jnp.square(jnp.abs(x) / self.x_max)
        soft_boundary_ratio = self._r("soft_boundary_ratio", 0.7)
        boundary_progress = jnp.maximum(0.0, (jnp.abs(x) / self.x_max - soft_boundary_ratio) / (1.0 - soft_boundary_ratio))
        r_boundary = jnp.square(boundary_progress)
        r_outward = jnp.square(jnp.maximum(0.0, x * x_dot / self.x_max))
        r_pose_progress = jnp.clip(state.info["prev_pose_error"] - r_pose, -1.0, 1.0)
        up_mask = state.info["goal_binary"]
        r_upright_height = jnp.sum(up_mask * 0.5 * (1.0 - jnp.cos(theta_abs)))
        far_from_goal = jnp.clip(pose_terms / 2.0, 0.0, 1.0)
        speed_cap = self._r("swing_velocity_cap", 8.0)
        swing_gate = 1.0 - jnp.exp(-self._r("stability_pose_scale") * r_pose)
        r_swing_energy = jnp.sum(up_mask * far_from_goal * jnp.minimum(jnp.square(omega_abs), speed_cap**2)) * swing_gate
        centered = jnp.maximum(0.0, 1.0 - jnp.square(jnp.abs(x) / self.stable_x_threshold))
        r_stability = jnp.exp(
            -self._r("stability_pose_scale", 4.0) * r_pose
            - self._r("stability_velocity_scale", 0.05) * r_vel
        ) * centered
        r_delta_a = jnp.sum(jnp.square(force - state.info["prev_action"]))

        track_collision = jnp.abs(x) > self.x_max
        overspin = jnp.max(jnp.abs(omega_abs)) > self.omega_limit
        stable_now = (
            (r_pose < self.pose_threshold)
            & (jnp.max(jnp.abs(omega_abs)) < self.omega_threshold)
            & (jnp.abs(x) < self.stable_x_threshold)
            & (~track_collision)
        )
        stable_steps = jnp.where(stable_now, state.info["stable_steps"] + 1, 0)
        success_now = stable_steps >= self.stable_steps_required
        episode_success = state.info["episode_success"] | success_now
        steps_after_success = jnp.where(episode_success, state.info["steps_after_success"] + 1, 0)
        success_rollout_complete = episode_success & (steps_after_success >= self.post_success_steps)
        done = track_collision | overspin | success_rollout_complete
        success_bonus = jnp.where(stable_steps == self.stable_steps_required, self._r("success_bonus", 50.0), 0.0)

        reward = (
            -self._r("w_pose", 4.0) * r_pose
            - self._r("w_vel", 0.005) * r_vel
            - self._r("w_act", 0.0002) * r_act
            - self._r("w_track", 6.0) * r_track
            - self._r("w_boundary", 25.0) * r_boundary
            - self._r("w_outward", 2.0) * r_outward
            - self._r("w_spin", 0.002) * r_vel
            - self._r("w_delta_a", 0.0001) * r_delta_a
            + self._r("w_pose_progress", 8.0) * r_pose_progress
            + self._r("w_swing_energy", 0.03) * r_swing_energy
            + self._r("w_upright_height", 2.0) * r_upright_height
            + self._r("w_stability", 6.0) * r_stability
            + success_bonus
            - jnp.where(track_collision, self.collision_penalty, 0.0)
        )
        info = {
            **state.info,
            "prev_action": force,
            "prev_pose_error": r_pose,
            "stable_steps": stable_steps,
            "episode_success": episode_success,
            "steps_after_success": steps_after_success,
        }
        metrics = {
            "success": episode_success.astype(jnp.float32),
            "pose_error": r_pose,
            "track_collision": track_collision.astype(jnp.float32),
            "max_abs_x": jnp.abs(x),
            "energy": r_act,
            "initial_pose_fraction": state.info["initial_pose_fraction"],
            "reward": reward,
        }
        obs = self._get_obs(data, state.info["goal_binary"])
        return mjx_env.State(data, obs, reward, done.astype(jnp.float32), metrics, info)

    def _get_obs(self, data: mjx.Data, goal_binary: jax.Array) -> jax.Array:
        q = data.qpos
        qd = data.qvel
        relative = q[1:4]
        absolute = relative_to_absolute(relative)
        omega_abs = jnp.cumsum(qd[1:4])
        return jnp.concatenate(
            [
                q[0:1],
                qd[0:1],
                jnp.stack([jnp.sin(relative), jnp.cos(relative)], axis=-1).reshape(-1),
                jnp.stack([jnp.sin(absolute), jnp.cos(absolute)], axis=-1).reshape(-1),
                omega_abs,
                goal_binary,
            ]
        )

    def _r(self, name: str, default: float = None) -> float:
        if default is None:
            default = _RC_DEFAULTS.get(name, 0.0)
        return float(self.reward_cfg.get(name, default))

    @staticmethod
    def _empty_metrics(pose_error: jax.Array) -> dict[str, jax.Array]:
        zero = jnp.asarray(0.0)
        return {
            "success": zero,
            "pose_error": pose_error,
            "track_collision": zero,
            "max_abs_x": zero,
            "energy": zero,
            "initial_pose_fraction": zero,
            "reward": zero,
        }
