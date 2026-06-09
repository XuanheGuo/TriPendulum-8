"""Render an MJX PPO checkpoint through the original MuJoCo environment."""

from __future__ import annotations

import argparse

import jax
import numpy as np
import yaml

from envs.tripendulum_env import TriPendulumGoalEnv
from mjx_backend.env import TriPendulumMJXEnv
from mjx_backend.evaluate import load_policy
from utils.video_utils import save_mp4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mjx_ppo.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--goal", default="UUU")
    parser.add_argument("--output", default="videos/mjx_ppo_UUU.mp4")
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    mjx_env = TriPendulumMJXEnv(config["env"])
    policy = load_policy(config, args.checkpoint, mjx_env)
    env_cfg = dict(config["env"])
    env_cfg.pop("initial_pose_fractions", None)
    env_cfg.pop("initial_pose_probabilities", None)
    env = TriPendulumGoalEnv(env_cfg, render_mode="rgb_array")
    obs, _ = env.reset(goal=args.goal)
    frames = []
    rng = jax.random.PRNGKey(0)
    try:
        for _ in range(args.max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            rng, action_rng = jax.random.split(rng)
            normalized_action, _ = policy(np.asarray(obs), action_rng)
            physical_action = np.asarray(normalized_action) * float(config["env"].get("f_max", 40.0))
            obs, _, terminated, truncated, _ = env.step(physical_action)
            if terminated or truncated:
                break
    finally:
        env.close()
    save_mp4(frames, args.output, fps=30)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
