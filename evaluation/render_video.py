from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO, SAC

from envs.tripendulum_env import TriPendulumGoalEnv
from utils.video_utils import save_mp4


def load_model(path, env):
    try:
        return SAC.load(path, env=env)
    except Exception:
        return PPO.load(path, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--goal", default="UUU")
    parser.add_argument("--output", default="videos/sac_UUU.mp4")
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    env = TriPendulumGoalEnv(render_mode="rgb_array")
    model = load_model(args.model, env)
    obs, _ = env.reset(goal=args.goal)
    frames = []
    terminated = truncated = False
    while not (terminated or truncated):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
    save_mp4(frames, args.output, fps=args.fps)
    env.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
