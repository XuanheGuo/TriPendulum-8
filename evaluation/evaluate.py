from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from envs.goals import GOAL_NAMES
from envs.tripendulum_env import TriPendulumGoalEnv


def load_model(path, env):
    try:
        return SAC.load(path, env=env)
    except Exception:
        return PPO.load(path, env=env)


def evaluate_goal_metrics(
    model,
    goals=None,
    episodes_per_goal=5,
    deterministic=True,
    env_config=None,
    reward_config=None,
    max_steps=None,
    base_seed=None,
):
    goals = list(goals or GOAL_NAMES)
    env_cfg = dict(env_config or {})
    if reward_config is not None:
        env_cfg["reward"] = reward_config
    env = TriPendulumGoalEnv(env_cfg if env_cfg else None)
    goal_metrics = {}
    try:
        for goal_index, goal in enumerate(goals):
            stats = []
            for episode_index in range(episodes_per_goal):
                episode_seed = None
                if base_seed is not None:
                    episode_seed = int(base_seed) + goal_index * 1000 + episode_index
                obs, _ = env.reset(seed=episode_seed, goal=goal)
                total_reward = 0.0
                energy = 0.0
                max_abs_x = 0.0
                collisions = 0
                pose_errors = []
                final_pose_error = np.nan
                min_pose_error = np.nan
                stable_time = 0
                success = False
                terminated = truncated = False
                episode_length = 0
                while not (terminated or truncated):
                    if max_steps is not None and episode_length >= max_steps:
                        truncated = True
                        break
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_length += 1
                    total_reward += float(reward)
                    energy += float(info.get("r_act", 0.0))
                    stable_time = max(stable_time, int(info.get("stable_steps", 0)))
                    max_abs_x = max(max_abs_x, abs(float(info.get("x", 0.0))))
                    collisions += int(info.get("track_collision", False))
                    pose_errors.append(float(info.get("r_pose", np.nan)))
                    final_pose_error = float(info.get("r_pose", np.nan))
                    success = success or bool(info.get("success", False))
                if pose_errors:
                    min_pose_error = float(np.nanmin(pose_errors))
                stats.append(
                    {
                        "success": float(success),
                        "reward": total_reward,
                        "pose_error": float(np.nanmean(pose_errors)) if pose_errors else float("nan"),
                        "final_pose_error": final_pose_error,
                        "min_pose_error": min_pose_error,
                        "episode_length": episode_length,
                        "stable_time": stable_time,
                        "energy": energy,
                        "max_abs_x": max_abs_x,
                        "track_collision": float(collisions > 0),
                    }
                )
            goal_metrics[goal] = {
                "success_rate": float(np.mean([s["success"] for s in stats])),
                "avg_reward": float(np.mean([s["reward"] for s in stats])),
                "avg_pose_error": float(np.nanmean([s["pose_error"] for s in stats])),
                "avg_final_pose_error": float(np.nanmean([s["final_pose_error"] for s in stats])),
                "avg_min_pose_error": float(np.nanmean([s["min_pose_error"] for s in stats])),
                "avg_episode_length": float(np.mean([s["episode_length"] for s in stats])),
                "avg_stable_time": float(np.mean([s["stable_time"] for s in stats])),
                "track_collision_rate": float(np.mean([s["track_collision"] for s in stats])),
                "max_abs_x": float(np.max([s["max_abs_x"] for s in stats])),
                "avg_energy": float(np.mean([s["energy"] for s in stats])),
            }
    finally:
        env.close()
    return goal_metrics


def evaluate_model(model, episodes_per_goal=5, deterministic=True, goals=None):
    metrics = evaluate_goal_metrics(
        model,
        goals=goals or GOAL_NAMES,
        episodes_per_goal=episodes_per_goal,
        deterministic=deterministic,
    )
    rows = []
    for goal, goal_metrics in metrics.items():
        rows.append(
            {
                "goal": goal,
                "success_rate": goal_metrics["success_rate"],
                "average_reward": goal_metrics["avg_reward"],
                "average_pose_error": goal_metrics["avg_pose_error"],
                "average_final_pose_error": goal_metrics["avg_final_pose_error"],
                "average_min_pose_error": goal_metrics["avg_min_pose_error"],
                "average_stable_time": goal_metrics["avg_stable_time"],
                "average_energy": goal_metrics["avg_energy"],
                "average_max_cart_displacement": goal_metrics["max_abs_x"],
                "track_collision_count": goal_metrics["track_collision_rate"] * episodes_per_goal,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes-per-goal", type=int, default=5)
    parser.add_argument("--goals", nargs="*", default=None)
    parser.add_argument("--output", default="evaluation_results.csv")
    args = parser.parse_args()

    env = TriPendulumGoalEnv()
    model = load_model(args.model, env)
    env.close()
    df = evaluate_model(model, args.episodes_per_goal, goals=args.goals)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
