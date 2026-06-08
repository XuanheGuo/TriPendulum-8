from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mujoco
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from dashboard.plots import plot_transition_heatmap
from envs.goals import GOAL_ABS_ANGLES, GOAL_NAMES
from envs.tripendulum_env import TriPendulumGoalEnv
from utils.angle_utils import angle_error


def load_model(path, env):
    try:
        return SAC.load(path, env=env)
    except Exception:
        return PPO.load(path, env=env)


def absolute_to_relative(theta_abs):
    theta_abs = np.asarray(theta_abs, dtype=np.float64)
    return np.array(
        [
            theta_abs[0],
            angle_error(theta_abs[1], theta_abs[0]),
            angle_error(theta_abs[2], theta_abs[1]),
        ],
        dtype=np.float64,
    )


def set_start_pose(env, start_goal):
    env.data.qpos[:] = 0.0
    env.data.qvel[:] = 0.0
    env.data.qpos[1:4] = absolute_to_relative(GOAL_ABS_ANGLES[start_goal])
    mujoco.mj_forward(env.model, env.data)


def compute_transition_matrix(model, trials=3):
    env = TriPendulumGoalEnv()
    matrix = np.zeros((len(GOAL_NAMES), len(GOAL_NAMES)), dtype=float)
    for i, start in enumerate(GOAL_NAMES):
        for j, target in enumerate(GOAL_NAMES):
            successes = 0
            for _ in range(trials):
                obs, _ = env.reset(goal=target)
                set_start_pose(env, start)
                obs = env._get_obs()
                terminated = truncated = False
                ok = False
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    ok = ok or bool(info.get("success", False))
                successes += int(ok)
            matrix[i, j] = successes / trials
    env.close()
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--csv", default="transition_success_matrix.csv")
    parser.add_argument("--heatmap", default="transition_success_heatmap.png")
    args = parser.parse_args()

    env = TriPendulumGoalEnv()
    model = load_model(args.model, env)
    env.close()
    matrix = compute_transition_matrix(model, args.trials)
    pd.DataFrame(matrix, index=GOAL_NAMES, columns=GOAL_NAMES).to_csv(args.csv)
    plot_transition_heatmap(matrix, GOAL_NAMES, args.heatmap)
    print(pd.DataFrame(matrix, index=GOAL_NAMES, columns=GOAL_NAMES).to_string())


if __name__ == "__main__":
    main()
