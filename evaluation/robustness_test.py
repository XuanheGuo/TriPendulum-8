from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import mujoco
from stable_baselines3 import PPO, SAC

from envs.goals import GOAL_NAMES
from envs.tripendulum_env import TriPendulumGoalEnv


def load_model(path, env):
    try:
        return SAC.load(path, env=env)
    except Exception:
        return PPO.load(path, env=env)


def run_case(model, case, magnitude, episodes):
    env = TriPendulumGoalEnv()
    base_mass = env.model.body_mass.copy()
    base_friction = env.model.dof_frictionloss.copy()
    base_body_pos = env.model.body_pos.copy()
    link2_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link2")
    link3_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link3")
    rows = []
    for goal in GOAL_NAMES:
        for _ in range(episodes):
            env.model.body_mass[:] = base_mass
            env.model.dof_frictionloss[:] = base_friction
            env.model.body_pos[:] = base_body_pos
            env.data.qfrc_applied[:] = 0.0
            obs, _ = env.reset(goal=goal)
            if case == "initial_angle":
                env.data.qpos[1:4] += np.random.uniform(-magnitude, magnitude, size=3)
            elif case == "mass":
                env.model.body_mass[:] *= 1.0 + np.random.uniform(-magnitude, magnitude)
            elif case == "friction":
                env.model.dof_frictionloss[:] *= 1.0 + np.random.uniform(0.0, magnitude)
            elif case == "link_length":
                scale = 1.0 + np.random.uniform(-magnitude, magnitude)
                env.model.body_pos[link2_id, 2] = base_body_pos[link2_id, 2] * scale
                env.model.body_pos[link3_id, 2] = base_body_pos[link3_id, 2] * scale
            terminated = truncated = False
            ok = False
            recovery_step = None
            step = 0
            failure_reason = "timeout"
            action_queue = []
            while not (terminated or truncated):
                noisy_obs = obs.copy()
                if case == "observation_noise":
                    noisy_obs += np.random.normal(0.0, magnitude, size=noisy_obs.shape)
                action, _ = model.predict(noisy_obs, deterministic=True)
                if case == "action_delay":
                    action_queue.append(action)
                    action = action_queue.pop(0) if len(action_queue) > 3 else np.zeros_like(action)
                if case == "cart_force":
                    env.data.qfrc_applied[0] = np.random.uniform(-magnitude, magnitude)
                else:
                    env.data.qfrc_applied[:] = 0.0
                obs, _, terminated, truncated, info = env.step(action)
                step += 1
                if info.get("success", False) and recovery_step is None:
                    ok = True
                    recovery_step = step
                if info.get("track_collision", False):
                    failure_reason = "track_collision"
                elif info.get("omega_over_limit", False):
                    failure_reason = "omega_over_limit"
            rows.append(
                {
                    "case": case,
                    "magnitude": magnitude,
                    "goal": goal,
                    "recovered": ok,
                    "recovery_time": recovery_step if recovery_step is not None else np.nan,
                    "failure_reason": "recovered" if ok else failure_reason,
                }
            )
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--output", default="robustness_results.csv")
    args = parser.parse_args()
    cases = [
        ("cart_force", 5.0),
        ("initial_angle", 0.15),
        ("observation_noise", 0.02),
        ("action_delay", 1.0),
        ("mass", 0.1),
        ("link_length", 0.1),
        ("friction", 0.2),
    ]
    env = TriPendulumGoalEnv()
    model = load_model(args.model, env)
    env.close()
    rows = []
    for case, magnitude in cases:
        rows.extend(run_case(model, case, magnitude, args.episodes))
    df = pd.DataFrame(rows)
    summary = df.groupby(["case", "magnitude"]).agg(
        recovery_rate=("recovered", "mean"),
        recovery_time=("recovery_time", "mean"),
        failure_reason=("failure_reason", lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown"),
    )
    df.to_csv(args.output, index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
