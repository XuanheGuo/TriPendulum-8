"""Deterministic MJX policy evaluation, reported per absolute goal."""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from brax.io import model

from mjx_backend.env import GOAL_BINARY, TriPendulumMJXEnv
from mjx_backend.training import make_train_fn


GOAL_NAMES = ("DDD", "DDU", "DUD", "UDD", "DUU", "UDU", "UUD", "UUU")


def load_policy(config: dict, checkpoint: str, env):
    train_fn = make_train_fn(config, num_timesteps=0)
    make_policy, _, _ = train_fn(environment=env)
    params = model.load_params(checkpoint)
    return jax.jit(make_policy(params, deterministic=True))


def reset_goal(env, rng, goal_index: int):
    state = env.reset(rng)
    goal_binary = GOAL_BINARY[goal_index]
    goal_abs = goal_binary * jnp.pi
    rng_q, rng_qd = jax.random.split(rng)
    q = env.sys.init_q.at[1:4].set(jax.random.uniform(rng_q, (3,), minval=-0.05, maxval=0.05))
    qd = jnp.zeros_like(state.pipeline_state.qd)
    qd = qd.at[1:4].set(jax.random.uniform(rng_qd, (3,), minval=-0.05, maxval=0.05))
    pipeline_state = env.pipeline_init(q, qd)
    theta_abs = jnp.cumsum(q[1:4])
    pose_error = jnp.sum(1.0 - jnp.cos(theta_abs - goal_abs))
    info = {
        **state.info,
        "goal_index": jnp.asarray(goal_index),
        "goal_binary": goal_binary,
        "goal_abs": goal_abs,
        "initial_pose_fraction": jnp.asarray(0.0),
        "prev_pose_error": pose_error,
    }
    obs = env._get_obs(pipeline_state, goal_binary)
    return state.replace(pipeline_state=pipeline_state, obs=obs, info=info)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mjx_ppo.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="mjx_evaluation.json")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    env = TriPendulumMJXEnv(config["env"], backend="mjx")
    policy = load_policy(config, args.checkpoint, env)
    step_fn = jax.jit(env.step)
    results = {}
    for goal_index, goal_name in enumerate(GOAL_NAMES):
        reset_fn = jax.jit(lambda rng: reset_goal(env, rng, goal_index))
        successes, rewards, collisions, pose_errors = [], [], [], []
        for episode in range(args.episodes):
            state = reset_fn(jax.random.PRNGKey(10000 + episode))
            total_reward = 0.0
            rng = jax.random.PRNGKey(20000 + episode)
            for _ in range(int(config["ppo"].get("episode_length", 1000))):
                rng, action_rng = jax.random.split(rng)
                action, _ = policy(state.obs, action_rng)
                state = step_fn(state, action)
                total_reward += float(state.reward)
                if bool(state.done):
                    break
            successes.append(float(state.metrics["success"]))
            collisions.append(float(state.metrics["track_collision"]))
            pose_errors.append(float(state.metrics["pose_error"]))
            rewards.append(total_reward)
        results[goal_name] = {
            "success_rate": float(np.mean(successes)),
            "avg_reward": float(np.mean(rewards)),
            "avg_final_pose_error": float(np.mean(pose_errors)),
            "track_collision_rate": float(np.mean(collisions)),
        }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
