"""Deterministic MJX policy evaluation, reported per absolute goal."""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from brax.io import model
from mujoco import mjx
from mujoco_playground._src import wrapper

from envs.goals import GOAL_NAMES
from mjx_backend.env import GOAL_BINARY, TriPendulumMJXEnv, absolute_to_relative
from mjx_backend.training import make_train_fn


def load_policy(config: dict, checkpoint: str, env):
    train_fn = make_train_fn(config, num_timesteps=0)
    make_policy, _, _ = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    params = model.load_params(checkpoint)
    return jax.jit(make_policy(params, deterministic=True))


def reset_goal(env, rng, goal_index: int, pose_fraction: float = 1.0):
    state = env.reset(rng)
    goal_binary = GOAL_BINARY[goal_index]
    goal_abs = goal_binary * jnp.pi
    # Start near the goal (matches training distribution), not at qpos0
    target_q = absolute_to_relative(goal_abs * pose_fraction)
    rng_q, rng_qd = jax.random.split(rng)
    q = jnp.asarray(env.mj_model.qpos0).at[0].set(0.0)
    q = q.at[1:4].set(target_q + jax.random.uniform(rng_q, (3,), minval=-0.1, maxval=0.1))
    qd = jax.random.uniform(rng_qd, (state.data.qvel.shape[0],), minval=-0.1, maxval=0.1)
    qd = qd.at[0].set(jnp.clip(qd[0], -0.05, 0.05))
    data = state.data.replace(qpos=q, qvel=qd)
    data = mjx.forward(env.mjx_model, data)
    theta_abs = jnp.cumsum(q[1:4])
    pose_error = jnp.sum(1.0 - jnp.cos(theta_abs - goal_abs))
    info = {
        **state.info,
        "goal_index": jnp.asarray(goal_index),
        "goal_binary": goal_binary,
        "goal_abs": goal_abs,
        "initial_pose_fraction": jnp.asarray(pose_fraction),
        "prev_pose_error": pose_error,
    }
    obs = env._get_obs(data, goal_binary)
    return state.replace(data=data, obs=obs, info=info)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mjx_ppo.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="mjx_evaluation.json")
    parser.add_argument("--pose-fraction", type=float, default=1.0,
                        help="Initial pose fraction: 1.0=near goal, 0.0=full swing-up")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    env = TriPendulumMJXEnv(config["env"])
    policy = load_policy(config, args.checkpoint, env)
    episode_length = int(config["ppo"].get("episode_length", 1000))

    @jax.jit
    def run_episode(reset_rng: jax.Array, policy_rng: jax.Array, goal_index: int):
        state = reset_goal(env, reset_rng, goal_index, pose_fraction=args.pose_fraction)

        def step_fn(carry, _):
            state, rng, total_reward, steps = carry
            already_done = state.done > 0.5
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(state.obs, action_rng)
            new_state = env.step(state, action)
            # Freeze state after done to prevent physics divergence / NaN
            next_state = jax.tree_util.tree_map(
                lambda n, o: jnp.where(already_done, o, n), new_state, state
            )
            total_reward = total_reward + jnp.where(already_done, 0.0, new_state.reward)
            steps = steps + jnp.where(already_done, 0, 1)
            return (next_state, rng, total_reward, steps), None

        (final_state, _, total_reward, steps), _ = jax.lax.scan(
            step_fn, (state, policy_rng, jnp.asarray(0.0), jnp.asarray(0)), None, length=episode_length
        )
        return (
            final_state.metrics["success"],
            final_state.metrics["track_collision"],
            final_state.metrics["pose_error"],
            total_reward,
            steps.astype(jnp.float32),
        )

    batched_run = jax.jit(jax.vmap(run_episode, in_axes=(0, 0, None)))

    results = {}
    print(f"\nEvaluating from pose_fraction={args.pose_fraction} (1.0=near goal, 0.0=full swing-up)\n")
    for goal_index, goal_name in enumerate(GOAL_NAMES):
        reset_rngs = jax.random.split(jax.random.PRNGKey(10000), args.episodes)
        policy_rngs = jax.random.split(jax.random.PRNGKey(20000), args.episodes)
        successes, collisions, pose_errors, rewards, steps = batched_run(reset_rngs, policy_rngs, goal_index)
        results[goal_name] = {
            "success_rate": float(jnp.mean(successes)),
            "avg_reward": float(jnp.mean(rewards)),
            "avg_final_pose_error": float(jnp.mean(pose_errors)),
            "track_collision_rate": float(jnp.mean(collisions)),
            "avg_steps": float(jnp.mean(steps)),
        }
        r = results[goal_name]
        print(f"{goal_name}: success={r['success_rate']:.2f}  pose_err={r['avg_final_pose_error']:.3f}"
              f"  collision={r['track_collision_rate']:.2f}  steps={r['avg_steps']:.0f}"
              f"  reward={r['avg_reward']:.1f}")
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
