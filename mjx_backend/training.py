"""Shared Brax PPO construction and checkpoint utilities."""

from __future__ import annotations

import functools

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train


def make_train_fn(config: dict, num_timesteps: int | None = None, restore_params=None):
    ppo = config["ppo"]
    hidden = tuple(int(value) for value in ppo.get("hidden_layer_sizes", [256, 256, 256]))
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=hidden,
        value_hidden_layer_sizes=hidden,
    )
    kwargs = dict(
        num_timesteps=int(ppo["num_timesteps"] if num_timesteps is None else num_timesteps),
        num_evals=int(ppo.get("num_evals", 20)),
        reward_scaling=float(ppo.get("reward_scaling", 0.1)),
        episode_length=int(ppo.get("episode_length", 1000)),
        normalize_observations=bool(ppo.get("normalize_observations", True)),
        action_repeat=1,
        unroll_length=int(ppo.get("unroll_length", 20)),
        num_minibatches=int(ppo.get("num_minibatches", 32)),
        num_updates_per_batch=int(ppo.get("num_updates_per_batch", 4)),
        discounting=float(ppo.get("discounting", 0.99)),
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
        entropy_cost=float(ppo.get("entropy_cost", 0.01)),
        num_envs=int(ppo.get("num_envs", 4096)),
        batch_size=int(ppo.get("batch_size", 2048)),
        seed=int(ppo.get("seed", 42)),
        network_factory=network_factory,
    )
    if restore_params is not None:
        kwargs["restore_params"] = restore_params
    return functools.partial(ppo_train.train, **kwargs)
