"""Absolute goal-pose definitions for TriPendulum-8."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

GOAL_NAMES = ("DDD", "DDU", "DUD", "UDD", "DUU", "UDU", "UUD", "UUU")

GOAL_BINARY = {
    "DDD": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "DDU": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "DUD": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "UDD": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "DUU": np.array([0.0, 1.0, 1.0], dtype=np.float32),
    "UDU": np.array([1.0, 0.0, 1.0], dtype=np.float32),
    "UUD": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    "UUU": np.array([1.0, 1.0, 1.0], dtype=np.float32),
}

GOAL_ABS_ANGLES = {
    name: GOAL_BINARY[name].astype(np.float64) * np.pi for name in GOAL_NAMES
}


@dataclass(frozen=True)
class Goal:
    name: str
    binary: np.ndarray
    abs_angles: np.ndarray


def get_goal(name: str) -> Goal:
    name = name.upper()
    if name not in GOAL_NAMES:
        raise ValueError(f"Unknown goal {name!r}; expected one of {GOAL_NAMES}")
    return Goal(
        name=name,
        binary=GOAL_BINARY[name].copy(),
        abs_angles=GOAL_ABS_ANGLES[name].copy(),
    )


def sample_goal(rng=None, allowed_goals=None) -> Goal:
    names = tuple(allowed_goals) if allowed_goals is not None else GOAL_NAMES
    if rng is not None and hasattr(rng, "choice"):
        name = str(rng.choice(names))
    else:
        name = random.choice(names)
    return get_goal(name)
