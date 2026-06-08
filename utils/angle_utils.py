"""Angle helpers for relative MuJoCo joints and absolute goal poses."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle):
    """Normalize an angle or array of angles to [-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def relative_to_absolute(q):
    """Convert [q1, q2, q3] relative joint angles to absolute link angles."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 3:
        raise ValueError(f"Expected final dimension of size 3, got shape {q.shape}")
    return wrap_angle(np.cumsum(q, axis=-1))


def angle_error(theta, theta_goal):
    """Return wrapped signed angle error theta - theta_goal."""
    return wrap_angle(np.asarray(theta) - np.asarray(theta_goal))
