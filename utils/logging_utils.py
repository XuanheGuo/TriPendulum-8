"""Config and filesystem helpers."""

from __future__ import annotations

import os
from copy import deepcopy

import yaml


def deep_update(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    default_path = cfg.pop("defaults", None)
    if default_path:
        if not os.path.isabs(default_path):
            default_path = os.path.join(project_root(), default_path)
        with open(default_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        cfg = deep_update(base, cfg)
    return cfg


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
