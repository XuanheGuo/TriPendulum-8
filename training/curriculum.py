"""Curriculum goal sets for TriPendulum-8."""

from __future__ import annotations

ALL_GOALS = ("DDD", "DDU", "DUD", "UDD", "DUU", "UDU", "UUD", "UUU")

CURRICULUM_STAGES = {
    1: ("DDD", "UUU"),
    2: ("DDU", "DUD", "UDD"),
    3: ("DUU", "UDU", "UUD"),
    4: ALL_GOALS,
    5: ALL_GOALS,
    6: ALL_GOALS,
}


def goals_for_stage(stage: int):
    if stage in CURRICULUM_STAGES:
        return CURRICULUM_STAGES[stage]
    if stage >= 4:
        return CURRICULUM_STAGES[4]
    raise ValueError(f"Unknown curriculum stage {stage}")


def get_current_stage(curriculum_config: dict | None = None) -> int:
    curriculum_config = curriculum_config or {}
    if not curriculum_config.get("enabled", False):
        return 4
    return int(curriculum_config.get("stage", 4))


def get_current_goals(curriculum_config: dict | None = None):
    curriculum_config = curriculum_config or {}
    if not curriculum_config.get("enabled", False):
        return list(ALL_GOALS)
    return list(goals_for_stage(get_current_stage(curriculum_config)))


def apply_curriculum(env_config: dict, curriculum_config: dict | None):
    cfg = dict(env_config)
    curriculum_config = curriculum_config or {}
    if not curriculum_config.get("enabled", False):
        return cfg
    stage = int(curriculum_config.get("stage", 4))
    cfg["allowed_goals"] = goals_for_stage(stage)
    if stage >= 5:
        cfg["random_initial_state"] = True
    return cfg
