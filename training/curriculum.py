"""Curriculum definitions and mutable sequential training state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

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


@dataclass
class SequentialCurriculumState:
    goals: tuple[str, ...] = ALL_GOALS
    current_index: int = 0
    retain_previous_goals: bool = True
    consecutive_passes: int = 0
    goal_start_timestep: int = 0
    completed: bool = False
    history: list[dict] = field(default_factory=list)
    state_path: str | None = None

    @property
    def stage_id(self) -> int:
        return self.current_index + 1

    @property
    def current_goal(self) -> str:
        return self.goals[self.current_index]

    @property
    def training_goals(self) -> list[str]:
        if self.retain_previous_goals:
            return list(self.goals[: self.current_index + 1])
        return [self.current_goal]

    def register_evaluation(self, timestep: int, metrics: dict, passed: bool) -> None:
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        self.history.append(
            {
                "timestep": int(timestep),
                "stage_id": self.stage_id,
                "goal": self.current_goal,
                "passed": bool(passed),
                "metrics": metrics,
            }
        )
        self.save()

    def advance(self, timestep: int) -> bool:
        if self.current_index >= len(self.goals) - 1:
            self.completed = True
            self.save()
            return False
        self.current_index += 1
        self.consecutive_passes = 0
        self.goal_start_timestep = int(timestep)
        self.save()
        return True

    def to_dict(self) -> dict:
        return {
            "goals": list(self.goals),
            "current_index": self.current_index,
            "stage_id": self.stage_id,
            "current_goal": self.current_goal,
            "training_goals": self.training_goals,
            "retain_previous_goals": self.retain_previous_goals,
            "consecutive_passes": self.consecutive_passes,
            "goal_start_timestep": self.goal_start_timestep,
            "completed": self.completed,
            "history": self.history,
        }

    def save(self) -> None:
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_curriculum_state(curriculum_config: dict | None, checkpoint_dir: str | None = None):
    cfg = curriculum_config or {}
    if not cfg.get("enabled", False) or cfg.get("mode", "sequential") != "sequential":
        return None

    goals = tuple(cfg.get("goal_order", ALL_GOALS))
    unknown = [goal for goal in goals if goal not in ALL_GOALS]
    if unknown:
        raise ValueError(f"Unknown curriculum goals: {unknown}")

    state_path = cfg.get("state_path")
    if not state_path and checkpoint_dir:
        state_path = os.path.join(checkpoint_dir, "curriculum_state.json")

    state = SequentialCurriculumState(
        goals=goals,
        current_index=max(0, min(int(cfg.get("start_index", 0)), len(goals) - 1)),
        retain_previous_goals=bool(cfg.get("retain_previous_goals", True)),
        state_path=state_path,
    )
    if state_path and os.path.exists(state_path) and cfg.get("resume_state", True):
        with open(state_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if tuple(saved.get("goals", goals)) == goals:
            state.current_index = max(0, min(int(saved.get("current_index", 0)), len(goals) - 1))
            state.consecutive_passes = int(saved.get("consecutive_passes", 0))
            state.goal_start_timestep = int(saved.get("goal_start_timestep", 0))
            state.completed = bool(saved.get("completed", False))
            state.history = list(saved.get("history", []))
    state.save()
    return state


def get_current_stage(curriculum_config: dict | None = None, state=None) -> int:
    if state is not None:
        return state.stage_id
    curriculum_config = curriculum_config or {}
    if not curriculum_config.get("enabled", False):
        return 4
    return int(curriculum_config.get("stage", 4))


def get_current_goals(curriculum_config: dict | None = None, state=None):
    if state is not None:
        return state.training_goals
    curriculum_config = curriculum_config or {}
    if not curriculum_config.get("enabled", False):
        return list(ALL_GOALS)
    return list(goals_for_stage(get_current_stage(curriculum_config)))


def apply_curriculum(env_config: dict, curriculum_config: dict | None, state=None):
    cfg = dict(env_config)
    curriculum_config = curriculum_config or {}
    if not curriculum_config.get("enabled", False):
        return cfg
    if state is not None:
        cfg["allowed_goals"] = tuple(state.training_goals)
        return cfg
    stage = int(curriculum_config.get("stage", 4))
    cfg["allowed_goals"] = goals_for_stage(stage)
    if stage >= 5:
        cfg["random_initial_state"] = True
    return cfg
