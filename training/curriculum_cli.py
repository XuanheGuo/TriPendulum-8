from __future__ import annotations

import argparse
import json
import os


def load_state(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Curriculum state not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path, state):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def show(state):
    print("stage_id:", state.get("stage_id"))
    print("current_goal:", state.get("current_goal"))
    print("training_goals:", state.get("training_goals"))
    print("difficulty_stage:", state.get("difficulty_stage", 1))
    print("difficulty_label:", state.get("difficulty_label", "unknown"))
    print("initial_pose_fraction:", state.get("current_initial_pose_fraction", 0.0))
    print("consecutive_passes:", state.get("consecutive_passes"))
    print("completed:", state.get("completed"))
    history = state.get("history", [])
    if history:
        print("last_evaluation:")
        print(json.dumps(history[-1], indent=2))
    else:
        print("last_evaluation: none")


def advance(state, timestep):
    goals = state["goals"]
    current_index = int(state.get("current_index", 0))
    fractions = state.get("difficulty_fractions", [1.0, 0.5, 0.25, 0.0])
    difficulty_index = int(state.get("difficulty_index", 0))
    current_goal = goals[current_index]
    if current_goal != "DDD" and difficulty_index < len(fractions) - 1:
        difficulty_index += 1
        state["difficulty_index"] = difficulty_index
        fraction = float(fractions[difficulty_index])
        state["difficulty_stage"] = difficulty_index + 1
        state["current_initial_pose_fraction"] = fraction
        state["difficulty_label"] = (
            "halfway_90deg" if fraction >= 0.49 else "quarter_45deg" if fraction >= 0.24 else "full_down_swingup"
        )
        state["consecutive_passes"] = 0
        state["goal_start_timestep"] = int(timestep)
        return "difficulty"
    if current_index >= len(goals) - 1:
        state["completed"] = True
        return "complete"
    state["current_index"] = current_index + 1
    state["difficulty_index"] = 0
    state["stage_id"] = state["current_index"] + 1
    state["current_goal"] = goals[state["current_index"]]
    if state.get("retain_previous_goals", True):
        state["training_goals"] = goals[: state["current_index"] + 1]
    else:
        state["training_goals"] = [state["current_goal"]]
    state["consecutive_passes"] = 0
    state["goal_start_timestep"] = int(timestep)
    state["completed"] = False
    fraction = 0.0 if state["current_goal"] == "DDD" else float(fractions[0])
    state["difficulty_stage"] = 1
    state["current_initial_pose_fraction"] = fraction
    state["difficulty_label"] = "near_target" if fraction > 0 else "full_down_swingup"
    return "goal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="checkpoints/curriculum_state.json")
    parser.add_argument("command", choices=("show", "advance"))
    parser.add_argument("--timestep", type=int, default=0)
    args = parser.parse_args()

    state = load_state(args.state)
    if args.command == "show":
        show(state)
        return

    old_goal = state.get("current_goal")
    advance_type = advance(state, args.timestep)
    if advance_type == "difficulty":
        save_state(args.state, state)
        print(
            f"Advanced difficulty for {old_goal}: {state['difficulty_label']} "
            f"(fraction={state['current_initial_pose_fraction']})"
        )
    elif advance_type == "goal":
        save_state(args.state, state)
        print(f"Advanced curriculum: {old_goal} -> {state['current_goal']}")
    else:
        save_state(args.state, state)
        print("Curriculum is already at the final goal.")


if __name__ == "__main__":
    main()
