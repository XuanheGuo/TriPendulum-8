import numpy as np

GOAL_NAMES = ["DDD", "DDU", "DUD", "UDD", "DUU", "UDU", "UUD", "UUU"]

# Binary representations
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

# Absolute target angles (in radians)
GOAL_ABS_ANGLES = {
    "DDD": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "DDU": np.array([0.0, 0.0, np.pi], dtype=np.float32),
    "DUD": np.array([0.0, np.pi, 0.0], dtype=np.float32),
    "UDD": np.array([np.pi, 0.0, 0.0], dtype=np.float32),
    "DUU": np.array([0.0, np.pi, np.pi], dtype=np.float32),
    "UDU": np.array([np.pi, 0.0, np.pi], dtype=np.float32),
    "UUD": np.array([np.pi, np.pi, 0.0], dtype=np.float32),
    "UUU": np.array([np.pi, np.pi, np.pi], dtype=np.float32),
}

def get_goal(name):
    """
    Returns a dictionary of goal specs.
    """
    if name not in GOAL_NAMES:
        raise ValueError(f"Unknown goal name: {name}")
    return {
        "name": name,
        "binary": GOAL_BINARY[name].copy(),
        "abs_angles": GOAL_ABS_ANGLES[name].copy(),
    }

def sample_goal(allowed_goals=None):
    """
    Randomly samples a goal spec from a subset of allowed goals.
    """
    goals = allowed_goals if allowed_goals is not None else GOAL_NAMES
    name = np.random.choice(goals)
    return get_goal(name)

def goal_name_to_binary(name):
    if name not in GOAL_BINARY:
        raise ValueError(f"Unknown goal name: {name}")
    return GOAL_BINARY[name].copy()

def goal_name_to_abs_angles(name):
    if name not in GOAL_ABS_ANGLES:
        raise ValueError(f"Unknown goal name: {name}")
    return GOAL_ABS_ANGLES[name].copy()
