from gymnasium.envs.registration import register
from envs.tripendulum_env import TriPendulumGoalEnv
from envs.goals import GOAL_NAMES, GOAL_BINARY, GOAL_ABS_ANGLES, get_goal, sample_goal

# Register the environment with Gymnasium registry
register(
    id="TriPendulumGoalEnv-v0",
    entry_point="envs.tripendulum_env:TriPendulumGoalEnv",
    max_episode_steps=1000,
)

__all__ = [
    "TriPendulumGoalEnv",
    "GOAL_NAMES",
    "GOAL_BINARY",
    "GOAL_ABS_ANGLES",
    "get_goal",
    "sample_goal"
]
