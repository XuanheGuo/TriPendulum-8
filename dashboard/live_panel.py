from __future__ import annotations

from collections import defaultdict, deque


class LivePanel:
    """Small Colab-friendly metric accumulator.

    Use update(info, reward, episode_length) during training/evaluation and
    call summary() to get reward, success, energy, collision, and per-goal rates.
    """

    def __init__(self, window=100):
        self.window = window
        self.rewards = deque(maxlen=window)
        self.success = deque(maxlen=window)
        self.lengths = deque(maxlen=window)
        self.energy = deque(maxlen=window)
        self.track_collisions = deque(maxlen=window)
        self.omega_over_limit = deque(maxlen=window)
        self.goal_success = defaultdict(lambda: [0, 0])

    def update(self, info, reward, episode_length):
        goal = info.get("goal_name", "unknown")
        ok = bool(info.get("success", False))
        self.rewards.append(float(reward))
        self.success.append(float(ok))
        self.lengths.append(int(episode_length))
        self.energy.append(float(info.get("r_act", 0.0)))
        self.track_collisions.append(float(info.get("track_collision", False)))
        self.omega_over_limit.append(float(info.get("omega_over_limit", False)))
        self.goal_success[goal][0] += int(ok)
        self.goal_success[goal][1] += 1

    def summary(self):
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        return {
            "reward": avg(self.rewards),
            "success_rate": avg(self.success),
            "episode_length": avg(self.lengths),
            "track_collisions": sum(self.track_collisions),
            "average_energy": avg(self.energy),
            "omega_over_limit_count": sum(self.omega_over_limit),
            "goal_success_rate": {
                goal: s / n if n else 0.0 for goal, (s, n) in self.goal_success.items()
            },
        }
