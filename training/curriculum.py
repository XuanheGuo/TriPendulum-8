from utils.logging_utils import setup_logger

logger = setup_logger("curriculum")

class CurriculumManager:
    """
    Manages the 6 stages of curriculum learning for TriPendulum-8.
    """
    STAGES = {
        1: {
            "name": "Stage 1: Simple Goal Pair",
            "goals": ["DDD", "UUU"],
            "description": "Learn basic downward stability (passive) and upward swing-up/balance (hardest)."
        },
        2: {
            "name": "Stage 2: Single-Up Configurations",
            "goals": ["DDU", "DUD", "UDD"],
            "description": "Learn configurations where only one link is upright."
        },
        3: {
            "name": "Stage 3: Double-Up Configurations",
            "goals": ["DUU", "UDU", "UUD"],
            "description": "Learn configurations where two links are upright."
        },
        4: {
            "name": "Stage 4: Combined Multi-Goal Stabilization",
            "goals": ["DDD", "DDU", "DUD", "UDD", "DUU", "UDU", "UUD", "UUU"],
            "description": "stabilize all 8 targets from a hanging start."
        },
        5: {
            "name": "Stage 5: Global Recovery",
            "goals": ["DDD", "DDU", "DUD", "UDD", "DUU", "UDU", "UUD", "UUU"],
            "description": "Start from fully random joint angles and velocities, stabilizing to any of the 8 goals."
        },
        6: {
            "name": "Stage 6: Dynamic Tracking (Any-to-Any)",
            "goals": ["DDD", "DDU", "DUD", "UDD", "DUU", "UDU", "UUD", "UUU"],
            "description": "Dynamic online target-switching inside each episode."
        }
    }

    def __init__(self, config):
        self.config = config.get("curriculum", {})
        self.enabled = self.config.get("enabled", True)
        self.auto_advance = self.config.get("auto_advance", True)
        self.advance_success_rate = self.config.get("advance_success_rate", 0.85)
        self.current_stage = 1
        
    def get_current_stage(self):
        return self.current_stage
        
    def get_stage_info(self, stage=None):
        stage = stage if stage is not None else self.current_stage
        return self.STAGES.get(stage, {})
        
    def get_current_goals(self):
        return self.STAGES[self.current_stage]["goals"]
        
    def should_advance(self, success_rate):
        """
        Determines whether the agent has mastered the current stage.
        """
        if not self.enabled:
            return False
        if self.current_stage >= 6:
            return False
        return success_rate >= self.advance_success_rate
        
    def advance(self):
        if self.current_stage < 6:
            self.current_stage += 1
            logger.info(f"ADVANCED to {self.STAGES[self.current_stage]['name']}!")
            logger.info(f"Active Goals: {self.get_current_goals()}")
            return True
        return False

    def update(self, success_rate):
        """
        Takes the current success rate and advances the curriculum if conditions are met.
        Returns the updated stage.
        """
        if self.should_advance(success_rate):
            self.advance()
        return self.current_stage
