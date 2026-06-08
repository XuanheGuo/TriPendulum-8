import os
import json
from utils.logging_utils import setup_logger

logger = setup_logger("curriculum_monitor")

class CurriculumMonitor:
    """
    Monitors and saves the history of curriculum stage transitions.
    """
    def __init__(self, save_dir="results/analytics"):
        self.save_dir = save_dir
        self.history = []
        self.load_existing()
        
    def load_existing(self):
        path = os.path.join(self.save_dir, "curriculum_history.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load existing curriculum history: {e}")

    def record_transition(self, timestep, from_stage, to_stage):
        """
        Logs a curriculum stage transition.
        """
        transition = {
            "timestep": int(timestep),
            "from_stage": int(from_stage),
            "to_stage": int(to_stage)
        }
        self.history.append(transition)
        logger.info(f"Recorded curriculum transition: Stage {from_stage} -> Stage {to_stage} at step {timestep}")
        self.save()
        
    def save(self):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, "curriculum_history.json")
            with open(path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save curriculum history: {e}")
            
    def get_history(self):
        return self.history
