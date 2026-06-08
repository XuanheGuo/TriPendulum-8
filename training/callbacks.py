import os
import yaml
from stable_baselines3.common.callbacks import BaseCallback

from training.curriculum import CurriculumManager
from analytics.analytics_manager import AnalyticsManager
from utils.logging_utils import setup_logger

logger = setup_logger("callbacks")

class CurriculumAnalyticsCallback(BaseCallback):
    """
    SB3 custom callback that integrates Curriculum Learning updates 
    and Analytics System evaluations.
    """
    def __init__(self, config_path=None, verbose=0):
        super().__init__(verbose)
        
        # Load configuration
        if config_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(curr_dir, "..", "configs", "default.yaml")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Initialize curriculum manager
        self.curriculum_manager = CurriculumManager(self.config)
        self.curriculum_enabled = self.config.get("curriculum", {}).get("enabled", True)
        
        # Initialize analytics manager
        self.analytics_enabled = self.config.get("analytics", {}).get("enabled", True)
        self.eval_freq = self.config.get("analytics", {}).get("eval_freq", 50000)
        self.last_eval_step = 0
        
        self.analytics_manager = None # Will initialize on training start
        self.config_path = config_path

    def _on_training_start(self) -> None:
        # Initialize analytics manager with current log dir and policy
        if self.analytics_enabled:
            # Check if model has a logger/tb logger
            tensorboard_log_dir = self.model.logger.dir if self.model.logger else None
            self.analytics_manager = AnalyticsManager(
                model=self.model,
                config_path=self.config_path,
                tensorboard_log_dir=tensorboard_log_dir
            )
            
        # Sync initial curriculum stage to envs
        self._sync_curriculum_stage(self.curriculum_manager.get_current_stage())

    def _on_step(self) -> bool:
        # Check if it is time to run evaluation and report analytics
        step = self.num_timesteps
        if self.analytics_enabled and (step >= self.last_eval_step + self.eval_freq):
            self.last_eval_step = (step // self.eval_freq) * self.eval_freq
            logger.info(f"Step {step}: Triggering Analytics and Evaluation...")
            
            # 1. Run evaluation and diagnostics
            current_stage = self.curriculum_manager.get_current_stage()
            allowed_goals = self.curriculum_manager.get_current_goals()
            
            eval_results = self.analytics_manager.run_evaluation(
                timestep=step,
                stage_id=current_stage,
                stage_goals=allowed_goals
            )
            
            stage_success_rate = eval_results.get("stage_success_rate", 0.0)
            logger.info(f"Stage {current_stage} Goals Success Rate: {stage_success_rate:.2%}")
            
            # Save best model checkpoint
            if not hasattr(self, "best_success_rate") or stage_success_rate >= self.best_success_rate:
                self.best_success_rate = stage_success_rate
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                checkpoint_dir = os.path.join(curr_dir, "..", "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                algo_name = self.model.__class__.__name__.lower()
                save_path = os.path.join(checkpoint_dir, f"{algo_name}_best.zip")
                self.model.save(save_path)
                logger.info(f"New best success rate: {self.best_success_rate:.2%}. Saved model to {save_path}")
            
            
            # 3. Update Curriculum
            if self.curriculum_enabled:
                old_stage = self.curriculum_manager.get_current_stage()
                new_stage = self.curriculum_manager.update(stage_success_rate)
                
                if new_stage != old_stage:
                    logger.info(f"Curriculum advanced from Stage {old_stage} to Stage {new_stage}")
                    self._sync_curriculum_stage(new_stage)
                    # Reset success tracking for the new stage
                    self.analytics_manager.curriculum_monitor.record_transition(step, old_stage, new_stage)

        return True

    def _sync_curriculum_stage(self, stage):
        """
        Updates the curriculum stage variable inside the vector environments.
        """
        try:
            if hasattr(self.training_env, "env_method"):
                self.training_env.env_method("set_curriculum_stage", stage)
            elif hasattr(self.training_env, "set_curriculum_stage"):
                self.training_env.set_curriculum_stage(stage)
            else:
                # If wrapped otherwise, try accessing raw envs
                logger.warning("Could not locate set_curriculum_stage on training_env, trying direct attributes.")
        except Exception as e:
            logger.error(f"Failed to sync curriculum stage {stage} to env. Error: {e}")
