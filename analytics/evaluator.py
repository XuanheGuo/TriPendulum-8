import numpy as np
from envs.tripendulum_env import TriPendulumGoalEnv

class Evaluator:
    """
    Handles model evaluation for specific goal poses.
    Uses an isolated evaluation environment to prevent interference with training.
    """
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.env = TriPendulumGoalEnv(config_path=config_path, render_mode="rgb_array")
        
    def evaluate_goal(self, model, goal_name, n_episodes=5, curriculum_stage=1):
        """
        Runs evaluation episodes for a given goal under a specific curriculum stage.
        
        Returns:
            list of dict: Episode logs containing steps and final outcomes.
        """
        self.env.set_curriculum_stage(curriculum_stage)
        episode_logs = []
        
        for _ in range(n_episodes):
            obs, info = self.env.reset(goal=goal_name)
            
            reward_sum = 0.0
            length = 0
            pose_errors = []
            energies = []
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # Predict deterministic action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                reward_sum += reward
                length += 1
                pose_errors.append(info.get("pose_error", 0.0))
                energies.append(info.get("r_act", 0.0))
                
            episode_logs.append({
                "reward_sum": reward_sum,
                "length": length,
                "success": bool(info.get("success", False)),
                "collision": bool(info.get("track_collision", False)),
                "overspin": bool(info.get("overspin", False)),
                "pose_errors": pose_errors,
                "energies": energies,
                "max_abs_x": float(info.get("max_abs_x", 0.0)),
                "stable_steps": int(info.get("stable_steps", 0)),
                "final_pose_error": float(info.get("pose_error", 0.0))
            })
            
        return episode_logs
        
    def close(self):
        self.env.close()
