import os
import json
import numpy as np
from envs.tripendulum_env import TriPendulumGoalEnv
from utils.video_utils import save_video
from utils.logging_utils import setup_logger

logger = setup_logger("video_recorder")

class VideoRecorder:
    """
    Renders policy rollouts and saves them as diagnostic videos and JSON files.
    """
    def __init__(self, config_path=None, video_dir="videos/diagnostics", algorithm="SAC"):
        self.config_path = config_path
        self.video_dir = video_dir
        self.algorithm = algorithm
        self.env = TriPendulumGoalEnv(config_path=config_path, render_mode="rgb_array")
        
    def record_goal_episode(
        self,
        model,
        goal_name,
        timestep,
        stage_id,
        stage_goals,
        selection_reason
    ):
        """
        Records a single episode for a specific goal. Saves .mp4 and .json.
        """
        filename_base = f"{self.algorithm.lower()}_step_{timestep}_stage_{stage_id}_goal_{goal_name}"
        video_path = os.path.join(self.video_dir, f"{filename_base}.mp4")
        json_path = os.path.join(self.video_dir, f"{filename_base}.json")
        
        self.env.set_curriculum_stage(stage_id)
        obs, info = self.env.reset(goal=goal_name)
        
        frames = []
        # Render initial frame
        frame = self.env.render()
        if frame is not None:
            frames.append(frame)
            
        reward_sum = 0.0
        length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            reward_sum += reward
            length += 1
            
            frame = self.env.render()
            if frame is not None:
                frames.append(frame)
                
        # Save video file
        success_save = save_video(frames, video_path, fps=30)
        
        # Save JSON metadata
        meta = {
            "algorithm": self.algorithm,
            "timestep": int(timestep),
            "stage_id": int(stage_id),
            "stage_goals": stage_goals,
            "goal": goal_name,
            "selection_reason": selection_reason,
            "success": bool(info.get("success", False)),
            "episode_reward": float(reward_sum),
            "episode_length": int(length),
            "final_pose_error": float(info.get("pose_error", 0.0)),
            "max_abs_x": float(info.get("max_abs_x", 0.0)),
            "track_collision": bool(info.get("track_collision", False)),
            "overspin": bool(info.get("overspin", False))
        }
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Saved episode diagnostic JSON metadata to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON metadata. Error: {e}")
            
        return meta
        
    def close(self):
        self.env.close()
