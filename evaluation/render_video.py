import os
import argparse
import yaml
import json
import numpy as np
import cv2
from stable_baselines3 import SAC, PPO

from envs.tripendulum_env import TriPendulumGoalEnv
from envs.goals import get_goal
from evaluation.evaluate import load_policy
from utils.video_utils import save_video
from utils.logging_utils import setup_logger

logger = setup_logger("render_video")

def parse_args():
    parser = argparse.ArgumentParser(description="Render policy rollout video with text overlay")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 policy zip file")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--goal", type=str, default="UUU", help="Target goal pose (DDD, UUU, etc.)")
    parser.add_argument("--output", type=str, default="videos/sac_rollout.mp4", help="Path to save MP4 video")
    parser.add_argument("--steps", type=int, default=500, help="Maximum steps to render")
    parser.add_argument("--stage", type=int, default=4, help="Curriculum stage context for starting state")
    return parser.parse_args()

def draw_overlay(frame, info, step, reward, goal_name, x, pose_error):
    """
    Draws a dashboard text overlay onto the RGB image frame using OpenCV.
    """
    # Create a copy to avoid mutating the original frame buffer
    img = frame.copy()
    
    # Text configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Stats panel (Top Left)
    stats = [
        f"Goal: {goal_name}",
        f"Step: {step} | Reward: {reward:.3f}",
        f"Cart X: {x:.3f} m (limit: 1.0m)",
        f"Pose Error: {pose_error:.3f} rad",
        f"Poles (Abs): " + ", ".join([f"{a:.2f}" for a in info.get("theta_abs", [0.0]*3)]) + " rad",
    ]
    
    # Semi-transparent overlay box for readability
    cv2.rectangle(img, (10, 10), (280, 130), (20, 20, 20), -1)
    
    # Draw stats text (color is RGB since MuJoCo renders RGB)
    for idx, text in enumerate(stats):
        y_pos = 30 + idx * 18
        cv2.putText(img, text, (20, y_pos), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    # Boundary warning
    if info.get("track_collision", False) or abs(x) > 1.0:
        cv2.rectangle(img, (10, 140), (280, 170), (20, 20, 255), -1) # Red box
        cv2.putText(img, "RAIL COLLISION", (20, 160), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
    # Overspin warning
    if info.get("overspin", False):
        cv2.rectangle(img, (10, 180), (280, 210), (20, 20, 255), -1) # Red box
        cv2.putText(img, "JOINT OVERSPIN WARNING", (20, 200), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
    # Success Indicator
    if info.get("success", False):
        cv2.rectangle(img, (10, 220), (280, 250), (20, 255, 20), -1) # Green box
        cv2.putText(img, f"STABLE: {info.get('stable_steps', 0)} / 50", (20, 240), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        
    return img

def main():
    args = parse_args()
    
    # Load configuration
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(curr_dir, ".."))
    config_path = args.config if args.config else os.path.join(project_root, "configs", "default.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Create environment with render mode rgb_array
    env = TriPendulumGoalEnv(config_path=config_path, render_mode="rgb_array")
    
    # Load policy
    try:
        model = load_policy(args.model, env)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        env.close()
        return
        
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    env.set_curriculum_stage(args.stage)
    obs, info = env.reset(goal=args.goal)
    
    logger.info(f"Simulating rollout for goal {args.goal}...")
    
    frames = []
    
    # Capture initial frame
    raw_frame = env.render()
    if raw_frame is not None:
        overlay_frame = draw_overlay(raw_frame, info, 0, 0.0, args.goal, info["x"], 9.99)
        frames.append(overlay_frame)
        
    reward_sum = 0.0
    step = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and step < args.steps:
        step += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        
        raw_frame = env.render()
        if raw_frame is not None:
            overlay_frame = draw_overlay(
                frame=raw_frame,
                info=info,
                step=step,
                reward=reward_sum,
                goal_name=args.goal,
                x=info["x"],
                pose_error=info.get("pose_error", 0.0)
            )
            frames.append(overlay_frame)
            
    # Save the MP4 file
    success = save_video(frames, output_path, fps=30)
    
    # Save JSON rollout data alongside
    json_path = output_path.replace(".mp4", ".json")
    rollout_data = {
        "model": args.model,
        "goal": args.goal,
        "steps": step,
        "total_reward": reward_sum,
        "success": bool(info.get("success", False)),
        "max_abs_x": float(info.get("max_abs_x", 0.0)),
        "track_collision": bool(info.get("track_collision", False)),
        "overspin": bool(info.get("overspin", False)),
    }
    with open(json_path, "w") as f:
        json.dump(rollout_data, f, indent=2)
        
    logger.info(f"Saved rollout video data summary to {json_path}")
    env.close()

if __name__ == "__main__":
    main()
