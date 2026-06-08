import os
import argparse
import yaml
import numpy as np
from tabulate import tabulate
from stable_baselines3 import SAC, PPO

from envs.tripendulum_env import TriPendulumGoalEnv
from envs.goals import GOAL_NAMES
from analytics.metrics import aggregate_episode_metrics
from utils.logging_utils import setup_logger

logger = setup_logger("evaluate")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TriPendulum-8 policy")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 policy zip file")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to evaluate per goal")
    parser.add_argument("--stage", type=int, default=5, help="Curriculum stage context for evaluation (5 = random start, 6 = dynamic)")
    return parser.parse_args()

def load_policy(model_path, env):
    """
    Attempts to load the model file automatically determining if it is SAC or PPO.
    """
    filename = os.path.basename(model_path).lower()
    if "sac" in filename:
        logger.info(f"Loading SAC model from {model_path}...")
        return SAC.load(model_path, env=env)
    elif "ppo" in filename:
        logger.info(f"Loading PPO model from {model_path}...")
        return PPO.load(model_path, env=env)
        
    # Fallback check inside zip structure or prompt user (here we try both)
    try:
        logger.info(f"Trying to load model as SAC...")
        return SAC.load(model_path, env=env)
    except Exception:
        try:
            logger.info(f"Trying to load model as PPO...")
            return PPO.load(model_path, env=env)
        except Exception as e:
            raise ValueError(f"Could not load policy as SAC or PPO: {e}")

def main():
    args = parse_args()
    
    # Load environment configuration
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(curr_dir, ".."))
    config_path = args.config if args.config else os.path.join(project_root, "configs", "default.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env_cfg = config.get("env", {})
    dt = env_cfg.get("frame_skip", 5) * 0.002
    
    # Create environment
    env = TriPendulumGoalEnv(config_path=config_path, render_mode="rgb_array")
    
    # Load policy
    try:
        model = load_policy(args.model, env)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        env.close()
        return
        
    logger.info(f"Evaluating policy on all 8 goals ({args.episodes} episodes each, Curriculum Stage {args.stage})...")
    
    results = []
    
    for goal_name in GOAL_NAMES:
        env.set_curriculum_stage(args.stage)
        episode_logs = []
        
        for ep in range(args.episodes):
            obs, info = env.reset(goal=goal_name)
            
            reward_sum = 0.0
            length = 0
            pose_errors = []
            energies = []
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
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
            
        # Aggregate metrics
        m = aggregate_episode_metrics(episode_logs, dt=dt)
        results.append([
            goal_name,
            f"{m['success_rate']:.1%}",
            f"{m['average_reward']:.1f}",
            f"{m['average_pose_error']:.3f}",
            f"{m['stability_time']:.2f}s",
            f"{m['average_energy']:.3f}",
            f"{m['average_max_cart_displacement']:.2f}m",
            f"{m['track_collision_rate']:.1%}",
            f"{m['overspin_rate']:.1%}"
        ])
        
    # Print results table
    headers = [
        "Goal", "Success Rate", "Avg Reward", "Avg Pose Err", 
        "Stable Duration", "Avg Energy", "Max Cart Disp", "Collision Rate", "Overspin Rate"
    ]
    print("\n" + tabulate(results, headers=headers, tablefmt="grid") + "\n")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
