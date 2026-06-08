import os
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import SAC, PPO

from envs.tripendulum_env import TriPendulumGoalEnv
from envs.goals import GOAL_NAMES, get_goal
from evaluation.evaluate import load_policy
from utils.logging_utils import setup_logger

logger = setup_logger("transition_matrix")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Goal Transition Matrix")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 policy zip file")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per transition pair")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save output files")
    return parser.parse_args()

def evaluate_transition(env, model, start_goal, target_goal, max_pre_steps=400, max_post_steps=500):
    """
    Evaluates policy's capability to stabilize at start_goal, then transition to target_goal.
    
    Returns:
        success (bool): Whether transition was successful
        steps_taken (int): Steps taken to stabilize at target_goal after transition (-1 if failed)
    """
    # 1. Reset and set to Stage 4 (start near DDD)
    env.set_curriculum_stage(4)
    obs, info = env.reset(goal=start_goal)
    
    # 2. Run until stabilized at start_goal (or max_pre_steps reached)
    stabilized_at_start = False
    for _ in range(max_pre_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("success", False) or terminated:
            if info.get("success", False):
                stabilized_at_start = True
            break
            
    if not stabilized_at_start:
        # Failed to stabilize at the start goal, skip transition evaluation but mark as failed
        return False, -1
        
    # 3. Dynamic target switch to target_goal
    env.goal = get_goal(target_goal)
    env.stable_steps = 0 # Reset stabilization steps for new goal
    
    # 4. Run until stabilized at target_goal (or max_post_steps reached)
    success = False
    steps_taken = -1
    for step in range(1, max_post_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify success condition specifically for the target_goal
        if info.get("success", False):
            success = True
            steps_taken = step
            break
        if terminated:
            break
            
    return success, steps_taken

def main():
    args = parse_args()
    
    # Load configuration
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(curr_dir, ".."))
    config_path = args.config if args.config else os.path.join(project_root, "configs", "default.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Create env
    env = TriPendulumGoalEnv(config_path=config_path, render_mode="rgb_array")
    
    # Load policy
    try:
        model = load_policy(args.model, env)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        env.close()
        return
        
    save_dir = os.path.join(project_root, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Generating transition matrix (8x8 = 64 pairs, {args.trials} trials each)...")
    
    # Initialize matrix grids
    success_matrix = np.zeros((8, 8))
    time_matrix = np.zeros((8, 8))
    
    raw_results = []
    
    for i, start_g in enumerate(GOAL_NAMES):
        for j, target_g in enumerate(GOAL_NAMES):
            logger.info(f"Evaluating transition: {start_g} -> {target_g}")
            
            trial_successes = []
            trial_steps = []
            
            for trial in range(args.trials):
                success, steps = evaluate_transition(env, model, start_g, target_g)
                trial_successes.append(success)
                if success:
                    trial_steps.append(steps)
                    
            success_rate = np.mean(trial_successes)
            avg_steps = np.mean(trial_steps) if trial_steps else -1.0
            
            success_matrix[i, j] = success_rate
            time_matrix[i, j] = avg_steps
            
            raw_results.append({
                "from_goal": start_g,
                "to_goal": target_g,
                "success_rate": float(success_rate),
                "average_steps": float(avg_steps),
                "trials": [
                    {"success": bool(s), "steps": int(st)} for s, st in zip(trial_successes, [st if s else -1 for s, st in zip(trial_successes, trial_steps or [-1]*args.trials)])
                ]
            })
            
    # 5. Save outputs
    # Save CSV
    df_success = pd.DataFrame(success_matrix, index=GOAL_NAMES, columns=GOAL_NAMES)
    csv_path = os.path.join(save_dir, "transition_matrix.csv")
    df_success.to_csv(csv_path)
    logger.info(f"Saved transition CSV to {csv_path}")
    
    # Save Heatmap image
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_success, annot=True, cmap="YlGnBu", fmt=".1%", cbar=True, vmin=0.0, vmax=1.0)
    plt.title("Goal-to-Goal Transition Success Rates (8x8)")
    plt.xlabel("Target Goal")
    plt.ylabel("Initial Goal")
    plt.tight_layout()
    
    heatmap_path = os.path.join(save_dir, "transition_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    logger.info(f"Saved transition heatmap to {heatmap_path}")
    
    # Save JSON Report
    report = {
        "algorithm": model.__class__.__name__,
        "model_file": args.model,
        "trials_per_pair": args.trials,
        "results": raw_results
    }
    json_path = os.path.join(save_dir, "transition_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved transition JSON report to {json_path}")
    
    env.close()

if __name__ == "__main__":
    main()
