import os
import argparse
import yaml
import json
import numpy as np
from collections import deque
from stable_baselines3 import SAC, PPO

from envs.tripendulum_env import TriPendulumGoalEnv
from envs.goals import GOAL_NAMES
from evaluation.evaluate import load_policy
from utils.logging_utils import setup_logger

logger = setup_logger("robustness_test")

def parse_args():
    parser = argparse.ArgumentParser(description="Robustness and Perturbation Testing")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 policy zip file")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per test type")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save JSON report")
    return parser.parse_args()

def run_test_loop(env, model, num_episodes, perturb_fn=None, obs_noise_std=0.0, action_delay_steps=0):
    """
    Generic test runner.
    perturb_fn is called at step N during the rollout (e.g. to apply an impulse).
    obs_noise_std is the standard deviation of Gaussian noise added to observations.
    action_delay_steps is the number of steps to delay the actions.
    """
    success_count = 0
    recovery_times = []
    failures = {"collision": 0, "overspin": 0, "timeout": 0}
    
    for ep in range(num_episodes):
        # Select UUU (upright) as a standard hard test goal, or sample randomly
        goal_name = "UUU"
        env.set_curriculum_stage(4) # hanging start
        obs, info = env.reset(goal=goal_name)
        
        terminated = False
        truncated = False
        step = 0
        
        # Action queue for delay modeling
        action_queue = deque(maxlen=action_delay_steps + 1) if action_delay_steps > 0 else None
        
        # Track recovery details
        perturbation_applied = False
        post_perturbation_steps = 0
        stabilized_after_perturb = False
        
        while not (terminated or truncated):
            step += 1
            
            # Apply observation noise
            if obs_noise_std > 0.0:
                obs += np.random.normal(0.0, obs_noise_std, size=obs.shape)
                
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply action delay
            if action_queue is not None:
                action_queue.append(action)
                if len(action_queue) < action_delay_steps + 1:
                    action = np.zeros_like(action) # zero control during initial lag
                else:
                    action = action_queue[0]
                    
            # Apply physical disturbance (e.g., velocity kick on the cart)
            if perturb_fn is not None and step == 150: # wait until it swings up / stabilizes
                perturb_fn(env)
                perturbation_applied = True
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            if perturbation_applied:
                post_perturbation_steps += 1
                if info.get("success", False) and not stabilized_after_perturb:
                    stabilized_after_perturb = True
                    recovery_times.append(post_perturbation_steps)
                    
        # Check outcomes
        success = info.get("success", False)
        # If perturbation was applied, success is defined as recovering after the kick
        if perturb_fn is not None:
            if perturbation_applied and stabilized_after_perturb and not info.get("track_collision", False) and not info.get("overspin", False):
                success = True
            else:
                success = False
                
        if success:
            success_count += 1
        else:
            if info.get("track_collision", False):
                failures["collision"] += 1
            elif info.get("overspin", False):
                failures["overspin"] += 1
            else:
                failures["timeout"] += 1
                
    recovery_rate = success_count / num_episodes
    avg_recovery_steps = np.mean(recovery_times) if recovery_times else -1.0
    
    return {
        "success_rate": recovery_rate,
        "avg_recovery_steps": avg_recovery_steps,
        "failures": failures
    }

def main():
    args = parse_args()
    
    # Load configuration
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
        
    save_dir = os.path.join(project_root, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Running robustness benchmarks ({args.episodes} episodes per test)...")
    
    results = {}
    
    # 1. Baseline Performance (No Perturbations)
    logger.info("Test 1: Baseline stability")
    results["baseline"] = run_test_loop(env, model, args.episodes)
    
    # 2. Cart External Impulse Disturbance (Velocity kick)
    logger.info("Test 2: Cart external force impulse")
    def apply_impulse(env_inst):
        # Add 3.0 m/s velocity kick to the cart slider
        env_inst.data.qvel[0] += 3.0
        logger.info("--> Applied +3.0 m/s impulse to cart")
    results["impulse"] = run_test_loop(env, model, args.episodes, perturb_fn=apply_impulse)
    
    # 3. Observation Noise (Gaussian noise on sensor readings)
    logger.info("Test 3: Observation noise (std=0.05)")
    results["obs_noise"] = run_test_loop(env, model, args.episodes, obs_noise_std=0.05)
    
    # 4. Action Delay (1 step control delay = ~10ms)
    logger.info("Test 4: Action control delay (1 step)")
    results["action_delay"] = run_test_loop(env, model, args.episodes, action_delay_steps=1)
    
    # 5. Mass Mismatch (+20% cart and links mass)
    logger.info("Test 5: Model mass variation (+20%)")
    orig_mass = env.model.body_mass.copy()
    env.model.body_mass[:] = orig_mass * 1.20
    results["mass_mismatch"] = run_test_loop(env, model, args.episodes)
    env.model.body_mass[:] = orig_mass # Restore
    
    # 6. Joint Friction Mismatch (+50% hinge joint damping)
    logger.info("Test 6: Joint friction damping (+50%)")
    orig_damping = env.model.dof_damping.copy()
    env.model.dof_damping[:] = orig_damping * 1.50
    results["friction_mismatch"] = run_test_loop(env, model, args.episodes)
    env.model.dof_damping[:] = orig_damping # Restore
    
    # 7. Print summary table
    print("\n================ ROBUSTNESS TEST REPORT ================")
    print(f"Policy: {args.model}")
    print(f"Episodes per benchmark: {args.episodes}")
    print("--------------------------------------------------------")
    for test_name, res in results.items():
        rec_time_str = f"{res['avg_recovery_steps']*dt:.3f}s" if res['avg_recovery_steps'] > 0 else "N/A"
        print(f"Test: {test_name:<18} | Success Rate: {res['success_rate']:.1%} | Recovery Time: {rec_time_str} | Failures: {res['failures']}")
    print("========================================================\n")
    
    # Save JSON report
    report = {
        "model": args.model,
        "episodes_per_test": args.episodes,
        "results": results
    }
    report_path = os.path.join(save_dir, "robustness_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved robustness JSON report to {report_path}")
    
    env.close()

if __name__ == "__main__":
    main()
