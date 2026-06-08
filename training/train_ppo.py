import os
import argparse
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.tripendulum_env import TriPendulumGoalEnv
from training.callbacks import CurriculumAnalyticsCallback
from utils.logging_utils import setup_logger

logger = setup_logger("train_ppo")

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on TriPendulum-8")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--ppo-config", type=str, default=None, help="Path to PPO YAML")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total timesteps to train")
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint if exists")
    parser.add_argument("--tb-log", type=str, default="runs/ppo", help="TensorBoard log dir")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Configurations
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(curr_dir, ".."))
    
    config_path = args.config if args.config else os.path.join(project_root, "configs", "default.yaml")
    ppo_config_path = args.ppo_config if args.ppo_config else os.path.join(project_root, "configs", "ppo.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(ppo_config_path, "r") as f:
        ppo_config = yaml.safe_load(f)
        
    total_timesteps = args.total_timesteps if args.total_timesteps else ppo_config.get("total_timesteps", 1000000)
    
    logger.info(f"Loaded config: {config_path}")
    logger.info(f"Loaded PPO config: {ppo_config_path}")
    
    # 2. Create Vectorized Training Environment
    def make_env():
        return TriPendulumGoalEnv(config_path=config_path)
    env = DummyVecEnv([make_env])
    
    # 3. Handle Checkpoint and Resume Paths
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "ppo_best.zip")
    
    # Extract training parameters from yaml
    lr = ppo_config.get("learning_rate", 3e-4)
    n_steps = ppo_config.get("n_steps", 2048)
    batch_size = ppo_config.get("batch_size", 64)
    n_epochs = ppo_config.get("n_epochs", 10)
    gamma = ppo_config.get("gamma", 0.99)
    gae_lambda = ppo_config.get("gae_lambda", 0.95)
    clip_range = ppo_config.get("clip_range", 0.2)
    ent_coef = ppo_config.get("ent_coef", 0.0)
    policy_kw = ppo_config.get("policy_kwargs", None)
    
    # 4. Instantiate or Load Model
    if args.resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        model = PPO.load(
            checkpoint_path,
            env=env,
            tensorboard_log=args.tb_log
        )
    else:
        logger.info("Initializing new PPO model...")
        model = PPO(
            policy=ppo_config.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kw,
            verbose=ppo_config.get("verbose", 1),
            tensorboard_log=args.tb_log
        )
        
    # 5. Set up Curriculum & Analytics Callbacks
    callback = CurriculumAnalyticsCallback(config_path=config_path)
    
    # 6. Start Learning
    logger.info(f"Starting PPO training for {total_timesteps} steps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="PPO_run"
        )
        logger.info("Training complete!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving checkpoint...")
        interrupted_path = os.path.join(checkpoint_dir, "ppo_interrupted.zip")
        model.save(interrupted_path)
        logger.info(f"Checkpoint saved to {interrupted_path}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
