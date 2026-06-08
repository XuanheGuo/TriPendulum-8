import os
import argparse
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.tripendulum_env import TriPendulumGoalEnv
from training.callbacks import CurriculumAnalyticsCallback
from utils.logging_utils import setup_logger

logger = setup_logger("train_sac")

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on TriPendulum-8")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--sac-config", type=str, default=None, help="Path to SAC YAML")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total timesteps to train")
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint if exists")
    parser.add_argument("--tb-log", type=str, default="runs/sac", help="TensorBoard log dir")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Configurations
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(curr_dir, ".."))
    
    config_path = args.config if args.config else os.path.join(project_root, "configs", "default.yaml")
    sac_config_path = args.sac_config if args.sac_config else os.path.join(project_root, "configs", "sac.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(sac_config_path, "r") as f:
        sac_config = yaml.safe_load(f)
        
    total_timesteps = args.total_timesteps if args.total_timesteps else sac_config.get("total_timesteps", 1000000)
    
    logger.info(f"Loaded config: {config_path}")
    logger.info(f"Loaded SAC config: {sac_config_path}")
    
    # 2. Create Vectorized Training Environment
    def make_env():
        return TriPendulumGoalEnv(config_path=config_path)
    env = DummyVecEnv([make_env])
    
    # 3. Handle Checkpoint and Resume Paths
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "sac_best.zip")
    
    # Extract training parameters from yaml
    lr = sac_config.get("learning_rate", 3e-4)
    buffer_size = sac_config.get("buffer_size", 1000000)
    batch_size = sac_config.get("batch_size", 256)
    tau = sac_config.get("tau", 0.005)
    gamma = sac_config.get("gamma", 0.99)
    train_freq = sac_config.get("train_freq", 1)
    gradient_steps = sac_config.get("gradient_steps", 1)
    ent_coef = sac_config.get("ent_coef", "auto")
    policy_kw = sac_config.get("policy_kwargs", None)
    
    # 4. Instantiate or Load Model
    if args.resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        model = SAC.load(
            checkpoint_path,
            env=env,
            tensorboard_log=args.tb_log
        )
    else:
        logger.info("Initializing new SAC model...")
        # Handle ent_coef parsing
        if ent_coef == "auto":
            ent_coef_val = "auto"
        else:
            ent_coef_val = float(ent_coef)
            
        model = SAC(
            policy=sac_config.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=lr,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef_val,
            policy_kwargs=policy_kw,
            verbose=sac_config.get("verbose", 1),
            tensorboard_log=args.tb_log
        )
        
    # 5. Set up Curriculum & Analytics Callbacks
    callback = CurriculumAnalyticsCallback(config_path=config_path)
    
    # 6. Start Learning
    logger.info(f"Starting SAC training for {total_timesteps} steps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="SAC_run"
        )
        logger.info("Training complete!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving checkpoint...")
        interrupted_path = os.path.join(checkpoint_dir, "sac_interrupted.zip")
        model.save(interrupted_path)
        logger.info(f"Checkpoint saved to {interrupted_path}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
