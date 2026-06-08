from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.tripendulum_env import TriPendulumGoalEnv
from training.callbacks import AutoCurriculumCallback, DiagnosticVideoCallback, EpisodeInfoCallback, make_checkpoint_callback
from training.curriculum import apply_curriculum, create_curriculum_state
from utils.logging_utils import ensure_dir, load_config


def make_env(cfg, curriculum_state=None):
    env_cfg = apply_curriculum(cfg.get("env", {}), cfg.get("curriculum"), curriculum_state)
    env_cfg["reward"] = cfg.get("reward", {})
    return Monitor(TriPendulumGoalEnv(env_cfg))


def make_callbacks(cfg, algo, curriculum_state=None):
    checkpoint_dir = ensure_dir(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints"))
    callbacks = [
        make_checkpoint_callback(int(algo.get("save_freq", 25000)), checkpoint_dir, "ppo"),
        EpisodeInfoCallback(),
    ]
    curriculum_cfg = cfg.get("curriculum", {})
    if curriculum_state is not None and curriculum_cfg.get("auto_advance", True):
        callbacks.append(
            AutoCurriculumCallback(
                curriculum_state=curriculum_state,
                env_config=cfg.get("env", {}),
                reward_config=cfg.get("reward", {}),
                eval_freq=int(curriculum_cfg.get("eval_freq", 25000)),
                n_eval_episodes=int(curriculum_cfg.get("n_eval_episodes", 5)),
                max_steps=int(curriculum_cfg.get("max_steps", 1000)),
                success_threshold=float(curriculum_cfg.get("success_threshold", 0.8)),
                max_pose_error=float(curriculum_cfg.get("max_pose_error", 0.25)),
                max_collision_rate=float(curriculum_cfg.get("max_collision_rate", 0.1)),
                consecutive_passes_required=int(curriculum_cfg.get("consecutive_passes_required", 2)),
                min_steps_per_goal=int(curriculum_cfg.get("min_steps_per_goal", 25000)),
                checkpoint_dir=checkpoint_dir,
                algorithm="ppo",
            )
        )
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("enabled", False):
        eval_env = make_env(cfg, curriculum_state)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=eval_cfg.get("best_model_save_path", checkpoint_dir),
                log_path=eval_cfg.get("log_path", "runs/eval"),
                eval_freq=int(eval_cfg.get("eval_freq", 50000)),
                n_eval_episodes=int(eval_cfg.get("n_eval_episodes", 5)),
                deterministic=True,
            )
        )
    video_cfg = cfg.get("diagnostic_video", {})
    if video_cfg.get("enabled", False):
        env_cfg = apply_curriculum(cfg.get("env", {}), cfg.get("curriculum"), curriculum_state)
        callbacks.append(
            DiagnosticVideoCallback(
                algorithm="ppo",
                env_config=env_cfg,
                reward_config=cfg.get("reward", {}),
                curriculum_config=cfg.get("curriculum", {}),
                curriculum_state=curriculum_state,
                eval_freq=int(video_cfg.get("eval_freq", 50000)),
                max_steps=int(video_cfg.get("max_steps", 1000)),
                n_eval_episodes=int(video_cfg.get("n_eval_episodes", 3)),
                mode=video_cfg.get("mode", "curriculum_worst"),
                worst_k=int(video_cfg.get("worst_k", 2)),
                goals=video_cfg.get("goals", ["DDD", "DUD", "UUU"]),
                fallback_goals=video_cfg.get("fallback_goals", ["DDD", "UUU"]),
                save_dir=video_cfg.get("save_dir", "videos/diagnostics"),
                fps=int(video_cfg.get("fps", 30)),
            )
        )
    return callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--save-path", default="checkpoints/ppo_final.zip")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    algo = cfg.get("algorithm", {})
    total_timesteps = args.total_timesteps or int(algo.get("total_timesteps", 200000))
    tensorboard_dir = ensure_dir(cfg.get("paths", {}).get("tensorboard_dir", "runs"))
    checkpoint_dir = ensure_dir(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints"))
    curriculum_state = create_curriculum_state(cfg.get("curriculum"), checkpoint_dir)

    env = make_env(cfg, curriculum_state)
    if args.resume_from:
        print(f"Resuming PPO from {args.resume_from}")
        model = PPO.load(args.resume_from, env=env, tensorboard_log=tensorboard_dir, verbose=1)
    else:
        model = PPO(
            algo.get("policy", "MlpPolicy"),
            env,
            learning_rate=float(algo.get("learning_rate", 3e-4)),
            n_steps=int(algo.get("n_steps", 2048)),
            batch_size=int(algo.get("batch_size", 256)),
            gamma=float(algo.get("gamma", 0.99)),
            gae_lambda=float(algo.get("gae_lambda", 0.95)),
            ent_coef=float(algo.get("ent_coef", 0.0)),
            clip_range=float(algo.get("clip_range", 0.2)),
            tensorboard_log=tensorboard_dir,
            verbose=1,
        )
    callbacks = make_callbacks(cfg, algo, curriculum_state)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True, reset_num_timesteps=not bool(args.resume_from))
    model.save(args.save_path)
    env.close()


if __name__ == "__main__":
    main()
