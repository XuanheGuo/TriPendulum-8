from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.tripendulum_env import TriPendulumGoalEnv
from training.callbacks import AutoCurriculumCallback, DiagnosticVideoCallback, EpisodeInfoCallback, make_checkpoint_callback
from training.curriculum import apply_curriculum, create_curriculum_state
from utils.logging_utils import ensure_dir, load_config


def make_single_env(cfg, curriculum_state=None, rank=0):
    env_cfg = apply_curriculum(cfg.get("env", {}), cfg.get("curriculum"), curriculum_state)
    env_cfg["reward"] = cfg.get("reward", {})
    env_cfg = deepcopy(env_cfg)
    base_seed = env_cfg.get("seed")
    env = TriPendulumGoalEnv(env_cfg)
    if base_seed is not None:
        env.reset(seed=int(base_seed) + int(rank))
    return Monitor(env)


def make_env_factory(cfg, curriculum_state, rank):
    def _init():
        return make_single_env(cfg, curriculum_state, rank)

    return _init


def make_train_env(cfg, curriculum_state=None):
    runtime_cfg = cfg.get("runtime", {})
    n_envs = max(1, int(runtime_cfg.get("n_envs", 1)))
    factories = [make_env_factory(cfg, curriculum_state, rank) for rank in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(factories)
    start_method = runtime_cfg.get("subproc_start_method")
    if start_method == "auto":
        start_method = "spawn" if os.name == "nt" else "forkserver"
    return SubprocVecEnv(factories, start_method=start_method)


def make_callbacks(cfg, algo, curriculum_state=None):
    checkpoint_dir = ensure_dir(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints"))
    n_envs = max(1, int(cfg.get("runtime", {}).get("n_envs", 1)))
    callbacks = [
        make_checkpoint_callback(
            max(int(algo.get("save_freq", 25000)) // n_envs, 1),
            checkpoint_dir,
            "sac",
            save_replay_buffer=bool(algo.get("save_replay_buffer", False)),
        ),
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
                require_pose_error=bool(curriculum_cfg.get("require_pose_error", False)),
                pose_error_key=curriculum_cfg.get("pose_error_key", "avg_final_pose_error"),
                max_collision_rate=float(curriculum_cfg.get("max_collision_rate", 0.1)),
                consecutive_passes_required=int(curriculum_cfg.get("consecutive_passes_required", 2)),
                min_steps_per_goal=int(curriculum_cfg.get("min_steps_per_goal", 25000)),
                checkpoint_dir=checkpoint_dir,
                algorithm="sac",
            )
        )
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("enabled", False):
        eval_env = DummyVecEnv([make_env_factory(cfg, curriculum_state, 10000)])
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=eval_cfg.get("best_model_save_path", checkpoint_dir),
                log_path=eval_cfg.get("log_path", "runs/eval"),
                eval_freq=max(int(eval_cfg.get("eval_freq", 50000)) // n_envs, 1),
                n_eval_episodes=int(eval_cfg.get("n_eval_episodes", 5)),
                deterministic=True,
            )
        )
    video_cfg = cfg.get("diagnostic_video", {})
    if video_cfg.get("enabled", False):
        env_cfg = apply_curriculum(cfg.get("env", {}), cfg.get("curriculum"), curriculum_state)
        callbacks.append(
            DiagnosticVideoCallback(
                algorithm="sac",
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
    parser.add_argument("--config", default="configs/sac.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--save-path", default="checkpoints/sac_best.zip")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-replay-buffer", default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("runtime", {})
    if args.n_envs is not None:
        cfg["runtime"]["n_envs"] = max(1, args.n_envs)
    if args.torch_num_threads is not None:
        cfg["runtime"]["torch_num_threads"] = max(1, args.torch_num_threads)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    algo = cfg.get("algorithm", {})
    runtime_cfg = cfg.get("runtime", {})
    n_envs = max(1, int(runtime_cfg.get("n_envs", 1)))
    worker_threads = max(1, int(runtime_cfg.get("worker_num_threads", 1)))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(worker_threads)
    try:
        import torch

        torch.set_num_threads(max(1, int(runtime_cfg.get("torch_num_threads", 1))))
    except ImportError:
        pass
    total_timesteps = args.total_timesteps or int(algo.get("total_timesteps", 300000))
    tensorboard_dir = ensure_dir(cfg.get("paths", {}).get("tensorboard_dir", "runs"))
    checkpoint_dir = ensure_dir(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints"))
    curriculum_state = create_curriculum_state(cfg.get("curriculum"), checkpoint_dir)

    env = make_train_env(cfg, curriculum_state)
    print(
        f"SAC runtime: n_envs={n_envs}, train_freq={algo.get('train_freq', 1)}, "
        f"gradient_steps={algo.get('gradient_steps', 1)}, torch_threads={runtime_cfg.get('torch_num_threads', 1)}, "
        f"device={runtime_cfg.get('device', 'auto')}"
    )
    if args.resume_from:
        print(f"Resuming SAC from {args.resume_from}")
        model = SAC.load(
            args.resume_from,
            env=env,
            tensorboard_log=tensorboard_dir,
            device=runtime_cfg.get("device", "auto"),
            train_freq=int(algo.get("train_freq", 1)),
            gradient_steps=int(algo.get("gradient_steps", 1)),
            verbose=1,
        )
        if args.resume_replay_buffer and os.path.exists(args.resume_replay_buffer):
            print(f"Loading replay buffer from {args.resume_replay_buffer}")
            model.load_replay_buffer(args.resume_replay_buffer)
        else:
            fresh_buffer_warmup = int(algo.get("resume_learning_starts", 5000))
            model.learning_starts = model.num_timesteps + fresh_buffer_warmup
            print(f"Using a fresh replay buffer; collecting {fresh_buffer_warmup} steps before updates resume")
    else:
        model = SAC(
            algo.get("policy", "MlpPolicy"),
            env,
            learning_rate=float(algo.get("learning_rate", 3e-4)),
            buffer_size=int(algo.get("buffer_size", 500000)),
            batch_size=int(algo.get("batch_size", 256)),
            gamma=float(algo.get("gamma", 0.99)),
            tau=float(algo.get("tau", 0.005)),
            train_freq=int(algo.get("train_freq", 1)),
            gradient_steps=int(algo.get("gradient_steps", 1)),
            learning_starts=int(algo.get("learning_starts", 5000)),
            ent_coef=algo.get("ent_coef", "auto"),
            tensorboard_log=tensorboard_dir,
            device=runtime_cfg.get("device", "auto"),
            verbose=1,
        )
    callbacks = make_callbacks(cfg, algo, curriculum_state)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True, reset_num_timesteps=not bool(args.resume_from))
    model.save(args.save_path)
    env.close()


if __name__ == "__main__":
    main()
