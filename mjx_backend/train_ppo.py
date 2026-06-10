"""Train TriPendulum-8 with GPU-vectorized MJX and Brax PPO."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import yaml
from brax.io import model
from mujoco_playground._src import wrapper
from tensorboardX import SummaryWriter

from mjx_backend.env import TriPendulumMJXEnv
from mjx_backend.training import make_train_fn


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mjx_ppo.yaml")
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-stage", type=int, default=None,
                        help="Resume from this stage (1-indexed). Loads stage_N_final.params from checkpoint_dir as starting params.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_timesteps is not None:
        config["ppo"]["num_timesteps"] = args.num_timesteps
    output_dir = Path(args.output_dir or config["paths"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print("JAX devices:", jax.devices())
    if jax.default_backend() != "gpu":
        raise RuntimeError("MJX training requires a JAX GPU backend; select a Colab GPU runtime.")

    writer = SummaryWriter(str(log_dir))
    history_path = output_dir / "metrics.jsonl"
    start_time = time.time()

    def progress(step: int, metrics: dict) -> None:
        elapsed = max(time.time() - start_time, 1e-6)
        serializable = {key: float(np.asarray(value)) for key, value in metrics.items()}
        serializable.update({"step": int(step), "elapsed_seconds": elapsed, "steps_per_second": step / elapsed})
        with open(history_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(serializable) + "\n")
        for key, value in serializable.items():
            if key not in {"step", "elapsed_seconds"}:
                writer.add_scalar(key, value, step)
        writer.flush()
        reward = serializable.get("eval/episode_reward", float("nan"))
        print(f"step={step:,} reward={reward:.3f} speed={step / elapsed:,.0f} steps/s")

    def save_checkpoint(step: int, _make_policy, params) -> None:
        path = checkpoint_dir / f"ppo_step_{int(step)}.params"
        model.save_params(path, params)
        model.save_params(checkpoint_dir / "latest.params", params)

    default_stages = [
        {"goals": ["DDD"], "timesteps": 50_000_000},
        {"goals": ["DDD", "DDU", "DUD", "UDD"], "timesteps": 150_000_000},
        {"goals": ["DDD", "DDU", "DUD", "UDD", "DUU", "UDU", "UUD", "UUU"], "timesteps": 300_000_000},
    ]
    stages = config.get("curriculum", {}).get("stages", default_stages)

    resume_stage = args.resume_stage  # 1-indexed; None means start fresh
    params = None
    if resume_stage is not None:
        prev_ckpt = checkpoint_dir / f"stage_{resume_stage - 1}_final.params"
        latest_ckpt = checkpoint_dir / "latest.params"
        if resume_stage > 1 and prev_ckpt.exists():
            params = model.load_params(prev_ckpt)
            print(f"Resuming from {prev_ckpt}")
        elif latest_ckpt.exists():
            params = model.load_params(latest_ckpt)
            print(f"Resuming from {latest_ckpt}")
        else:
            print(f"Warning: no checkpoint found for stage {resume_stage}, starting fresh")

    start_stage = (resume_stage - 1) if resume_stage is not None else 0
    for i, stage in enumerate(stages[start_stage:], start=start_stage):
        stage_goals = stage["goals"]
        stage_timesteps = int(stage["timesteps"])
        print(f"\n=== Stage {i+1}/{len(stages)}: goals={stage_goals}, timesteps={stage_timesteps:,} ===")
        stage_env = TriPendulumMJXEnv({**config["env"], "allowed_goals": stage_goals})
        stage_train_fn = make_train_fn(config, num_timesteps=stage_timesteps, restore_params=params)
        make_policy, params, metrics = stage_train_fn(
            environment=stage_env,
            progress_fn=progress,
            policy_params_fn=save_checkpoint,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        model.save_params(checkpoint_dir / f"stage_{i+1}_final.params", params)
        print(f"Stage {i+1} complete. metrics={metrics}")
    model.save_params(checkpoint_dir / "final.params", params)
    writer.close()
    print("Training complete:", checkpoint_dir / "final.params")
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
