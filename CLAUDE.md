# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Goal-conditioned RL for a cart-mounted serial triple inverted pendulum. A single actuator (horizontal cart force) must learn to swing and balance three links to 8 goal poses (D/U per link, e.g. UUU = all upright). Main algorithm is SAC; PPO is the baseline. There is also a GPU-parallel JAX/MJX backend in `mjx_backend/`.

## Commands

```bash
# Install
pip install -r requirements.txt

# Train (SAC main, PPO baseline)
python training/train_sac.py --config configs/sac.yaml --total-timesteps 2000000
python training/train_ppo.py --config configs/ppo.yaml

# Evaluate / render
python evaluation/evaluate.py --model checkpoints/sac_best.zip
python evaluation/render_video.py --model checkpoints/sac_best.zip --goal UUU
python evaluation/transition_matrix.py --model checkpoints/sac_best.zip

# Curriculum inspection
python training/curriculum_cli.py show
python training/curriculum_cli.py advance
```

## Architecture

**Environment** (`envs/tripendulum_env.py`): Gymnasium `gym.Env`. 20-dim observation (cart state, joint angles/velocities in relative + absolute coordinates, 3-bit goal binary). 1D continuous action (force). Multi-component reward in `utils/reward.py` (pose error, swing energy, velocity, action cost, boundary, stability).

**Training loop** (`training/train_sac.py`): loads YAML config → builds vectorized envs (SubprocVecEnv on Unix) → SB3 SAC agent → runs with three callbacks (curriculum auto-advance, diagnostic video, checkpoint).

**Curriculum** (`training/curriculum.py`): Sequential stages — DDD → {DDU, DUD, UDD} → {DUU, UDU, UUD} → all 8. Auto-advances every 50k steps when success rate ≥ 0.625 for 2 consecutive passes. State persisted to `checkpoints/curriculum_state.json`.

**Goals** (`envs/goals.py`): 8 poses defined by absolute link angles; binary 3-bit encoding appended to observation.

**MJX backend** (`mjx_backend/`): Independent GPU-parallel reimplementation using JAX + MuJoCo MJX. Has its own config and training pipeline; see `MJX_README.md`.

## Output Layout

```
checkpoints/          # model zips, curriculum_state.json
runs/                 # TensorBoard logs
videos/diagnostics/   # per-eval MP4s + JSON metadata
```

## Key Config Parameters (`configs/sac.yaml`)

- `frame_skip: 4`, `x_max: 4.8`, `f_max: 40.0`
- `buffer_size: 500000`, `batch_size: 256`, `learning_rate: 3e-4`
- `eval_freq: 50000`, `success_threshold: 0.625`, `consecutive_passes_required: 2`
