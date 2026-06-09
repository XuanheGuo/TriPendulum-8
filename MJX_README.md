# TriPendulum-8 MJX/JAX PPO Backend

This directory is a separate GPU training backend. It does not replace or remove the original Gymnasium, native MuJoCo, and Stable-Baselines3 implementation.

## Design

- Physics: MuJoCo MJX through Brax `PipelineEnv`
- Learning: Brax PPO in JAX
- Parallelism: 4096 GPU environments by default
- Action: one normalized policy action mapped to cart force `[-40, 40] N`
- Goals: all eight absolute link postures trained concurrently
- Observation: the same 20 values used by the original environment
- Rendering: the original native MuJoCo environment, outside the training loop

Each reset uniformly selects one of the eight goal encodings. Initial-state difficulty is sampled independently:

```text
20% near target
20% halfway to target
20% quarter way to target
40% full swing-up from the down region
```

This distribution keeps target stabilization, recovery, and full swing-up data present in every PPO batch. It avoids replay-buffer contamination and catastrophic forgetting from a strictly sequential curriculum.

## Local GPU

Use Linux or WSL2 with an NVIDIA GPU and a CUDA-compatible JAX installation:

```bash
pip install -U "jax[cuda12]"
pip install -r requirements-mjx.txt
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Train:

```bash
python -m mjx_backend.train_ppo \
  --config configs/mjx_ppo.yaml \
  --num-timesteps 50000000 \
  --output-dir outputs/mjx_ppo_run
```

Evaluate all goals:

```bash
python -m mjx_backend.evaluate \
  --config configs/mjx_ppo.yaml \
  --checkpoint outputs/mjx_ppo_run/checkpoints/final.params \
  --episodes 20
```

Render with native MuJoCo:

```bash
MUJOCO_GL=egl python -m mjx_backend.render_video \
  --checkpoint outputs/mjx_ppo_run/checkpoints/final.params \
  --goal DDU \
  --output videos/mjx_ppo_DDU.mp4
```

## Colab

Open `notebooks/colab_mjx_train.ipynb`, select a GPU runtime, set a unique `RUN_NAME`, and run all cells. Outputs are stored under:

```text
MyDrive/TriPendulum-8-MJX/<RUN_NAME>/
```

The first JAX compilation can take several minutes. Throughput measurements are meaningful only after compilation finishes.

## Important Compatibility Notes

MJX and native MuJoCo are intended to represent the same model, but numerical trajectories are not bit-identical. Final policies must therefore be evaluated again in the original native MuJoCo environment before sim-to-real work. Diagnostic rendering is deliberately kept out of the thousands-of-environments training loop.
