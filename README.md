# TriPendulum-8

TriPendulum-8 is a Goal-Conditioned reinforcement learning project for a cart-mounted serial triple inverted pendulum. The system has one active actuator only: horizontal force on the cart.

The action is strictly:

```text
action = [F_cart]
```

No active torque is applied to any pendulum hinge joint. The MuJoCo model exposes passive hinge joints for the three links and a single motor actuator on the cart slider.

## Angle Convention

MuJoCo stores native relative joint angles:

- `q1`: link 1 relative to the cart/world vertical down reference
- `q2`: link 2 relative to link 1
- `q3`: link 3 relative to link 2

Goal poses are defined by absolute link angles in world coordinates:

```python
theta1_abs = q1
theta2_abs = q1 + q2
theta3_abs = q1 + q2 + q3
```

Names such as `DDD`, `DUD`, and `UUU` refer to the three physical pendulum links' absolute world orientations, not the three relative hinge angles.

## Goals

`D = Down = 0`, `U = Up = pi`.

The eight goal poses are:

- `DDD = [0, 0, 0]`
- `DDU = [0, 0, pi]`
- `DUD = [0, pi, 0]`
- `UDD = [pi, 0, 0]`
- `DUU = [0, pi, pi]`
- `UDU = [pi, 0, pi]`
- `UUD = [pi, pi, 0]`
- `UUU = [pi, pi, pi]`

Reward, success checks, evaluation metrics, and transition tests are all based on these absolute angles.

## Physical Constraints

The cart track is finite: `x in [-x_max, x_max]`. The environment applies a continuous boundary penalty and terminates with a collision penalty when the cart exceeds this range. This prevents policies from relying on an unrealistic infinite track or extreme cart runs.

The real device is modeled as staggered multi-layer pendulum hardware, so the links are not treated as coplanar rods. For that reason:

- no inter-link self-collision penalty is used,
- visual link crossing is not considered a collision,
- inter-link geometric overlap does not terminate an episode,
- the MuJoCo XML does not enable pendulum self-collision termination.

The main simulation constraints are cart track boundaries, cart force limits, action-rate penalty, angular-velocity limits, energy cost, and optional future noise/delay/randomization.

## Algorithms

- PPO is provided as a baseline.
- SAC is the main continuous-control algorithm.

Both train a goal-conditioned policy:

```text
policy(obs, goal_binary) -> F_cart
```

## Quick Start

```bash
pip install -r requirements.txt

python training/train_ppo.py --config configs/ppo.yaml
python training/train_sac.py --config configs/sac.yaml
python evaluation/evaluate.py --model checkpoints/sac_best.zip
```

For Colab Pro, open `notebooks/colab_train.ipynb`, set the top parameter cell if needed, then run `Runtime -> Run all`. The notebook installs dependencies, checks MuJoCo, writes Colab-specific configs, starts TensorBoard, trains SAC/PPO, records dynamic diagnostic videos, evaluates all 8 goals, builds the 8x8 transition heatmap, renders a final video, and packages results.

The notebook defaults to Google Drive persistence while keeping high-frequency TensorBoard writes on the Colab local SSD:

```python
USE_GOOGLE_DRIVE = True
PROJECT_DIR = "/content/TriPendulum-8"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/TriPendulum-8-outputs"
RUN_NAME = "sac_colab_pro_run"
LOCAL_RUNTIME_DIR = "/content/TriPendulum-8-runtime"
```

All important outputs are written under:

```text
/content/drive/MyDrive/TriPendulum-8-outputs/sac_colab_pro_run/
├── checkpoints/
├── runs/
├── videos/
│   └── diagnostics/
├── evaluation/
└── configs/
```

This keeps model files, checkpoints, TensorBoard logs, diagnostic videos, evaluation CSV/PNG files, rendered videos, and config copies after the Colab runtime disconnects.

TensorBoard should read local logs from `/content/TriPendulum-8-runtime/.../runs`, not directly from Google Drive. The notebook starts a small background sync that copies local event files to Drive every few minutes. This avoids the common Colab failure where TensorBoard shows `Data could not be loaded` while reading Drive-mounted event files.

## Diagnostic Videos

Training can periodically generate dynamic deterministic diagnostic videos. Because TriPendulum-8 is a goal-conditioned multi-posture control problem, it is usually not enough to keep watching only `UUU` or a few fixed targets.

The recommended mode is `curriculum_worst`: at each diagnostic trigger, the callback evaluates the currently relevant goals, ranks them by failure, and records the worst goals for inspection.

```yaml
diagnostic_video:
  enabled: true
  eval_freq: 50000
  max_steps: 1000
  n_eval_episodes: 3
  save_dir: "videos/diagnostics"
  fps: 30
  mode: "curriculum_worst"
  worst_k: 2
  fallback_goals:
    - DDD
    - UUU
```

Supported modes:

- `fixed`: record configured `goals`.
- `curriculum`: record the current curriculum stage goals.
- `all`: record all 8 goals.
- `worst`: evaluate all 8 goals and record the lowest-success goals.
- `curriculum_worst`: if curriculum is enabled, evaluate current stage goals and record the worst goals; otherwise fall back to `worst`.

At every `eval_freq` training timesteps, the callback first creates a separate evaluation environment, computes success rate, average reward, pose error, episode length, track collision rate, and max cart displacement for relevant goals, then records the selected goals. Files are written as:

```text
videos/diagnostics/sac_step_150000_stage_2_goal_DUD.mp4
videos/diagnostics/sac_step_150000_stage_2_goal_DUD.json
videos/diagnostics/diagnostic_step_150000.json
```

The per-video JSON sidecar records algorithm, timestep, curriculum status, stage id, stage goals, selected goal, selection reason, success, reward, episode length, final absolute-pose error, max cart displacement, track collision, and overspin status. The `diagnostic_step_*.json` report records evaluated goals, video goals, and all per-goal metrics.

These videos are intended to inspect whether the policy:

- fails most on a current curriculum-stage posture,
- approaches or hits the finite track boundary,
- exploits reward through excessive spinning,
- gets close to the absolute target pose but cannot stabilize,
- fails because cart displacement grows too large.

## Automatic Sequential Curriculum

The default configuration trains goals in this order:

```text
DDD -> DDU -> DUD -> UDD -> DUU -> UDU -> UUD -> UUU
```

The current new goal is evaluated deterministically every `10000` timesteps. It advances after two consecutive evaluations satisfy the configured success, pose-error, and collision thresholds. Previously learned goals remain in the training sampler to reduce catastrophic forgetting.

```yaml
curriculum:
  enabled: true
  mode: sequential
  auto_advance: true
  retain_previous_goals: true
  success_threshold: 0.8
  max_collision_rate: 0.2
  consecutive_passes_required: 2
```

Progress is persisted in `checkpoints/curriculum_state.json`. TensorBoard exposes `curriculum/stage_id`, `curriculum/eval_success_rate`, `curriculum/eval_pose_error`, `curriculum/eval_collision_rate`, and `curriculum/consecutive_passes`.

Start local SAC training with:

```bash
python training/train_sac.py \
  --config configs/sac.yaml \
  --total-timesteps 2000000 \
  --save-path checkpoints/sac_curriculum_final.zip
```

The default logical track is `x in [-4.8, 4.8]`. The visual XML track and camera match this range. Boundary reward weight and collision penalty are intentionally stronger so the extra room supports swing-up without making wall-running attractive.

## Extensions

Planned extensions include HER, MPC/iLQR warm starts, domain randomization, system identification, observation/action delay, and sim-to-real transfer.
