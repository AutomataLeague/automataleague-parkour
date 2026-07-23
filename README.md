# automataleague

A league of robots competing at athletic tasks in [MuJoCo](https://mujoco.org/) —
parkour first, with boxing, sumo, and sword-fighting to follow. Robots are
pluggable and interchangeable across tasks; policies are trained with
[TorchRL](https://github.com/pytorch/rl) (PPO), GPU-parallel via
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp).

## Status — Parkour

Boston Dynamics **Spot** learns to run parkour courses defined by a centerline path,
hitting checkpoints from a start line to a finish.

- **Walking + steering: solved.** Spot walks flat corridors, steers curves (L/S),
  and completes the full **winding circuit** (a 65 m closed lap) staying on the path.
- **Track library:** `straight`, `l_curved`, `s_curved`, `circuit` — all built from a
  polyline centerline; a curve is just a dense polyline.
- **Stage 1 — obstacles** (in progress): physical terrain (paving, hurdles, staircase,
  up-and-over ramp, banked traverse) placed along a track, scaled by a
  `level_difficulty` (0 = flat … 4 = hardest). Trained by warm-starting from the
  flat-track policy.

## Setup (uv)

```bash
uv sync                              # local / render-only (light core deps)
uv sync --extra train --extra gpu    # GPU box: adds torch, torchrl, MuJoCo-Warp
```

Headless rendering needs a GL backend: `MUJOCO_GL=egl` (default) or `osmesa`.
GPU training runs on the Jetson **gym1** box (see the `gym1-server` guidance; on
Jetson use the JetPack torch wheel and `.venv/bin/python`, not `uv run`).

## Rendering a track

Compose a track + robot and render screenshots (no physics, no training):

```bash
uv run python tools/render_scene.py --robot spot --track circuit
uv run python tools/render_scene.py --robot spot --track l_curved
uv run python tools/render_scene.py --robot spot --track circuit --level 3   # with obstacles
```

Outputs land in `renders/`.

## Training

```bash
cd training
# flat circuit from scratch
MUJOCO_GL=egl python train_ppo.py env.course.track=circuit env.max_episode_steps=3500

# obstacles, warm-started from a flat-track policy (difficulty curriculum L1 -> L4)
MUJOCO_GL=egl python train_ppo.py env.course.track=circuit env.course.level_difficulty=1 \
  env.max_episode_steps=3500 env.reward_weights.height=0.0 \
  network.init_checkpoint=/path/to/flat_circuit.pt logger.exp_name=parkour_obst_L1
```

Key wandb signals: `train/checkpoints_reached`, `eval/dist_to_finish`,
`train/{success,fell,off_path}_rate`, plus an `eval/video`.

## Demo videos

Render a trained checkpoint from several synchronized cameras (record-once,
render-many):

```bash
MUJOCO_GL=egl python training/demo_render.py --checkpoint training/checkpoints/ppo_final.pt \
  --cameras drone,over_shoulder,side,top --out-dir videos/
```

## Layout

```
assets/spot/                     # vendored Boston Dynamics Spot (Menagerie, Apache-2.0)
automataleague/
  robots/                        # robot registry: RobotSpec contract + per-robot specs
  envs/parkour/
    tracks.py                    # centerline tracks (straight/L/S/circuit) + turtle builder
    obstacles.py                 # Stage-1 physical terrain, scaled by level_difficulty
    config.py                    # ParkourConfig / RewardConfig / TerminationConfig
    scene.py                     # MjSpec: paint the track + attach robot + place obstacles
    spatial.py state.py navigation.py observation.py rewards.py termination.py
    parkour_warp.py              # batched GPU env (MuJoCo-Warp)
    parkour_cpu.py               # single-env CPU env (rendering)
    render.py                    # camera presets + record/render trajectory
training/                        # PPO pipeline: train_ppo.py, utils_ppo.py, config, demo_render
tools/                           # render_scene.py, camera/obstacle previews, GPU validators
tests/                           # CPU-side unit tests for the whole task brain
```

## Reward (locomotion + navigation)

Dense **forward-velocity toward the next checkpoint** (the driver that makes walking
emerge) + checkpoint/finish bonuses + small upright shaping − action/joint-velocity
penalties, with fall and off-path terminations. Off-path is the perpendicular
distance to the centerline; the finish success point sits just past the visual line
so the robot must cross it.

## Adding a robot or track

- **Robot:** write `make_<robot>() -> RobotSpec` in `automataleague/robots/`, register
  in `ROBOTS`. Obs/action sizes derive from the joint count.
- **Track:** add a `make_<track>() -> Track` (centerline polyline) in `tracks.py` and
  register in `TRACKS`; everything else (checkpoints, off-path, obstacles, rendering)
  generalizes automatically.

## Credits

Spot model © Boston Dynamics, from MuJoCo Menagerie (Apache-2.0). See
`assets/spot/LICENSE`.
