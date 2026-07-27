# automataleague

A league of robots competing at athletic tasks in [MuJoCo](https://mujoco.org/).
Robots are pluggable and interchangeable across tasks; policies are trained with
[TorchRL](https://github.com/pytorch/rl) (PPO), GPU-parallel via
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp).

## Tasks

| Task | Status | Description |
|---|---|---|
| **Parkour** | ✅ implemented | Follow a centerline course (straight → curves → winding circuit) from start to finish, hitting checkpoints; Stage-1 physical obstacles in progress. `automataleague/envs/parkour/` |
| Boxing | ⬜ planned | Full-contact striking between two robots. |
| Sumo | ⬜ planned | Push the opponent out of a ring. |
| Sword-fighting | ⬜ planned | Armed duelling. |

Each task lives in `automataleague/envs/<task>/` and reuses the shared robot
registry and PPO pipeline.

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

## Using environments

Reusable environments and PPO code live in the `automataleague` package; scripts
that drive them live in `examples/` and `tools/`.

```python
from automataleague import make_env, list_environments

print([s.env_id for s in list_environments()])   # ['parkour-1']
env = make_env("parkour-1", robot="spot", level=4)
```

`automataleague.training` (models/env/ppo/curriculum) is the task-agnostic PPO
pipeline: direct-stepping on-policy loop, GAE, `ClipPPOLoss`, LR/clip annealing,
wandb, video logging, checkpoints. `make_ppo_models` sizes networks from env specs,
so it works for any robot/obs dim.

## Training

Entry points live in `examples/`, run from the repo root with `uv run`; Hydra
config is `examples/config_ppo.yaml`.

```bash
# single difficulty level, standard TorchRL PPO scheme
uv run python examples/ppo_single.py env.course.level_difficulty=2

# curriculum across difficulty levels, warm-starting each stage from the last
uv run python examples/ppo_curriculum.py
```

Key wandb signals: `train/checkpoints_reached`, `eval/dist_to_finish`,
`train/{success,fell,off_path}_rate`, plus an `eval/video`. Checkpoints land in
`checkpoints/`.

## Tools

```bash
# render a trained checkpoint from several synchronized cameras (record-once, render-many)
MUJOCO_GL=egl uv run python tools/demo_render.py --checkpoint checkpoints/ppo_final.pt \
  --cameras drone,over_shoulder,side,top --out-dir videos/

# race two trained agents head-to-head or time-trial on the same track
MUJOCO_GL=egl uv run python tools/eval_1v1.py --track circuit \
  --agents runA.pt runB.pt --names Alice Bob --level 0 --out results/race.json

# rank checkpoints on a track/difficulty by finish rate then median lap time
MUJOCO_GL=egl uv run python tools/leaderboard.py \
  --entries tools/leaderboard_entries.json --track circuit --level 2 --seeds 5
```

## Layout

```
assets/spot/                     # vendored Boston Dynamics Spot (Menagerie, Apache-2.0)
automataleague/
  envs/
    registry.py                  # EnvSpec registry: make_env / list_environments
    parkour/
      tracks.py                  # centerline tracks (straight/L/S/circuit) + turtle builder
      obstacles.py                # Stage-1 physical terrain, scaled by level_difficulty
      config.py                  # ParkourConfig / RewardConfig / TerminationConfig
      scene.py                   # MjSpec: paint the track + attach robot + place obstacles
      spatial.py state.py navigation.py observation.py rewards.py termination.py
      parkour_warp.py            # batched GPU env (MuJoCo-Warp)
      parkour_cpu.py             # single-env CPU env (rendering)
      render.py                  # camera presets + record/render trajectory
  robots/                        # robot registry: RobotSpec contract + per-robot specs
  training/                      # task-agnostic PPO: models.py, env.py, ppo.py, curriculum.py
examples/                        # entry points: ppo_single.py, ppo_curriculum.py, config_ppo.yaml
tools/                           # demo_render.py, eval_1v1.py, leaderboard.py, render_scene.py, ...
checkpoints/                     # trained checkpoints (*.pt, gitignored)
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
