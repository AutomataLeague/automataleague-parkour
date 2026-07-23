# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

`automataleague` — a league of robots competing at athletic tasks in MuJoCo
(parkour, boxing, sumo, sword-fighting). Environments are the deliverable; robots
are pluggable; policies are trained with TorchRL PPO and GPU-parallel MuJoCo-Warp.
The training pipeline is copied/adapted from `../smaller_projects/spaceX`.

## Setup

```bash
uv sync                              # Phase A: render-only core
uv sync --extra train --extra gpu    # Phase B: training + MuJoCo-Warp (GPU box)
```

## Commands

```bash
# Render scene screenshots for design review (Phase A)
uv run python tools/render_scene.py --robot spot

# Inspect a compiled parkour model
uv run python -m automataleague.envs.parkour.scene
```

Headless rendering: `MUJOCO_GL=egl` (default) or `osmesa`.

## Architecture

### Robots (`automataleague/robots/`)
A `RobotSpec` (base.py) is the whole contract a task needs: model path, base body,
nominal height, ordered joint/actuator names, home stance, action scale. Obs/action
dims are **derived** from joint count. Register robots in `__init__.py::ROBOTS`.
`spot.py` is the first entry (12 position-controlled leg joints).

### Parkour task (`automataleague/envs/parkour/`)
- `config.py` — `ParkourConfig` (corridor + checkpoint geometry, single source of
  truth for scene AND reward), `RewardConfig`, `TerminationConfig`.
- `scene.py` — `build_parkour_model(robot, cfg)` builds the corridor as an `MjSpec`
  and `attach`es the robot at the spawn frame → `(MjModel, SceneInfo)`. `SceneInfo`
  exposes resolved ids / home qpos so downstream code ignores name prefixes.
- `parkour_warp.py` / `parkour_cpu.py` / `rewards.py` — Phase B (batched GPU env,
  CPU render env, reward terms).

### Training (`training/`)
Task-agnostic PPO copied from spaceX: direct-stepping on-policy loop, GAE,
`ClipPPOLoss`, LR/clip annealing, wandb, video logging, checkpoints. `make_ppo_models`
sizes networks from env specs, so it works for any robot/obs dim. (Phase B.)

## Design decisions

- **Robot ↔ task decoupling** via `RobotSpec`. One robot per training run; compare
  robots by running separate configs (mixing robots in one batched env is out of scope).
- **Programmatic scene** (not static XML) so checkpoint coords can't drift between
  scene and reward.
- **Position-target actions**: `q_target = home + action_scale * action` around the
  standing stance — the standard trick that makes quadruped locomotion learnable.

## Conventions

- Design/plan docs live in `docs/superpowers/` and are **gitignored** (scaffolding,
  not deliverables) — do not commit them.
- Vendored models (`assets/`) keep their upstream LICENSE.
