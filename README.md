# automataleague

A league of robots competing at athletic tasks in [MuJoCo](https://mujoco.org/) —
parkour first, with boxing, sumo, and sword-fighting to follow. Robots are
pluggable and interchangeable across tasks; policies are trained with
[TorchRL](https://github.com/pytorch/rl) (PPO), GPU-parallel via
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp).

## Status

**Parkour v0** (in progress): a robot (Boston Dynamics **Spot** to start) learns to
walk down a flat corridor from a start line to a finish line, hitting checkpoints
along the way. See `docs/superpowers/specs/` for the design.

## Setup (uv)

```bash
# Local / render-only (Phase A): light core deps
uv sync

# GPU box (Phase B): add the training + Warp stack
uv sync --extra train --extra gpu
```

## Rendering a scene (Phase A)

Compose the corridor + robot and render screenshots for design review — no
physics, no training:

```bash
uv run python tools/render_scene.py --robot spot
uv run python tools/render_scene.py --robot spot --length 12 --half-width 1.0
```

Outputs land in `renders/`. Headless machines need a GL backend:
`MUJOCO_GL=egl` (default) or `MUJOCO_GL=osmesa`.

## Layout

```
assets/spot/            # vendored Boston Dynamics Spot (MuJoCo Menagerie, see LICENSE)
automataleague/
  robots/               # robot registry: base contract + per-robot specs
  envs/parkour/         # parkour task: config, scene composition, (envs + rewards: Phase B)
training/               # task-agnostic PPO pipeline (Phase B)
tools/render_scene.py   # static scene renderer (Phase A)
```

## Adding a robot

Write a `make_<robot>() -> RobotSpec` factory in `automataleague/robots/`, register
it in `ROBOTS`, and it drops into every task. Observation/action sizes are derived
from the robot's joint count.

## Credits

Spot model © Boston Dynamics, from MuJoCo Menagerie (Apache-2.0). See
`assets/spot/LICENSE`.
