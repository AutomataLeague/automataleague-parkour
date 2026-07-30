<p align="center">
  <img src="assets/automataleague_logo.png" alt="Automata League" width="340">
</p>

# Automata League Parkour

The parkour competition environment of the **Automata League**: a robot learns to run
**parkour courses** (tracks with obstacles it must overcome to reach the finish) in
[MuJoCo](https://mujoco.org/). Boston Dynamics **Spot** is the example robot, but you can
plug in your own (see [Adding a custom robot](#adding-a-custom-robot)). The examples train
with [TorchRL](https://github.com/pytorch/rl) PPO, GPU parallel via
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp); any other TorchRL agent can
be used the same way (see [Training](#training)).

## Setup

```bash
uv sync                              # core: environment building and rendering
uv sync --extra train --extra gpu    # adds torch, torchrl, MuJoCo Warp (GPU box)
```

Headless rendering needs a GL backend: `MUJOCO_GL=egl`.

## The parkour environment

The parkour task lives in `automataleague/envs/parkour/`. Environments are named and
versioned in a registry (`automataleague/envs/registry.py`) and imported by id.

| Environment | Track | Difficulty levels | Obstacles | Domain randomization |
|---|---|---|---|---|
| **`parkour-1`** | winding circuit (closed loop) | 5 &nbsp;(0 flat to 4 hardest) | 5 &nbsp;(paving, hurdle, staircase, ramp, side incline) | Yes &nbsp;(per episode obstacle scaling) |

```python
from automataleague import make_env, list_environments

list_environments()                                       # [EnvSpec(env_id="parkour-1", ...)]

# single CPU env (for rendering or evaluation)
env = make_env("parkour-1", robot="spot", level=4, backend="cpu")

# batched GPU env (for training)
env = make_env("parkour-1", robot="spot", level=4, backend="warp", num_envs=2048)
```

* `level` picks the difficulty (0 to 4); it sets the obstacle heights and the matching
  action range.
* `backend="cpu"` is a single env; `backend="warp"` is the batched GPU env used for training.
* Override any config field inline, e.g.
  `make_env("parkour-1", robot="spot", height_scan=True, race_mode=True)`.

## Training

Two entry points in `examples/`, run from the repo root. Hydra config is
`examples/config_ppo.yaml`; any value can be overridden on the command line.

```bash
# train one difficulty level (standard TorchRL PPO scheme)
uv run python examples/ppo_single.py env.course.level_difficulty=2

# train across levels in sequence, warm starting each stage from the previous
uv run python examples/ppo_curriculum.py
```

* Override anything via Hydra, e.g. `collector.total_frames=20_000_000 env.num_envs=4096`.
* Curriculum settings (which levels, frames per level, action scale per level, warm start)
  live under the `curriculum:` block in the config.
* Checkpoints are written to `checkpoints/`.

## Simulation speed

Throughput of the batched MuJoCo Warp env for `parkour-1` (level 2, height scan on),
measured as total environment steps per second across all parallel envs. Tracked per
machine so we can compare hardware and watch it across versions. Throughput climbs with
`num_envs` as parallelism amortizes the fixed cost of each step.

| Machine | GPU | 1024 envs | 2048 envs | 4096 envs |
|---|---|---|---|---|
| gym1 | NVIDIA Jetson Orin | ≈ 10.5k | ≈ 12.0k | ≈ 13.0k |
| spark | NVIDIA GB10 | pending | pending | pending |
| laptop | NVIDIA RTX 4060 Laptop | pending | pending | pending |

For reference, the single environment CPU backend (used for rendering and evaluation)
runs at about 1.2k env steps per second on the laptop.

Measured by stepping the env after a warmup and computing `num_envs × steps / seconds`;
the same figure is logged as `train/fps` during a training run.

## Adding a custom robot

A robot is a `RobotSpec` (`automataleague/robots/base.py`), the whole contract a task
needs. Observation and action sizes are **derived from the joint count**, so any robot
plugs into the task unchanged.

1. Add `automataleague/robots/<name>.py` with a factory:

```python
from automataleague.robots.base import RobotSpec

def make_myrobot() -> RobotSpec:
    return RobotSpec(
        name="myrobot",
        mjcf_path="/abs/path/to/robot.xml",   # MJCF plus its mesh assets
        base_body="base",                      # floating base body name (unprefixed)
        nominal_height=0.5,                    # standing height (m)
        joint_names=[...],                     # actuated joints, canonical order
        actuator_names=[...],                  # same order as joint_names
        home_joint_qpos=[...],                 # standing stance (radians), per joint
        action_scale=0.3,                      # q_target = home + action_scale * action
    )
```

2. Register it in `automataleague/robots/__init__.py`:

```python
ROBOTS = {"spot": make_spot, "myrobot": make_myrobot}
```

3. Use it: `make_env("parkour-1", robot="myrobot")`, or train with
   `uv run python examples/ppo_single.py env.robot=myrobot`.

See `automataleague/robots/spot.py` for a complete example.

## Adding a custom reward

Reward weights are a `RewardConfig` (`automataleague/envs/parkour/config.py`); the terms
are combined in `compute_reward` (`automataleague/envs/parkour/rewards.py`).

**Retune existing terms** (no code): pass a `RewardConfig`, or override in training.

```python
from automataleague.envs.parkour.config import RewardConfig
env = make_env("parkour-1", robot="spot",
               reward_cfg=RewardConfig(forward=2.0, checkpoint=5.0, alive=-0.01))
```
```bash
uv run python examples/ppo_single.py env.reward_weights.forward=2.0 env.reward_weights.alive=-0.01
```

**Bring your own reward function**: pass `reward_fn` to `make_env` to replace the reward
entirely. It takes the same arguments as `compute_reward` (`rewards.py`) and returns
`(reward, components)`; if omitted, the default is used. Different robots often want
different rewards, so this lets the reward travel with the robot while the env stays fixed.

```python
import torch
from automataleague import make_env

def my_reward(state, prev_dist, cur_dist, reached_intermediate, reached_finish,
              fell, off_path, action, nominal_height, rc, forward_vel=None):
    progress = prev_dist - cur_dist                       # distance closed toward the goal
    reward = rc.progress * progress + rc.success * reached_finish.float()
    return reward, {}                                     # (reward tensor, components dict)

env = make_env("parkour-1", robot="go1", reward_fn=my_reward)
```

**Add a new term to the default** (two edits):

1. Add a weight field to `RewardConfig` in `config.py` (e.g. `energy: float = 0.0`).
2. Compute the term in `compute_reward` (`rewards.py`) and add it to the returned sum,
   e.g. `energy_pen = -rc.energy * (state.joint_vel ** 2).sum(-1)`.

Then expose it under `reward_weights` in `examples/config_ppo.yaml` to control it from training.

## Credits

Spot model © Boston Dynamics, from MuJoCo Menagerie (Apache 2.0). See `assets/spot/LICENSE`.
