<p align="center">
  <img src="assets/automataleague_logo.png" alt="Automata League" width="340">
</p>

# Automata League Parkour

The parkour competition environment of the **Automata League**: a legged robot learns to run
a winding circuit in [MuJoCo](https://mujoco.org/), either **clearing obstacles** to reach the
finish or **racing the flat course** for the fastest lap. Boston Dynamics **Spot** and Unitree
**Go1** ship as example robots; plug in your own the same way (see
[Adding a custom robot](#adding-a-custom-robot), which walks through how Go1 was added). The
examples train with [TorchRL](https://github.com/pytorch/rl) PPO, GPU parallel via
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp); any other TorchRL agent works the
same way (see [Training](#training)).

Sibling of [automataleague-sumo](https://github.com/AutomataLeague/automataleague-sumo), whose
architecture this mirrors.

## Setup

```bash
uv sync                              # core: environment building, rendering, task logic + tests
uv sync --extra train --extra gpu    # adds torchrl, hydra, wandb, MuJoCo Warp (GPU box)
```

Headless rendering needs a GL backend: `MUJOCO_GL=egl`.

## The parkour environment

<p align="center">
  <img src="assets/parkour_env.png" alt="Three frames of a parkour run: the opening straight, a bend taken on the racing line, and the paving obstacle field" width="820">
</p>

<p align="center">
  <sub>Left and centre: a trained racer on the flat circuit (level 0), starting the lap and
  carrying speed through a bend. Right: the paving field at level 2. Yellow lines are the
  checkpoint gates, white is the corridor boundary — stray past it and the episode ends.</sub>
</p>

The parkour task lives in `automataleague_parkour/envs/parkour/`. Environments are named and
versioned in a registry (`automataleague_parkour/envs/registry.py`) and imported by id.

| Environment | Track | Difficulty levels | Obstacles | Domain randomization |
|---|---|---|---|---|
| **`parkour-1`** | winding circuit (closed loop) | 5 &nbsp;(0 flat to 4 hardest) | paving, hurdle, staircase, ramp, side incline | Yes &nbsp;(per-episode obstacle scaling) |

```python
from automataleague_parkour import make_env, list_environments

list_environments()                                       # [EnvSpec(env_id="parkour-1", ...)]

# single CPU env (for rendering or evaluation)
env = make_env("parkour-1", robot="spot", level=0, backend="cpu")

# batched GPU env (for training)
env = make_env("parkour-1", robot="spot", level=2, backend="warp", num_envs=2048)
```

* `level` picks the difficulty (0 to 4); it sets the obstacle heights and the matching action range.
* `backend="cpu"` is a single env; `backend="warp"` is the batched GPU env used for training.
* Override any config field inline, e.g.
  `make_env("parkour-1", robot="spot", race_mode=True, track_perception="none")`.

### Difficulty: flat (no obstacles) to obstacle courses

`level_difficulty` is the single knob:

* **Level 0 is flat: no obstacles at all**, just the winding track. It is the place to start, and
  the setting for pure running and for racing.
* **Levels 1 to 4 add obstacles** of increasing height (paving, hurdle, staircase, ramp, side
  incline) and widen the action range to match, so the robot can lift its feet high enough to
  clear them. The completion trainer runs with the forward **height scan** on by default
  (`config_ppo.yaml`), so the policy sees the obstacles ahead instead of feeling them on contact.

### Two ways to run it: complete or race

* **Complete** (default): reach the finish. The reward shapes progress toward the goal plus a big
  finish bonus. Trained with `examples/ppo_single.py` / `examples/ppo_curriculum.py`.
* **Race** (`race_mode=true`): the fastest lap. The reward is a time trial (along-track speed,
  minus a per-step time cost, plus a finish bonus). Trained with `examples/ppo_race.py`. See
  [Racing](#racing).

### The reward

The completion reward (`RewardConfig` in `envs/parkour/config.py`, combined in
`rewards.py`). Unlike our sibling sumo project these weights are **not** on a common
whole-episode scale, so read the middle column before comparing them:

| term | what it pays for | kind | weight |
|---|---|---|---|
| `forward` | along-track speed toward the goal, **saturating at `target_speed`** | per step | 1.5 |
| `target_speed` | the speed `forward` stops paying above (m/s) | — | 1.0 |
| `progress` | distance closed toward the next gate (potential shaping) | per step | 2.0 |
| `checkpoint` | reaching a gate | one-off | 10.0 |
| `success` | reaching the finish | terminal | 100.0 |
| `alive` | per-step survival bonus; **negative makes it a time cost** | per step | 0.0 |
| `upright` / `height` | exp-shaped posture keeping, deliberately small so standing still never wins | per step | 0.05 each |
| `action` / `joint_vel` | regularizers (magnitude, smoothness) | per step | 0.01 / 0.001 |
| `fall` | falling | one-off penalty | 10.0 |
| `off_path` | straying past `half_width` of the centerline; also terminates | one-off penalty | 25.0 |
| `feet_air_time` | swing duration per foot, for a clean stepping gait instead of a shuffle | per step | 0.0 (off) |

`forward` is the dense driver that makes walking emerge, and the **saturation matters**:
above `target_speed` it is flat, so nothing in this reward pushes a policy faster. That
is what the racing preset changes — it zeroes `progress` and `checkpoint`, sets `alive`
negative to make every step cost time, and raises `success` to 300 so a finished lap
still clears the accumulated cost. Racing speed then comes from the clock, not from a
shaping term.

## Training

Entry points in `examples/`, run from the repo root. The Hydra config is `examples/config_ppo.yaml`;
any value can be overridden on the command line. Checkpoints are written to `checkpoints/<run>/`.
For the staged recipe, the gates to check at each stage, the results it delivers, and the
mistakes that cost us GPU time, see [training-recipe.md](training-recipe.md). On the shipped
defaults at 30M frames per level the chain scores 6/6 flat, 5/6 paving, 3/6 hurdle, 6/6
staircase, 4/6 ramp, and 6/6 racing over 6 noisy starts.

### Without obstacles vs with obstacles

```bash
# flat: learn to run the circuit with no obstacles (level 0)
uv run python examples/ppo_single.py env.course.level_difficulty=0

# with obstacles: pick a level 1..4
uv run python examples/ppo_single.py env.course.level_difficulty=2
```

Override anything via Hydra, e.g. `collector.total_frames=20_000_000 env.num_envs=4096`.

Every run sees the same thing: `[proprio 49 | height_scan 12 | track_preview 17] = 78`.
Both sensors are on at every level, so the policy always has the terrain ahead and the
corridor ahead, and any checkpoint warm-starts any run — see
[Observation layout](training-recipe.md#observation-layout-and-warm-starting).

### Curriculum (flat to obstacles, in sequence)

```bash
uv run python examples/ppo_curriculum.py
```

Trains each level in turn, warm-starting each stage from the previous stage's best checkpoint
(flat first, then progressively harder obstacles). Which levels, the per-level frame budget and
action scale, and the warm-start toggle live under the `curriculum:` block in the config.

**Every stage renders an eval video when it finishes**, to `videos/<run_name>.mp4`, so a
curriculum leaves one clip per level (`parkour1_curriculum_L1.mp4` … `_L4.mp4`) and you can
see a level that trained badly without waiting for the whole chain. Controlled by
`logger.stage_video` / `logger.stage_video_camera`; set `logger.stage_video=false` to skip.
Rendering runs after the checkpoint is written and can never fail a completed run.

## Racing

A time trial on the same circuit: the goal is the fastest lap, not just reaching the finish.
`examples/ppo_race.py` uses the racing preset (`examples/config_race.yaml`): `race_mode`
navigation (the agent is scored on along-track speed and gate crossings, so it finds the
time-optimal line instead of dipping to each gate centre), a lap-time reward, and **track
perception** so it can see the course ahead.

```bash
# race the flat circuit for the fastest lap (level 0, no obstacles)
uv run python examples/ppo_race.py

# race an obstacle course instead
uv run python examples/ppo_race.py env.course.level_difficulty=2
```

**Track perception** (`env.course.track_perception`) is what lets a policy plan a line. It
appends lookahead points along the track ahead to the observation:

* `boundary` (**the default**): the left and right corridor edges at each lookahead, so the
  policy sees the drivable channel and can cut the apex.
* `centerline`: the midline, so the policy tracks the centre of the track.
* `none`: blind to the track ahead. For ablations.

The lap-time reward, `entropy_coeff` and the rest of the preset are in `examples/config_race.yaml`;
override any of it on the command line.

## Watch and evaluate a policy

Small tools in `tools/` consume a trained checkpoint. Watch a policy run the course:

```bash
MUJOCO_GL=egl uv run python tools/render_policy.py checkpoints/race_L0/ppo_best.pt -o lap.mp4
MUJOCO_GL=egl uv run python tools/render_policy.py CKPT --camera drone_side   # chase|drone_side|top|...
```

Score it over several noisy starts (the honest ranking signal, since the training curve's greedy
eval is not the same as robust performance):

```bash
uv run python tools/eval_policy.py checkpoints/race_L0/ppo_best.pt --seeds 8
#   finished 8/8 starts
#   lap time  median 18.7s  best 18.6s  [...]
```

A lap counts as finished on the env's terminal outcome, and lap time is the control steps to the
finish over 50 Hz. Both run on the single-env CPU backend, so no GPU is needed to watch or rank.

## Simulation speed

Throughput of the batched MuJoCo Warp env, as total environment steps per second across
all parallel envs. It climbs with `num_envs` because parallelism amortizes the fixed cost
of each step, so a figure only means something with its batch size and its sensor config
beside it — the shipped observation is 78 columns, or 49 with both sensors turned off.

```bash
MUJOCO_GL=egl uv run python tools/benchmark_env.py                  # sweeps num_envs
uv run python tools/benchmark_env.py --backend cpu                  # single-env baseline
```

Observed on `parkour-1` at level 2 with the height scan on (61-dim observation), across
the machines this was developed on:

| num_envs | observed env steps/s |
|---:|---|
| 512 | 10k – 60k |
| 1024 | 10k – 85k |
| 2048 | 12k – 105k |
| 4096 | 13k – 110k |

The spread is the hardware, not the code: run `benchmark_env.py` to get the number for
yours. The single-env CPU backend used for rendering and evaluation runs at roughly
**1.2k steps/s**.

End-to-end PPO training is slower than the raw env figure because it includes the policy
forward pass and the update — on the same box that stepped the env at ~105k, training
sustained ~54k frames/s. That end-to-end number is what `run_ppo` logs as `train/fps`;
`benchmark_env.py` reports the env alone, so compare like with like.

## Adding a custom robot

A robot is a `RobotSpec` (`automataleague_parkour/robots/base.py`), the whole contract a task
needs. Observation and action sizes are **derived from the joint count**, so a robot with
a different number of legs or joints plugs into the same env, reward, and PPO code with no
changes to any of them.

Unitree **Go1** is the worked example. It is nothing like Spot: half the standing height
(0.27 m vs 0.46 m), a different joint naming scheme, and a stock MuJoCo Menagerie model.
Adding it took the three steps below and one asset fix. The full result is
`automataleague_parkour/robots/unitree_go1.py`; here is what each step actually involved.

**1. Vendor the model.** Drop the MJCF and its meshes under `assets/<name>/`, keeping the
upstream `LICENSE`. Go1 came from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`unitree_go1/`).

One asset fix was needed: Menagerie sets a default geom `margin="0.001"`, which the Warp
backend rejects (`non-zero margin with MULTICCD`). Removing that one attribute from
`go1.xml` was the only edit to the model. Worth knowing before you vendor any Menagerie robot.

**2. Write the factory** in `automataleague_parkour/robots/<name>.py`. Every field is read off the
model's own docs / the `home` keyframe:

```python
from automataleague_parkour.robots.base import RobotSpec

def make_go1() -> RobotSpec:
    return RobotSpec(
        name="go1",
        mjcf_path=_GO1_XML,                    # assets/unitree_go1/go1.xml
        base_body="trunk",                     # floating base body name (unprefixed)
        nominal_height=0.27,                   # standing height from the home keyframe (m)
        joint_names=_JOINTS,                   # 12 leg joints, canonical order
        actuator_names=_ACTUATORS,             # same order as joint_names
        home_joint_qpos=_HOME_QPOS,            # standing stance: (hip 0, thigh 0.9, calf -1.8) x4
        action_scale=0.3,                      # q_target = home + action_scale * action
        foot_geom_names=["FR", "FL", "RR", "RL"],  # feet, for the optional gait reward
    )
```

The two things that matter most:

* **Joint order is the contract.** `joint_names`, `actuator_names`, and `home_joint_qpos`
  must line up index for index, and match the actuator order in the MJCF. Get this wrong and
  the policy drives the wrong joint. Go1's order is front-right, front-left, rear-right,
  rear-left, each leg hip then thigh then calf.
* **`home_joint_qpos` is the stance the policy perturbs around.** Actions are offsets from
  it (`q_target = home + action_scale * action`), so a good standing pose is what makes
  locomotion learnable. Read it from the model's `home` keyframe.

`foot_geom_names` is optional: name the foot geoms and the foot air-time gait reward can
find them (see [Adding a custom reward](#adding-a-custom-reward)). Leave it empty to skip.

**3. Register it** in `automataleague_parkour/robots/__init__.py`:

```python
ROBOTS = {"spot": make_spot, "go1": make_go1}
```

That is all. Now use it anywhere Spot goes:

```python
env = make_env("parkour-1", robot="go1", backend="warp", num_envs=2048)
```
```bash
uv run python examples/ppo_single.py env.robot=go1
```

The PPO networks size themselves from the env's obs/action specs, so the same
`examples/ppo_single.py` trains Go1 with no code change. Different robots often want a
different reward though: Go1 walks cleanly once the foot air-time gait term is on. That is
what the `reward_fn` hook in the next section is for, and why the reward travels with the
robot while the env stays fixed.

Compare `automataleague_parkour/robots/spot.py` and `automataleague_parkour/robots/unitree_go1.py` side by
side to see exactly what changes from one robot to the next.

## Adding a custom reward

Reward weights are a `RewardConfig` (`automataleague_parkour/envs/parkour/config.py`); the terms
are combined in `compute_reward` (`automataleague_parkour/envs/parkour/rewards.py`).

**Retune existing terms** (no code): pass a `RewardConfig`, or override in training.

```python
from automataleague_parkour.envs.parkour.config import RewardConfig
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
from automataleague_parkour import make_env

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

## Tests

```bash
uv run pytest -m "not gpu"            # CPU suite: env, observations, rewards, navigation, preview
uv run pytest -m gpu                  # the batched MuJoCo Warp backend (needs CUDA + mujoco-warp)
```

The task logic (observations, rewards, termination, the CPU env) is written in tensors, so the
task-logic tests run on a bare `uv sync` with no training stack. The training-integration tests
additionally need `--extra train` (torchrl), and the `gpu`-marked tests need `--extra gpu`.

## Licence

Apache-2.0, see [LICENSE](LICENSE). Vendored robot models keep their upstream licences.

## Credits

* Spot model © Boston Dynamics, from MuJoCo Menagerie (Apache 2.0). See `assets/spot/LICENSE`.
* Go1 model © Unitree Robotics, from MuJoCo Menagerie (BSD 3-Clause). See `assets/unitree_go1/LICENSE`.
