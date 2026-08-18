# Training a parkour policy

How to train a parkour-1 policy with PPO, what to check at each stage, and the
things that went wrong here. The short version lives in the [README](README.md);
this is the long one, and everything below runs through the shipped `examples/`
entry points.

Frame counts are the reproducible quantity; wall clock depends entirely on your
GPU. `env.num_envs` is the number of parallel worlds; lower it if you run out of
memory, it changes throughput, not the recipe. All commands assume
`MUJOCO_GL=egl`, because evaluation renders headlessly.

There are two games on the same circuit, and they train the same way:

* **Complete** the course (reach the finish), flat or over obstacles. Reward
  shapes progress plus a finish bonus. `examples/ppo_single.py`, `ppo_curriculum.py`.
* **Race** the course (fastest lap). Reward is a time trial. `examples/ppo_race.py`.

---

## What we learned

Five findings from this project. Each cost GPU time, and the rest of this
document is the evidence.

**1. The racing line came from perception, not from the reward.** Early racers
took the safe, wide line through every bend, well off the shortest path. The
tempting fix is a reward term that penalises distance from the apex, and it is
the wrong one: it hard-codes a line the robot then tracks whether or not it is
fast. What actually worked was giving the policy more to *see*. The default
`track_perception=boundary` feeds the left and right corridor edges at each
lookahead, so the policy perceives the drivable channel rather than just the
centreline, and it finds the tighter line on its own because that line is faster.
On the same setup, boundary perception laps about a second quicker than
centreline perception and visibly cuts inside it at the corners. **Give the
policy the information, not the answer.**

**2. Widen the action range and let the gait emerge; do not prescribe it.**
`action_scale` is a hard geometric cap on how far each joint can move from its
home pose (`q_target = home + action_scale * action`), not a gain or a speed
limit. A racer held to the stock flat-course scale ran a cramped, stiff gait; at
a wider scale it discovered a longer, faster stride we never described. The rule
that generalises: pick `action_scale` for the range of motion the task needs,
raise it *between* runs with a warm start, and never impose a specific gait
through the reward when a wider action space will let a better one appear.

**3. The forward reward saturates at `target_speed`; past it, only the clock
drives speed.** The `forward` term rewards along-track speed *up to*
`target_speed` and is flat above it (`config.py` says so). Once a racer is at the
cap, the only thing left pulling it faster is the time cost: every step it has
not finished is charged `alive`, and the finish bonus arrives sooner the faster
it goes. If a racer plateaus *below* the cap it is stuck in a conservative gait,
and the fix that worked here was counter-intuitive: **lower** the PPO entropy
coefficient, not raise it. At the default `0.01` the policy stayed too jittery to
commit and ran a cautious gait; dropping it to `0.003` let the racer commit to a
faster stride and took a flat lap from **23.0 s to 18.7 s** (6/6 seeds), while
raising it to `0.03` made the policy diverge. A wider action range (finding 2) is
the other lever. If a racer plateaus *at* the cap instead, raise `target_speed` so
the dense signal keeps paying past it.

**4. The checkpoint the trainer calls "best" is the one that scored highest on a
single greedy eval rollout, which is not the same as the most robust one.**
`run_ppo` writes `ppo_best.pt` whenever an eval episode beats the previous best
eval reward. That eval is one deterministic rollout from a clean start. It
routinely disagreed with real performance measured over several noisy starts:
the true best full-lap policy was often an earlier `ppo_eval_*.pt` from the same
run. Rank the saved eval checkpoints yourself before you trust one (stage 4).

**5. A deterministic evaluation lied for days, and the finish counter under-counted
finishes.** A clean, fixed-start rollout is out of distribution relative to the
noisy-start training, and it produced eval numbers that disagreed with real
performance until we switched to noisy starts. Separately, when scoring "did it
finish", counting passes through gate *centres* under-counts in `race_mode`,
because a fast racer crosses each gate off-centre on the racing line; it twice
told us a healthy policy had failed every seed. Judge a rollout by the env's own
terminal `outcome`, and start it with the same noise the policy trained on.

**The common thread:** every one of these was a silent instrument. A reward that
looked like progress while the policy tracked the wrong line, a "best" checkpoint
that was not, a finish counter reading zero on a policy that finished every lap.
When something is wrong and nothing is complaining, suspect the measurement
before the model.

---

## The recipe

Four stages, each with a gate before the next block of GPU time.

### 0. Preflight (no training)

Two checks before you spend anything.

**Does the reward pay for the goal you are asking for?** For the time trial the
finish bonus has to outweigh the accumulated time cost, or finishing is net
negative and the policy learns to stall. With the shipped race preset that is
`success = 300` against `alive = -0.1` over at most `max_episode_steps = 2000`, so
a finished lap clears the `0.1 * 2000 = 200` worst-case time cost with margin. If
you lengthen the episode or raise `alive`, re-check that `success` still wins.

**What does `action_scale` actually buy the robot?** It is a geometric cap, so
choose it for the range of motion the level needs, not by feel. The obstacle
curriculum raises it with the level (below); a flat racer wants a wider scale
than the level-0 default so a faster gait is reachable (the shipped race preset
uses `0.5`).

### 1. Learn to run the flat circuit (level 0)

```bash
uv run python examples/ppo_single.py env.course.level_difficulty=0
```

Level 0 is flat: no obstacles, just the winding closed loop. It is the foundation
every other run warm-starts from, so get a clean flat runner before anything
else. Checkpoints land in `checkpoints/parkour1_L0/`.

> **Gate.** Watch `eval/max_checkpoint` climb and `eval/reached_finish` reach 1.
> A flat lap should be well within reach; if it is not, the problem is upstream
> of the obstacles.

### 2. Climb the obstacle curriculum (levels 1 to 4)

```bash
uv run python examples/ppo_curriculum.py
```

Levels 1 to 4 add obstacles of rising height (paving, hurdle, staircase, ramp,
side incline) and widen the action range to match, so the feet can clear them.
The curriculum trains each level in turn and **warm-starts each stage from the
previous stage's best checkpoint**, because a policy that already runs the flat
loop learns to hop a hurdle far faster than one starting cold. Turn the forward
**height scan** on for these levels so the policy sees the obstacle ahead instead
of feeling it on contact; it is on by default in the curriculum. Which levels,
the per-level frame budget and action scale, and the warm-start toggle live under
the `curriculum:` block in `examples/config_ppo.yaml`.

> **Gate.** Check `eval/reached_finish` per level before letting the curriculum
> advance. A level that never finishes will warm-start the next one from a broken
> policy.

### 3. Race the flat circuit (time trial)

```bash
uv run python examples/ppo_race.py                        # flat circuit, fastest lap
uv run python examples/ppo_race.py env.course.level_difficulty=2 env.course.height_scan=true
```

`ppo_race.py` swaps in the racing preset (`examples/config_race.yaml`):
`race_mode` navigation, `track_perception=boundary`, a wider action scale, and the
lap-time reward. Warm-start it from a flat runner (stage 1) for a large head start:

```bash
uv run python examples/ppo_race.py network.init_checkpoint=checkpoints/parkour1_L0/ppo_best.pt
```

The two levers that move lap time are perception and action range, not reward
shaping (findings 1 and 2). To compare lines, train one racer on
`track_perception=boundary` and one on `centerline` and look at where they differ:
the boundary racer should carry more speed through the bends.

> **Gate.** A racer can plateau in a slow, safe gait below `target_speed`. If it
> does, it is a commitment problem, not a reward problem: **lower** the entropy
> coefficient (`0.01 -> 0.003` took our flat racer from 23.0 s to 18.7 s; raising
> it to 0.03 diverged) and widen the action range. Measure the actual speed before
> concluding the task has no headroom.

### 4. Find out which checkpoint is actually best

Do **not** assume `ppo_best.pt` or the last checkpoint is the best one (finding
4). Rank the saved eval checkpoints (`checkpoints/<run>/ppo_eval_*.pt`) yourself
over several noisy starts, judged by whether each one finishes, and keep the one
that finishes most often and, among those, laps fastest. Warm-start any
continuation from that checkpoint, not from `ppo_best.pt`.

---

## Measuring progress

**The training curve is not the measurement.** `eval/reward` is a single greedy
rollout and tracks lap time only loosely; it is a health signal, not a ranking.
Rank policies by replaying them, and hold two things fixed:

* **Start with the same noise the policy trained on.** A clean deterministic start
  is out of distribution and its numbers disagreed with real performance here for
  days. Perturb the initial joint pose a little, as training does, and average
  over several seeds. One deterministic rollout is not evidence.
* **Judge by the env's terminal `outcome`, not by a hand-rolled finish counter.**
  In `race_mode` a fast racer crosses gates off-centre, so counting gate-centre
  passes under-counts finishes and twice reported a healthy policy as failing
  every seed. `env.step(...)` returns `info["outcome"]`; a successful finish is
  `outcome == 1`. Lap time is then the number of steps to that finish over the
  control rate (50 Hz).

A minimal loop over the shipped CPU env (`make_env(..., backend="cpu")` returns a
Gym-style single env whose `step` gives `info["outcome"]`) is enough: run each
checkpoint from several perturbed starts, drop the ones that do not finish, and
compare median lap time among the rest.

---

## Results

### The racing line came from perception, not reward shaping

Racers trained without track perception, or with only the centreline ahead,
tracked the middle of the corridor and ran wide through every bend. Feeding the
corridor **edges** instead (`track_perception=boundary`) let the policy perceive
the drivable channel, and it moved to a tighter line on its own, because that line
is faster under the same time-trial reward. On matched runs the boundary racer
lapped about a second quicker than the centreline one and visibly cut inside it at
the corners. No apex-distance reward term was ever added; adding one would have
pinned a line the robot then tracks whether or not it is the fast one.

Perception is not a free universal upgrade, though. It nearly doubles the width of
the observation, and a richer observation is not equally usable by every learner:
the same boundary channel that sped one setup up left another slower than its
centreline version. Measure the change, do not assume it.

### `action_scale`, chosen for range of motion

`q_target = home + action_scale * action`, with the action in [-1, 1], so
`action_scale` is the maximum radians a joint can be driven from its home pose. It
is a hard cap on the poses the policy can reach, not a gain and not a speed limit.
The obstacle curriculum raises it with the level so the feet can clear taller
obstacles:

| level | obstacles | action_scale |
| --- | --- | --- |
| 0 | none (flat) | 0.30 |
| 1 | paving | 0.30 |
| 2 | hurdle | 0.50 |
| 3 | staircase | 0.70 |
| 4 | ramp / side incline | 0.79 |

For racing the lever is different: a flat racer held to the level-0 scale ran a
cramped gait, and widening the scale let a longer, faster stride appear that we
never specified. Change `action_scale` **between** runs with a warm start, never
on a schedule inside one, because it changes what the same action numbers mean.

---

## Numerical stability

Parkour's PPO uses a `TanhNormal` policy with a state-independent scale. It ships
without the extra guards our sibling sumo project added after several runs died on
non-finite numbers, so if you push the entropy coefficient or learning rate hard,
watch the policy's standard deviation: a collapsing sigma alongside a *rising*
reward is the signature of a run quietly wrecking itself (shorter episodes accrue
less per-step cost, so reward can climb while the policy degenerates).

One rule is worth stating in advance: **never bound a runaway quantity with
`torch.clamp` if it sits in a gradient path.** `clamp` has zero gradient outside
its range, so it deletes the corrective force on exactly the samples that ran
away, and it is self-reinforcing. Use a smooth squash (`limit * tanh(x / limit)`)
or skip the update, and bound the source, not the symptom.

---

## Other things that cost real time

* **The "best" eval checkpoint is a greedy single rollout.** `run_ppo` saves
  `ppo_best.pt` on eval reward, which optimises a clean 20-second rollout, not
  full-lap robustness. The genuinely best checkpoint was repeatedly an earlier
  `ppo_eval_*.pt`. Rank the series (stage 4) before trusting one.
* **Deterministic evaluation disagreed with reality for days.** A fixed, noiseless
  start is out of distribution relative to noisy-start training. Evaluate with the
  training noise on.
* **Counting gate-centre passes under-counts finishes in `race_mode`.** A racer
  crosses gates off-centre on the fast line, so a centre-based finish check read
  zero on policies that finished every lap. Use the terminal `outcome`.
* **Turn eval video off for long runs.** With `logger.video=true`, headless EGL
  can hang for a long time at process teardown; set `logger.video=false` for
  unattended training and render separately afterwards.
