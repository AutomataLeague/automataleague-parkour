"""Measure environment throughput, in total environment steps per second.

    MUJOCO_GL=egl uv run python tools/benchmark_env.py                    # warp sweep
    MUJOCO_GL=egl uv run python tools/benchmark_env.py --num-envs 4096
    uv run python tools/benchmark_env.py --backend cpu                    # single-env baseline

Throughput is `num_envs * steps / seconds` after a warmup, the same quantity
`run_ppo` logs as `train/fps`. It climbs with `num_envs` because each step has a
fixed cost that parallelism amortizes, so a single number means nothing without the
batch size next to it.

It also means nothing without the SENSOR CONFIG next to it: the observation is 49
columns blind and 78 with both the height scan and boundary preview on, and the
height scan raycasts. This script prints the config it measured for that reason.
"""
import argparse
import time

from automataleague_parkour import make_env


def measure(backend, num_envs, level, height_scan, track_perception, steps, warmup):
    """(steps_per_second, obs_dim) for one configuration."""
    import numpy as np
    import torch

    kwargs = dict(robot="spot", level=level, backend=backend,
                  height_scan=height_scan, track_perception=track_perception)
    env = make_env("parkour-1", **kwargs) if backend == "cpu" else \
        make_env("parkour-1", num_envs=num_envs, **kwargs)

    if backend == "cpu":
        env.reset()
        act = np.zeros(env.robot.n_joints)
        obs_dim = env.obs_dim

        def step():
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                env.reset()
    else:
        # Mirror the training loop exactly (`ppo.run_ppo`): step_and_maybe_reset is
        # what the collector uses, so the number here is the one training sees.
        td = env.reset()
        obs_dim = env.observation_spec["observation"].shape[-1]
        act = torch.zeros(num_envs, env.robot.action_dim, device=env.device)

        def step():
            nonlocal td
            td.set("action", act)
            _, td = env.step_and_maybe_reset(td)

    for _ in range(warmup):
        step()
    if backend != "cpu":
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if backend != "cpu":
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = 1 if backend == "cpu" else num_envs
    return n * steps / elapsed, obs_dim


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="warp", choices=["warp", "cpu"])
    ap.add_argument("--num-envs", type=int, nargs="*",
                    default=[512, 1024, 2048, 4096],
                    help="batch sizes to sweep (warp only)")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--height-scan", action="store_true", default=True)
    ap.add_argument("--no-height-scan", dest="height_scan", action="store_false")
    ap.add_argument("--track-perception", default="none",
                    choices=["none", "centerline", "boundary"])
    ap.add_argument("--steps", type=int, default=200, help="timed steps per config")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    batches = [1] if args.backend == "cpu" else args.num_envs
    print(f"backend={args.backend}  level={args.level}  "
          f"height_scan={args.height_scan}  track_perception={args.track_perception}  "
          f"steps={args.steps} (after {args.warmup} warmup)")
    print(f"{'num_envs':>9} {'obs_dim':>8} {'env steps/s':>13}")
    for n in batches:
        sps, obs_dim = measure(args.backend, n, args.level, args.height_scan,
                               args.track_perception, args.steps, args.warmup)
        print(f"{n:>9} {obs_dim:>8} {sps:>13,.0f}")


if __name__ == "__main__":
    main()
