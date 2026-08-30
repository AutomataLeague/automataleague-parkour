"""Rank a run's saved eval checkpoints by robustness, not by recency or eval reward.

`*_best.pt` is whatever scored highest on a single greedy eval rollout, which rewards
speed; on obstacle courses that is routinely NOT the checkpoint that finishes most
often. This replays every `*_eval_*.pt` in a run over several noisy starts and ranks by
finish count first, median lap second.

    MUJOCO_GL=egl uv run python tools/rank_series.py checkpoints/race_L2 --seeds 6
"""
import argparse
import glob
import os
import re

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from automataleague_parkour.envs.parkour.config import (  # noqa: E402
    DEFAULT_RESET_JOINT_NOISE,
    DEFAULT_RESET_POS_NOISE,
)
from automataleague_parkour.envs.parkour.parkour_cpu import ParkourEnvCPU  # noqa: E402
from automataleague_parkour.training.env import configs_from_cfg  # noqa: E402
from tools.policy_io import load_policy  # noqa: E402


def score(path, seeds, max_steps):
    actor, cfg = load_policy(path, torch.device("cpu"))
    course, rc, tc = configs_from_cfg(cfg)
    tc.max_episode_steps = max_steps
    env = ParkourEnvCPU(robot=cfg.env.robot, cfg=course, reward_cfg=rc,
                        term_cfg=tc, frame_skip=10)
    laps, wins = [], 0
    for s in range(seeds):
        obs = env.reset(seed=1000 + s, pos_noise=DEFAULT_RESET_POS_NOISE,
                        joint_noise=DEFAULT_RESET_JOINT_NOISE)
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            for t in range(max_steps):
                td = actor(TensorDict({"observation": torch.as_tensor(
                    obs, dtype=torch.float32).unsqueeze(0)}, [1]))
                obs, _, term, trunc, info = env.step(td["action"].squeeze(0).numpy())
                if term or trunc:
                    if int(info.get("outcome", 0)) == 1:
                        wins += 1
                        laps.append((t + 1) / 50.0)
                    break
    return wins, (float(np.median(laps)) if laps else None), env.obs_dim


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--every", type=int, default=1, help="evaluate every Nth checkpoint")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "*_eval_*.pt")),
                   key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
    files = files[::args.every]
    for extra in ("ppo_best.pt", "sac_best.pt", "td3_best.pt",
                  "ppo_final.pt", "sac_final.pt", "td3_final.pt"):
        p = os.path.join(args.run_dir, extra)
        if os.path.exists(p):
            files.append(p)
    if not files:
        raise SystemExit(f"no checkpoints in {args.run_dir}")

    print(f"ranking {len(files)} checkpoints in {args.run_dir} "
          f"({args.seeds} noisy starts each)")
    rows = []
    for p in files:
        wins, med, dim = score(p, args.seeds, args.max_steps)
        rows.append((wins, med if med is not None else 1e9, p, dim))
        print(f"  {os.path.basename(p):28} obs={dim:3}  {wins}/{args.seeds}"
              f"  {f'{med:.1f}s' if med is not None else '—'}")
    rows.sort(key=lambda r: (-r[0], r[1]))
    w, m, p, _ = rows[0]
    print(f"\nmost robust: {p}  ({w}/{args.seeds}, "
          f"{f'{m:.1f}s' if m < 1e9 else 'no finishes'})")


if __name__ == "__main__":
    main()
