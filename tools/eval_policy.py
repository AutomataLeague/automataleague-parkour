"""Score a trained PPO policy: finish rate and lap time over several noisy starts.

    uv run python tools/eval_policy.py checkpoints/race_L0/ppo_best.pt
    uv run python tools/eval_policy.py CKPT --seeds 8

Each start perturbs the base and joint pose with the SAME Gaussian noise the batched
trainer resets with, so a single lucky rollout is not mistaken for skill and the scores
are not measured out of distribution. A lap counts as finished when the env's terminal
outcome is a success (== 1); lap time is the control steps to the finish over 50 Hz.
This is the honest ranking signal for a checkpoint, and it disagrees with a single
greedy eval often enough to matter (see training-recipe.md, stage 4).
"""
import argparse

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type

from automataleague_parkour.envs.parkour.config import (
    DEFAULT_RESET_JOINT_NOISE,
    DEFAULT_RESET_POS_NOISE,
)
from automataleague_parkour.envs.parkour.parkour_cpu import ParkourEnvCPU
from automataleague_parkour.training import configs_from_cfg, load_policy


def rollout(env, actor, max_steps, *, seed, pos_noise, joint_noise):
    """One deterministic rollout from a noisy start; (outcome, frames_to_termination)."""
    obs = env.reset(seed=seed, pos_noise=pos_noise, joint_noise=joint_noise)
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for t in range(max_steps):
            td = actor(TensorDict(
                {"observation": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)}, [1]))
            obs, _, term, trunc, info = env.step(td["action"].squeeze(0).numpy())
            if term or trunc:
                return int(info.get("outcome", 0)), t + 1
    return 0, max_steps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="path to a ppo_*.pt checkpoint")
    ap.add_argument("--seeds", type=int, default=6, help="noisy starts to average over")
    # Separate units: metres on the base, radians on the joints. Both default to the
    # values the trainer resets with, so eval is in distribution by default.
    ap.add_argument("--pos-noise", type=float, default=DEFAULT_RESET_POS_NOISE,
                    help=f"base xy start noise sigma, metres (default: {DEFAULT_RESET_POS_NOISE})")
    ap.add_argument("--joint-noise", type=float, default=DEFAULT_RESET_JOINT_NOISE,
                    help="joint start noise sigma, radians "
                         f"(default: {DEFAULT_RESET_JOINT_NOISE})")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    actor, cfg = load_policy(args.checkpoint, args.device)
    course, rc, tc = configs_from_cfg(cfg)
    tc.max_episode_steps = args.max_steps
    env = ParkourEnvCPU(robot=cfg.env.robot, cfg=course, reward_cfg=rc,
                        term_cfg=tc, frame_skip=10)

    laps, wins = [], 0
    for s in range(args.seeds):
        outcome, frames = rollout(env, actor, args.max_steps, seed=1000 + s,
                                  pos_noise=args.pos_noise, joint_noise=args.joint_noise)
        if outcome == 1:
            wins += 1
            laps.append(frames / 50.0)

    print(args.checkpoint)
    print(f"  finished {wins}/{args.seeds} starts")
    if laps:
        laps.sort()
        print(f"  lap time  median {np.median(laps):.1f}s  best {laps[0]:.1f}s"
              f"  [{', '.join(f'{x:.1f}' for x in laps)}]")
    else:
        print("  no finishes")


if __name__ == "__main__":
    main()
