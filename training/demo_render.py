"""Render a trained parkour policy from multiple camera perspectives (on-demand).

Rolls the policy through one episode ONCE, then re-renders that same trajectory
from each requested camera — synchronized views, one mp4 each.

Usage:
    MUJOCO_GL=egl uv run python training/demo_render.py \
        --checkpoint training/checkpoints/ppo_final.pt \
        --cameras over_shoulder,drone,side --out-dir videos/
    # default --cameras = all presets
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torchrl.data import Bounded, Composite, Unbounded  # noqa: E402

from automataleague.envs.parkour import height_scan as hs  # noqa: E402
from automataleague.envs.parkour.parkour_cpu import ParkourEnvCPU  # noqa: E402
from automataleague.envs.parkour.render import (  # noqa: E402
    CAMERAS,
    record_trajectory,
    render_trajectory,
)
from automataleague.robots import get_robot  # noqa: E402
from utils_ppo import _configs_from_cfg, make_ppo_models  # noqa: E402


def _build_actor(cfg, robot, device):
    """Rebuild the actor from config/dims without a live GPU env (uses stub specs)."""
    class _Stub:
        pass

    # obs grows by SCAN_N when the checkpoint was trained with the height scan.
    scan_on = bool(getattr(getattr(cfg.env, "course", object()), "height_scan", False))
    obs_dim = robot.obs_dim + (hs.SCAN_N if scan_on else 0)

    stub = _Stub()
    stub.batch_size = torch.Size([1])
    stub.observation_spec = Composite(
        observation=Unbounded(shape=(1, obs_dim), device=device), shape=(1,)
    )
    stub.action_spec = Composite(
        action=Bounded(
            low=-torch.ones(1, robot.action_dim, device=device),
            high=torch.ones(1, robot.action_dim, device=device),
            device=device,
        ),
        shape=(1,),
    )
    # make_ppo_models reads env.action_spec as the leaf Bounded.
    stub.action_spec = stub.action_spec["action"]
    actor, _ = make_ppo_models(cfg, stub, device)
    return actor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cameras", default=",".join(CAMERAS),
                   help="comma-separated camera presets; default all")
    p.add_argument("--out-dir", default="videos")
    p.add_argument("--steps", type=int, default=600, help="max steps to roll out")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--loop", action="store_true",
                   help="keep resetting to fill --steps (else stop at first episode end)")
    args = p.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    unknown = [c for c in cameras if c not in CAMERAS]
    if unknown:
        raise SystemExit(f"Unknown camera(s) {unknown}. Available: {sorted(CAMERAS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    robot = get_robot(cfg.env.robot)

    actor = _build_actor(cfg, robot, device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    course, rc, tc = _configs_from_cfg(cfg)
    env = ParkourEnvCPU(robot=cfg.env.robot, cfg=course, reward_cfg=rc, term_cfg=tc,
                        frame_skip=getattr(cfg.env, "frame_skip", 10))

    # Roll the policy ONCE; record the trajectory.
    qpos_seq = record_trajectory(env, actor, args.steps, policy_device=str(device),
                                 loop_episodes=args.loop)
    print(f"recorded {len(qpos_seq)} frames; rendering {len(cameras)} camera(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    for cam in cameras:
        frames = render_trajectory(env.model, env.info, qpos_seq, camera=cam,
                                   size=(args.height, args.width))
        out = os.path.join(args.out_dir, f"parkour_{cfg.env.robot}_{cam}.mp4")
        imageio.mimwrite(out, frames, fps=args.fps)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
