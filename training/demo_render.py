"""Render a demo video of a trained parkour policy from a checkpoint.

Usage:
    MUJOCO_GL=egl uv run python training/demo_render.py \
        --checkpoint training/checkpoints/ppo_final.pt --out videos/parkour.mp4
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torchrl.data import Bounded, Composite, Unbounded  # noqa: E402

from utils_ppo import make_ppo_models, rollout_video  # noqa: E402


class _StubSpecEnv:
    """Minimal spec provider so make_ppo_models can build the actor without a live env."""

    def __init__(self, obs_dim, act_dim, device):
        self.batch_size = torch.Size([1])
        self.observation_spec = Composite(
            observation=Unbounded(shape=(1, obs_dim), device=device), shape=(1,)
        )
        self.action_spec = Composite(
            action=Bounded(
                low=-torch.ones(1, act_dim, device=device),
                high=torch.ones(1, act_dim, device=device),
                device=device,
            ),
            shape=(1,),
        )

    # make_ppo_models reads env.action_spec as the leaf Bounded.
    @property
    def action_spec_leaf(self):
        return self.action_spec["action"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="videos/parkour_demo.mp4")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    from automataleague.robots import get_robot
    robot = get_robot(cfg.env.robot)

    stub = _StubSpecEnv(robot.obs_dim, robot.action_dim, device)
    # make_ppo_models expects env.action_spec to be the leaf Bounded spec.
    stub.action_spec = stub.action_spec["action"]
    actor, _ = make_ppo_models(cfg, stub, device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    frames = rollout_video(actor, cfg, max_steps=args.steps, policy_device=str(device))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=args.fps)
    print(f"Wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
