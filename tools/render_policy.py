"""Render a trained policy running the parkour course, to a video.

    MUJOCO_GL=egl uv run python tools/render_policy.py checkpoints/race_L0/ppo_best.pt
    MUJOCO_GL=egl uv run python tools/render_policy.py CKPT -o lap.mp4 --camera drone_side

The policy runs on the same course it was trained on (level, sensors and reward all
come from the checkpoint's own config). Cameras: chase, drone_side, drone, top, side,
front, over_shoulder.

This is the same renderer the training entry points call after each stage
(`logger.stage_video`), so a clip made here is comparable to the per-level ones.
"""
import argparse

from automataleague_parkour.envs.parkour.render import CAMERAS
from automataleague_parkour.training import render_checkpoint


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="path to a ppo_*.pt checkpoint")
    ap.add_argument("-o", "--out", default="policy.mp4", help="output video path")
    ap.add_argument("--camera", default="chase", choices=sorted(CAMERAS),
                    help="camera preset (default: chase)")
    ap.add_argument("--max-steps", type=int, default=2500, help="step cap for the lap")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    steps = render_checkpoint(args.checkpoint, args.out, camera=args.camera,
                              max_steps=args.max_steps, device=args.device, fps=args.fps)
    print(f"wrote {args.out}  ({steps} steps, {steps / 50:.1f}s, camera={args.camera})")


if __name__ == "__main__":
    main()
