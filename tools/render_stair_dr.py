"""Render the staircase obstacle under 4 domain-randomization factors.

Validates the mocap-DR machinery visually: the same staircase, its step-rise scaled
per-env by a factor (0.5 .. 1.5 around the level mean), set through mocap_pos.z.

    MUJOCO_GL=egl uv run python tools/render_stair_dr.py --level 2
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from automataleague.envs.parkour.config import ParkourConfig  # noqa: E402
from automataleague.envs.parkour.scene import build_parkour_model  # noqa: E402


def _label(img, text):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 320, 34], fill=(20, 20, 24))
        d.text((10, 8), text, fill=(240, 220, 120))
        return np.asarray(im)
    except Exception:
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--factors", type=float, nargs="+", default=[0.5, 0.83, 1.17, 1.5])
    ap.add_argument("--out", default="renders/staircase_dr.png")
    args = ap.parse_args()

    cfg = ParkourConfig(track="circuit", level_difficulty=args.level)
    model, info = build_parkour_model("spot", cfg)
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos

    dr = info.obstacle_dr
    sel = dr["group"] == 1                      # group 1 = staircase
    idx = dr["mocap_idx"][sel]
    h_nom = dr["h_nom"][sel]
    H = dr["H"]
    top_nom = float(h_nom.max())                # tallest step at factor 1

    # staircase centre + path yaw from the mocap body positions
    xy = data.mocap_pos[idx, :2]
    center = xy.mean(0)
    d0 = xy[-1] - xy[0]
    yaw = float(np.arctan2(d0[1], d0[0]))

    renderer = mujoco.Renderer(model, height=520, width=760)
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([center[0], center[1], 0.18])
    cam.azimuth = np.degrees(yaw) - 108         # 3/4 side view: see depth + step heights
    cam.elevation, cam.distance = -16, 6.2

    frames = []
    for f in args.factors:
        data.mocap_pos[idx, 2] = h_nom * f - H  # DR: scale exposed step heights
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = _label(renderer.render(),
                     f"factor {f:.2f}   top step {top_nom * f * 100:.0f} cm")
        frames.append(img)
        print(f"  factor {f:.2f} -> staircase top {top_nom * f:.3f} m "
              f"(rises {top_nom * f / 4 * 100:.1f} cm/step)")
    renderer.close()

    top = np.concatenate(frames[0:2], axis=1)
    bot = np.concatenate(frames[2:4], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.imwrite(args.out, grid)
    print(f"wrote {args.out}  (level {args.level}, mean = factor 1.0)")


if __name__ == "__main__":
    main()
