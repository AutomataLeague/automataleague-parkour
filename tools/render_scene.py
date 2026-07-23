"""Render static screenshots of a parkour scene for design review.

No physics, no training — just compose the scene, set the robot to its home
stance, and render a few camera angles to PNGs. This is the Phase-A feedback loop.

Usage:
    uv run python tools/render_scene.py --robot spot
    uv run python tools/render_scene.py --robot spot --length 12 --half-width 1.0
"""

from __future__ import annotations

import argparse
import os

# Pick a headless GL backend before importing mujoco. Override with MUJOCO_GL=... .
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from automataleague.envs.parkour import ParkourConfig, build_parkour_model  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUT_DIR = os.path.join(_ROOT, "renders")


def _camera(lookat, azimuth, elevation, distance) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat = np.asarray(lookat, dtype=np.float64)
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    return cam


def render_scene(robot: str, cfg: ParkourConfig, width: int, height: int) -> dict[str, np.ndarray]:
    model, info = build_parkour_model(robot, cfg)
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    mujoco.mj_forward(model, data)

    print(
        f"model: nq={model.nq} nv={model.nv} nu={model.nu} "
        f"ngeom={model.ngeom} | base@home={info.home_qpos[info.base_qposadr:info.base_qposadr + 3]}"
    )

    L = cfg.length
    centre = [L / 2, 0.0, 0.3]
    views = {
        "iso": _camera(centre, azimuth=45, elevation=-22, distance=L * 1.1),
        "top": _camera(centre, azimuth=90, elevation=-89, distance=L * 1.15),
        "side": _camera(centre, azimuth=90, elevation=-12, distance=L * 1.0),
        "spawn": _camera([cfg.spawn_x + 1.0, 0.0, 0.4], azimuth=35, elevation=-18, distance=3.5),
    }

    renderer = mujoco.Renderer(model, height=height, width=width)
    images = {}
    for name, cam in views.items():
        renderer.update_scene(data, camera=cam)
        images[name] = renderer.render()
    renderer.close()
    return images


def main() -> None:
    p = argparse.ArgumentParser(description="Render parkour scene screenshots.")
    p.add_argument("--robot", default="spot")
    p.add_argument("--length", type=float, default=None)
    p.add_argument("--half-width", type=float, default=None)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    cfg = ParkourConfig()
    if args.length is not None:
        cfg.length = args.length
    if args.half_width is not None:
        cfg.half_width = args.half_width

    os.makedirs(_OUT_DIR, exist_ok=True)
    images = render_scene(args.robot, cfg, args.width, args.height)

    paths = []
    for name, img in images.items():
        path = os.path.join(_OUT_DIR, f"parkour_{args.robot}_{name}.png")
        imageio.imwrite(path, img)
        paths.append(path)
        print(f"  wrote {path}")

    print("\nRendered views:")
    for pth in paths:
        print(f"  {pth}")


if __name__ == "__main__":
    main()
