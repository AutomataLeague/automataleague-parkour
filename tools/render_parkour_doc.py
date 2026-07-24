"""Documentation renders for the parkour1 circuit.

Produces, in renders/parkour1/:
  * aerial_level0..4.png  — the whole circuit from above at each difficulty level
                            (level 0 = the clean track with no obstacles)
  * obstacle_<name>.png   — a nominal (factor 1.0) close-up of each obstacle

    MUJOCO_GL=egl uv run python tools/render_parkour_doc.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from automataleague.envs.parkour.config import ParkourConfig  # noqa: E402
from automataleague.envs.parkour.scene import build_parkour_model  # noqa: E402

OUT = "renders/parkour1"


def _aerial_camera(centerline):
    lo, hi = centerline.min(0), centerline.max(0)
    center = (lo + hi) / 2
    ext = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([center[0], center[1], 0.0])
    cam.azimuth, cam.elevation = 45, -62
    cam.distance = ext * 1.7 + 6
    return cam


def _obstacle_camera(model, data, name):
    ids = [i for i in range(model.ngeom) if model.geom(i).name.startswith(name)]
    xy = data.geom_xpos[ids, :2]
    lo, hi = xy.min(0), xy.max(0)
    center = (lo + hi) / 2
    span = float(np.linalg.norm(hi - lo))
    d = xy[np.argmax(np.linalg.norm(xy - center, axis=1))] - center
    yaw = float(np.arctan2(d[1], d[0]))
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([center[0], center[1], 0.15])
    cam.azimuth = np.degrees(yaw) - 108
    cam.elevation, cam.distance = -18, span * 0.9 + 3.0
    return cam


def main():
    os.makedirs(OUT, exist_ok=True)

    # whole-circuit aerials, one per difficulty level (0 = no obstacles)
    for lvl in range(5):
        model, info = build_parkour_model("spot", ParkourConfig(track="circuit",
                                                                level_difficulty=lvl))
        data = mujoco.MjData(model)
        data.qpos[:] = info.home_qpos
        mujoco.mj_forward(model, data)
        r = mujoco.Renderer(model, height=1080, width=1440)
        r.update_scene(data, camera=_aerial_camera(info.centerline))
        tag = "level0_no_obstacles" if lvl == 0 else f"level{lvl}"
        path = os.path.join(OUT, f"aerial_{tag}.png")
        imageio.imwrite(path, r.render()); r.close()
        print(f"wrote {path}")

    # nominal obstacle close-ups (at L2)
    model, info = build_parkour_model("spot", ParkourConfig(track="circuit", level_difficulty=2))
    data = mujoco.MjData(model); data.qpos[:] = info.home_qpos
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, height=560, width=840)
    label = {"obst_paving": "paving", "obst_hurdle": "hurdle", "obst_staircase": "staircase",
             "obst_ramp": "ramp", "obst_side_incline": "banked"}
    for name in info.obstacle_dr["names"]:
        r.update_scene(data, camera=_obstacle_camera(model, data, name))
        path = os.path.join(OUT, f"obstacle_{label[name]}.png")
        imageio.imwrite(path, r.render())
        print(f"wrote {path}")
    r.close()


if __name__ == "__main__":
    main()
