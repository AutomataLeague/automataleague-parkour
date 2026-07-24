"""Render every DR obstacle under 4 domain-randomization factors (one grid each).

Sets each obstacle's mocap pose for factors spanning the DR range and renders it, so
we can eye-check that height obstacles scale and tilted obstacles rotate as expected.

    MUJOCO_GL=egl uv run python tools/render_obstacle_dr.py --level 2
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from automataleague.envs.parkour.config import ParkourConfig  # noqa: E402
from automataleague.envs.parkour.obstacles import (  # noqa: E402
    _axis_quat, _cz_ground, _qmul, _qrot, _yaw_quat)
from automataleague.envs.parkour.scene import build_parkour_model  # noqa: E402


def _label(img, text):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 360, 34], fill=(20, 20, 24))
        d.text((10, 8), text, fill=(240, 220, 120))
        return np.asarray(im)
    except Exception:
        return img


def _set_pose(data, dr, gi, H, f):
    """Apply DR factor f to obstacle-group gi's mocap bodies (height + angle)."""
    label = f"factor {f:.2f}"
    h = dr["height"]
    if h is not None:
        sel = h["group"] == gi
        for idx, hn in zip(h["mocap_idx"][sel], h["h_nom"][sel]):
            data.mocap_pos[idx, 2] = hn * f - H
        if sel.any():
            label += f"   top {float(h['h_nom'][sel].max()) * f * 100:.0f} cm"
    a = dr["angle"]
    if a is not None:
        sel = a["group"] == gi
        for i in np.where(sel)[0]:
            ang = float(a["base_angle"][i]) * f
            q = _qmul(_yaw_quat(float(a["yaw"][i])), _axis_quat(list(a["axis"][i]), ang))
            half = list(a["half"][i])
            cz = _cz_ground(half, q) if a["cz_mode"][i] == 0 else \
                float(a["top_z"][i]) - _qrot(q, (0.0, 0.0, half[2]))[2]
            idx = int(a["mocap_idx"][i])
            data.mocap_quat[idx] = q
            data.mocap_pos[idx, 2] = cz
        if sel.any():
            label += f"   tilt {abs(float(a['base_angle'][sel][0])) * f:.0f} deg"
    return label


def _obstacle_camera(model, data, name):
    """Frame the geoms belonging to obstacle `name` from a 3/4 oblique view."""
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
    cam.elevation, cam.distance = -17, span * 0.9 + 3.0
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--factors", type=float, nargs="+", default=[0.5, 0.83, 1.17, 1.5])
    ap.add_argument("--out-dir", default="renders")
    args = ap.parse_args()

    cfg = ParkourConfig(track="circuit", level_difficulty=args.level)
    model, info = build_parkour_model("spot", cfg)
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    dr, H = info.obstacle_dr, info.obstacle_dr["H"]
    renderer = mujoco.Renderer(model, height=480, width=720)
    os.makedirs(args.out_dir, exist_ok=True)

    for gi, name in enumerate(dr["names"]):
        mujoco.mj_forward(model, data)
        cam = _obstacle_camera(model, data, name)
        frames = []
        for f in args.factors:
            lab = _set_pose(data, dr, gi, H, f)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(_label(renderer.render(), lab))
        grid = np.concatenate([np.concatenate(frames[0:2], 1),
                               np.concatenate(frames[2:4], 1)], 0)
        out = os.path.join(args.out_dir, f"dr_{name}.png")
        imageio.imwrite(out, grid)
        print(f"wrote {out}")
    renderer.close()


if __name__ == "__main__":
    main()
