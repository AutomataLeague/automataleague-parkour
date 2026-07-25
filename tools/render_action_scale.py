"""Explain Spot's action_scale -> foot-lift capability as a rendering.

For each action_scale, tuck the front-left leg to its MAX reachable foot height
(grid search within +/- scale of the home stance) and render it against horizontal
reference lines at the four staircase-top heights (L1..L4). Shows directly how much
range each action_scale unlocks and which obstacle levels the foot can clear.

    MUJOCO_GL=egl uv run python tools/render_action_scale.py
"""
from __future__ import annotations

import itertools
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from automataleague.envs.parkour.config import ParkourConfig  # noqa: E402
from automataleague.envs.parkour.scene import build_parkour_model  # noqa: E402
from automataleague.robots import get_robot  # noqa: E402

LEVELS = [("L1", 0.112), ("L2", 0.224), ("L3", 0.336), ("L4", 0.448)]
# obstacle-height reference lines, light -> amber (single accent family)
LINE_RGBA = [[0.75, 0.75, 0.78, 1], [0.95, 0.80, 0.45, 1],
             [0.98, 0.68, 0.22, 1], [0.95, 0.55, 0.12, 1]]
SCALES = [0.30, 0.50, 0.70, 0.79]


def _add_line(scn, frm, to, w, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w,
                         np.asarray(frm, float), np.asarray(to, float))
    scn.ngeom += 1


def _sphere(scn, pos, r, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0.]),
                        np.asarray(pos, float), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _label(img, lines):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 430, 20 + 18 * len(lines)], fill=(18, 18, 22))
        for i, (t, c) in enumerate(lines):
            d.text((10, 8 + 18 * i), t, fill=c)
        return np.asarray(im)
    except Exception:
        return img


def main():
    model, info = build_parkour_model("spot", ParkourConfig(track="circuit"))
    robot = get_robot("spot")
    data = mujoco.MjData(model)
    fl_qadr = [info.joint_qposadr[i] for i, jn in enumerate(robot.joint_names)
               if jn.startswith("fl_")]
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "spot/FL")
    ba = info.base_qposadr
    bx, by = float(info.home_qpos[ba]), float(info.home_qpos[ba + 1])

    data.qpos[:] = info.home_qpos
    mujoco.mj_forward(model, data)
    ground = float(data.geom_xpos[foot, 2])   # foot z in home stance (~on floor)

    renderer = mujoco.Renderer(model, height=620, width=760)
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([bx + 0.25, by, 0.28])
    cam.azimuth, cam.elevation, cam.distance = -72, -6, 2.1   # front-left profile

    frames = []
    for scale in SCALES:
        # max foot height reachable within +/- scale of home on the front-left leg
        best_q, best_z = None, -1e9
        for combo in itertools.product(np.linspace(-scale, scale, 5), repeat=3):
            q = info.home_qpos.copy()
            for k, adr in enumerate(fl_qadr):
                q[adr] += combo[k]
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            if data.geom_xpos[foot, 2] > best_z:
                best_z, best_q = float(data.geom_xpos[foot, 2]), q.copy()
        data.qpos[:] = best_q
        mujoco.mj_forward(model, data)
        lift = best_z - ground
        fx, fy = float(data.geom_xpos[foot, 0]), float(data.geom_xpos[foot, 1])

        renderer.update_scene(data, camera=cam)
        scn = renderer.scene
        for (lv, h), rgba in zip(LEVELS, LINE_RGBA):        # obstacle-height lines
            _add_line(scn, [fx, fy - 0.55, ground + h], [fx, fy + 0.55, ground + h],
                      0.006, rgba)
        _sphere(scn, [fx, fy, best_z], 0.035, [0.30, 0.85, 1.0, 1])   # the foot
        _add_line(scn, [fx, fy, ground], [fx, fy, best_z], 0.004, [0.30, 0.85, 1.0, 0.6])
        cleared = [lv for lv, h in LEVELS if lift >= h]
        frames.append(_label(renderer.render(), [
            (f"action_scale = {scale:.2f}  (+/-{np.degrees(scale):.0f} deg per joint)",
             (245, 245, 245)),
            (f"max foot lift = {lift * 100:.0f} cm", (120, 216, 255)),
            (f"clears: {', '.join(cleared) if cleared else 'none'}", (250, 205, 120)),
            ("lines = staircase tops  L1 11 / L2 22 / L3 34 / L4 45 cm", (170, 170, 175)),
        ]))

    grid = np.concatenate([np.concatenate(frames[0:2], 1),
                           np.concatenate(frames[2:4], 1)], 0)
    out = "renders/parkour1/action_scale_capability.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.imwrite(out, grid)
    print(f"wrote {out}")
    print("reference lines (foot-lift needed): " +
          " ".join(f"{lv}={h*100:.0f}cm" for lv, h in LEVELS))


if __name__ == "__main__":
    main()
