"""Render one frame from every camera preset (Spot posed mid-course) for review."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from automataleague.envs.parkour.render import CAMERAS, render_trajectory  # noqa: E402
from automataleague.envs.parkour.scene import build_parkour_model  # noqa: E402

_OUT = os.path.join(os.path.dirname(__file__), "..", "renders")

model, info = build_parkour_model("spot")
# Pose Spot standing at mid-course (x=4.5) so cameras have context around it.
q = info.home_qpos.copy()
q[info.base_qposadr] = 4.5
qpos_seq = q[None]  # single frame

os.makedirs(_OUT, exist_ok=True)
for name in CAMERAS:
    frame = render_trajectory(model, info, qpos_seq, camera=name, size=(720, 1280))[0]
    path = os.path.abspath(os.path.join(_OUT, f"camera_{name}.png"))
    imageio.imwrite(path, frame)
    print(f"wrote {path}")
