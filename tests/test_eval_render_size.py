import os
import sys

import mujoco
import pytest

# eval_1v1.py imports demo_render/utils_ppo flat (run from training/ via CLI); match
# that here, same convention as tests/test_ppo_utils.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from training.eval_1v1 import _ensure_offscreen


def _tiny_model():
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>'
    )


def test_ensure_offscreen_enlarges_for_portrait():
    model = _tiny_model()  # default offscreen 640x480
    _ensure_offscreen(model, (1920, 1080))  # (height, width)
    assert model.vis.global_.offwidth >= 1080
    assert model.vis.global_.offheight >= 1920


def test_ensure_offscreen_never_shrinks():
    model = _tiny_model()
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    _ensure_offscreen(model, (480, 640))  # smaller than current
    assert model.vis.global_.offwidth == 1280
    assert model.vis.global_.offheight == 720


def test_ensure_offscreen_allows_portrait_render():
    """The whole point: after enlarging, a portrait Renderer actually works."""
    model = _tiny_model()
    _ensure_offscreen(model, (1920, 1080))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=1920, width=1080)
    renderer.update_scene(data)
    frame = renderer.render()
    renderer.close() if hasattr(renderer, "close") else None
    assert frame.shape == (1920, 1080, 3)
