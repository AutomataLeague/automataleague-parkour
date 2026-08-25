"""Unitree Go1 — second robot in the league, added to test robot swapping.

Model vendored from MuJoCo Menagerie (``unitree_go1/go1.xml``): a small quadruped
with 12 position-controlled leg joints (hip/abduction, thigh, calf per leg) and a
``home`` standing keyframe. Like Spot it uses ``<position>`` actuators, so it fits the
env's position-target control (``q_target = home + action_scale * action``) with no
engine change. See ``assets/unitree_go1/LICENSE`` for attribution.
"""

from __future__ import annotations

import os

import numpy as np

from automataleague_parkour.robots.base import RobotSpec

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GO1_XML = os.path.join(_ROOT, "assets", "unitree_go1", "go1.xml")

# Canonical joint order matches go1.xml actuator order:
# per leg -> hip (abduction), thigh, calf. Legs: front-right, front-left, rear-right, rear-left.
_LEGS = ("FR", "FL", "RR", "RL")
_JOINTS = [f"{leg}_{dof}_joint" for leg in _LEGS for dof in ("hip", "thigh", "calf")]
_ACTUATORS = [f"{leg}_{dof}" for leg in _LEGS for dof in ("hip", "thigh", "calf")]

# Standing stance from the "home" keyframe: (hip=0, thigh=0.9, calf=-1.8) per leg.
# Base stands at z=0.27 in that keyframe.
_HOME_LEG = (0.0, 0.9, -1.8)
_HOME_QPOS = np.array(_HOME_LEG * len(_LEGS), dtype=np.float32)


def make_go1() -> RobotSpec:
    return RobotSpec(
        name="go1",
        mjcf_path=_GO1_XML,
        base_body="trunk",
        nominal_height=0.27,
        joint_names=list(_JOINTS),
        actuator_names=list(_ACTUATORS),
        home_joint_qpos=_HOME_QPOS,
        action_scale=0.3,  # walking-scale default; raise via config for obstacle levels
        foot_geom_names=["FR", "FL", "RR", "RL"],
    )
