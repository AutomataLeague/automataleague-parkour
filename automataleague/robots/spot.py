"""Boston Dynamics Spot — first robot in the league.

Model vendored from MuJoCo Menagerie (``boston_dynamics_spot/spot.xml``): a
quadruped with 12 position-controlled leg joints (hip-x, hip-y, knee per leg)
and a ``home`` standing keyframe. See ``assets/spot/LICENSE`` for attribution.
"""

from __future__ import annotations

import os

import numpy as np

from automataleague.robots.base import RobotSpec

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SPOT_XML = os.path.join(_ROOT, "assets", "spot", "spot.xml")

# Canonical joint order (matches spot.xml actuator order):
# per leg -> hip-x (abduction), hip-y (thigh), knee.
_LEGS = ("fl", "fr", "hl", "hr")
_JOINTS = [f"{leg}_{dof}" for leg in _LEGS for dof in ("hx", "hy", "kn")]

# Default standing stance from the "home" keyframe ctrl: (hx=0, hy=1.04, kn=-1.8)
# repeated for all four legs. Base stands at z=0.46 in that keyframe.
_HOME_LEG = (0.0, 1.04, -1.8)
_HOME_QPOS = np.array(_HOME_LEG * len(_LEGS), dtype=np.float32)


def make_spot() -> RobotSpec:
    return RobotSpec(
        name="spot",
        mjcf_path=_SPOT_XML,
        base_body="body",
        nominal_height=0.46,
        joint_names=list(_JOINTS),
        actuator_names=list(_JOINTS),  # actuators share joint names in spot.xml
        home_joint_qpos=_HOME_QPOS,
        action_scale=0.3,  # ~17 deg of offset around the stance; tuned in Phase B
        foot_geom_names=["FL", "FR", "HL", "HR"],
    )
