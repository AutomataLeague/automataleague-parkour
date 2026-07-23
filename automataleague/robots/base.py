"""Robot abstraction for the automataleague.

A ``RobotSpec`` is everything a task needs to know about a robot without knowing
its internals: where its model lives, how it stands, the order of its actuated
joints, and how big an action step is. Tasks (parkour, sumo, ...) are written
against this contract, so a new robot is a new ``RobotSpec`` and nothing else.

Observation/action dimensionality is *derived* from the joint count, never
hardcoded — so different robots naturally produce differently-sized policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class RobotSpec:
    """Static description of a robot for use by any task.

    Attributes:
        name: Registry key, e.g. ``"spot"``.
        mjcf_path: Absolute path to the robot's MJCF file (with its mesh assets).
        base_body: Name of the floating-base body (used for base-frame transforms
            and fall detection). Unprefixed — the scene composer adds a prefix.
        nominal_height: Standing height of the base (metres). Used by the height
            reward and the fall threshold.
        joint_names: Actuated joints in canonical order. Defines the ordering of
            joint-position/velocity observations and of the action vector.
        actuator_names: Actuators in the same order as ``joint_names``.
        home_joint_qpos: Default-stance angle for each actuated joint (radians),
            same order as ``joint_names``. The action offsets from this pose.
        action_scale: Magnitude (radians) of the position offset a unit action
            applies around ``home_joint_qpos``.
    """

    name: str
    mjcf_path: str
    base_body: str
    nominal_height: float
    joint_names: list[str]
    actuator_names: list[str]
    home_joint_qpos: np.ndarray
    action_scale: float = 0.3

    def __post_init__(self):
        self.home_joint_qpos = np.asarray(self.home_joint_qpos, dtype=np.float32)
        if len(self.home_joint_qpos) != len(self.joint_names):
            raise ValueError(
                f"{self.name}: home_joint_qpos has {len(self.home_joint_qpos)} entries "
                f"but there are {len(self.joint_names)} joints"
            )
        if len(self.actuator_names) != len(self.joint_names):
            raise ValueError(
                f"{self.name}: {len(self.actuator_names)} actuators vs "
                f"{len(self.joint_names)} joints — must match 1:1"
            )

    # --- derived dimensions -------------------------------------------------
    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def action_dim(self) -> int:
        return self.n_joints

    @property
    def obs_dim(self) -> int:
        """Proprioception + task observation size.

        Layout (see envs/parkour): base_lin_vel(3) + base_ang_vel(3) +
        projected_gravity(3) + joint_pos(n) + joint_vel(n) + prev_action(n) +
        task(4).
        """
        return 3 + 3 + 3 + 3 * self.n_joints + 4

    # --- model loading ------------------------------------------------------
    def load_spec(self) -> mujoco.MjSpec:
        """Load the robot as an editable ``MjSpec`` for scene composition."""
        return mujoco.MjSpec.from_file(self.mjcf_path)
