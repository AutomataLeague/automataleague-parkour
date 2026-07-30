"""Structured physics state shared by both backends.

Both the Warp and CPU envs read raw qpos/qvel, then call ``extract_state`` to get
a backend-agnostic ``ParkourState`` that the task logic (obs/reward/done) consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from automataleague_parkour.envs.parkour.scene import SceneInfo


@dataclass
class ParkourState:
    base_pos: Tensor            # [N,3] world position of the base
    base_quat: Tensor           # [N,4] (w,x,y,z)
    base_linvel_world: Tensor   # [N,3] linear velocity, world frame
    base_angvel_local: Tensor   # [N,3] angular velocity, body frame
    joint_pos: Tensor           # [N, n_joints]
    joint_vel: Tensor           # [N, n_joints]


def extract_state(qpos: Tensor, qvel: Tensor, info: SceneInfo) -> ParkourState:
    """Slice raw qpos/qvel into a structured, backend-agnostic state.

    MuJoCo free-joint convention: qvel[0:3] is linear velocity in the world frame,
    qvel[3:6] is angular velocity in the body frame.
    """
    qadr = info.base_qposadr
    dadr = info.base_dofadr
    jq = torch.as_tensor(info.joint_qposadr, device=qpos.device, dtype=torch.long)
    jd = torch.as_tensor(info.joint_dofadr, device=qvel.device, dtype=torch.long)
    return ParkourState(
        base_pos=qpos[:, qadr:qadr + 3],
        base_quat=qpos[:, qadr + 3:qadr + 7],
        base_linvel_world=qvel[:, dadr:dadr + 3],
        base_angvel_local=qvel[:, dadr + 3:dadr + 6],
        joint_pos=qpos[:, jq],
        joint_vel=qvel[:, jd],
    )
