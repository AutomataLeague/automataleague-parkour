"""Assemble the policy observation from structured state + navigation info.

Layout (obs_dim = 3+3+3 + 3*n_joints + 4, robot-derived):
  base_linvel_base(3) | base_angvel(3) | proj_gravity(3)
  | joint_pos-home(nj) | joint_vel(nj) | prev_action(nj)
  | to_cp_base(2) | dist(1) | heading_error(1)
"""

from __future__ import annotations

import torch
from torch import Tensor

from automataleague.envs.parkour.spatial import projected_gravity, quat_rotate_inverse
from automataleague.envs.parkour.state import ParkourState


def build_observation(
    state: ParkourState,
    to_cp_base: Tensor,
    dist: Tensor,
    heading_error: Tensor,
    prev_action: Tensor,
    home_joint_qpos: Tensor,
) -> Tensor:
    home = home_joint_qpos.to(state.joint_pos.device, state.joint_pos.dtype)
    lin_vel_base = quat_rotate_inverse(state.base_quat, state.base_linvel_world)
    grav = projected_gravity(state.base_quat)
    return torch.cat(
        [
            lin_vel_base,                       # 3
            state.base_angvel_local,            # 3
            grav,                               # 3
            state.joint_pos - home,             # nj
            state.joint_vel,                    # nj
            prev_action,                        # nj
            to_cp_base,                         # 2
            dist.unsqueeze(-1),                 # 1
            heading_error.unsqueeze(-1),        # 1
        ],
        dim=-1,
    )
