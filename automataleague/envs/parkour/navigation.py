"""Checkpoint navigation: distance/heading to the current gate, and advancement."""

from __future__ import annotations

import torch
from torch import Tensor

from automataleague.envs.parkour.spatial import quat_rotate_inverse
from automataleague.envs.parkour.state import ParkourState


def checkpoint_geometry(
    state: ParkourState, checkpoints_xy: Tensor, cp_idx: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Vector-to-gate (base frame), distance, and heading error for each env."""
    target = checkpoints_xy[cp_idx]                       # [N,2]
    to_cp_world = target - state.base_pos[:, :2]          # [N,2]
    dist = torch.linalg.norm(to_cp_world, dim=-1)         # [N]

    # Express the (x,y,0) target vector in the base frame, then keep xy.
    vec3 = torch.cat([to_cp_world, torch.zeros_like(to_cp_world[:, :1])], dim=-1)
    to_cp_base = quat_rotate_inverse(state.base_quat, vec3)[:, :2]
    heading_error = torch.atan2(to_cp_base[:, 1], to_cp_base[:, 0])
    return to_cp_base, dist, heading_error


def advance_checkpoints(
    dist: Tensor, cp_idx: Tensor, cp_radius: float, num_cp: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Advance the current-checkpoint index for envs within ``cp_radius``.

    Returns (new_idx, reached_intermediate, reached_finish).
    """
    reached = dist < cp_radius
    is_finish = cp_idx == (num_cp - 1)
    reached_finish = reached & is_finish
    reached_intermediate = reached & ~is_finish
    new_idx = torch.where(reached_intermediate, cp_idx + 1, cp_idx)
    return new_idx, reached_intermediate, reached_finish
