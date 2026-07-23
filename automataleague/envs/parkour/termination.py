"""Batched termination + outcome codes for the parkour task.

Codes: 0 ongoing, 1 success, 2 fell, 3 off_path (success has priority).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from automataleague.envs.parkour.config import TerminationConfig
from automataleague.envs.parkour.spatial import tilt_angle
from automataleague.envs.parkour.state import ParkourState


def compute_termination(
    state: ParkourState,
    step_count: Tensor,
    reached_finish: Tensor,
    lateral_dist: Tensor,
    half_width: float,
    tc: TerminationConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """`lateral_dist` [N] is the perpendicular distance from the base to the path
    centerline (see navigation.point_to_polyline_distance)."""
    device = state.base_pos.device
    max_tilt = math.radians(tc.max_tilt_deg)

    fell = (state.base_pos[:, 2] < tc.fall_height) | (tilt_angle(state.base_quat) > max_tilt)
    if tc.off_path:
        off_path = lateral_dist > half_width
    else:
        off_path = torch.zeros_like(fell)

    outcome = torch.zeros(state.base_pos.shape[0], dtype=torch.int32, device=device)
    outcome = torch.where(off_path, torch.tensor(3, device=device), outcome)
    outcome = torch.where(fell, torch.tensor(2, device=device), outcome)
    outcome = torch.where(reached_finish, torch.tensor(1, device=device), outcome)

    terminated = fell | off_path | reached_finish
    truncated = step_count >= tc.max_episode_steps
    return terminated, truncated, fell, off_path, outcome
