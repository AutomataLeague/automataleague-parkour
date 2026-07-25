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


def point_to_polyline_distance(points: Tensor, polyline: Tensor) -> Tensor:
    """Perpendicular distance from each point [N,2] to a polyline [M,2] -> [N].

    Minimum over all segments of the point-to-segment distance. Used for off-path
    detection on curved centerlines (straight centerline reduces to |lateral|).
    """
    a = polyline[:-1]                      # [S,2] segment starts
    b = polyline[1:]                       # [S,2] segment ends
    ab = b - a                             # [S,2]
    p = points.unsqueeze(1)                # [N,1,2]
    ap = p - a.unsqueeze(0)                # [N,S,2]
    ab2 = (ab * ab).sum(-1).clamp(min=1e-9)          # [S]
    t = (ap * ab.unsqueeze(0)).sum(-1) / ab2         # [N,S]
    t = t.clamp(0.0, 1.0)
    proj = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)   # [N,S,2]
    dist = torch.linalg.norm(p - proj, dim=-1)       # [N,S]
    return dist.min(dim=1).values                    # [N]


def centerline_frame(points: Tensor, polyline: Tensor) -> tuple[Tensor, Tensor]:
    """Nearest-segment perpendicular distance [N] and unit track tangent [N,2].

    Same projection as point_to_polyline_distance, but also returns the direction of
    travel along the nearest segment (used by race mode for along-track speed).
    """
    a, b = polyline[:-1], polyline[1:]               # [S,2]
    ab = b - a
    seglen = torch.linalg.norm(ab, dim=-1).clamp(min=1e-9)
    tang = ab / seglen.unsqueeze(-1)                 # [S,2] unit tangents
    p = points.unsqueeze(1)                          # [N,1,2]
    ap = p - a.unsqueeze(0)
    ab2 = (ab * ab).sum(-1).clamp(min=1e-9)
    t = ((ap * ab.unsqueeze(0)).sum(-1) / ab2).clamp(0.0, 1.0)   # [N,S]
    proj = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)
    dist = torch.linalg.norm(p - proj, dim=-1)       # [N,S]
    seg = dist.argmin(dim=1)                          # [N] nearest segment
    lateral = dist.gather(1, seg.unsqueeze(1)).squeeze(1)
    return lateral, tang[seg]


def race_nav(
    base_xy: Tensor, base_linvel_xy: Tensor, centerline: Tensor,
    checkpoints_xy: Tensor, cp_tangent: Tensor, cp_idx: Tensor, num_cp: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Race-mode navigation: reward forward motion along the track and advance by
    crossing each gate's plane at ANY lateral position (no forced dip to the gate
    centre), so the agent is free to take the ideal line inside the corridor.

    Returns (fwd_vel_along_track, lateral, new_idx, reached_intermediate, reached_finish).
    """
    lateral, tangent = centerline_frame(base_xy, centerline)
    fwd_vel = (base_linvel_xy * tangent).sum(dim=-1)             # along-track speed
    # gate crossed once the robot is past the gate plane (dot with the gate tangent > 0)
    passed = ((base_xy - checkpoints_xy[cp_idx]) * cp_tangent[cp_idx]).sum(dim=-1) > 0
    is_last = cp_idx == (num_cp - 1)
    reached_finish = passed & is_last
    reached_intermediate = passed & ~is_last
    new_idx = torch.where(reached_intermediate, cp_idx + 1, cp_idx)
    return fwd_vel, lateral, new_idx, reached_intermediate, reached_finish


def forward_velocity(
    state: ParkourState, checkpoints_xy: Tensor, cp_idx: Tensor, dist: Tensor
) -> Tensor:
    """Component of base velocity (m/s) pointing toward the current checkpoint."""
    to_cp = checkpoints_xy[cp_idx] - state.base_pos[:, :2]      # [N,2]
    goal_dir = to_cp / (dist.unsqueeze(-1) + 1e-6)
    return (state.base_linvel_world[:, :2] * goal_dir).sum(dim=-1)


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
