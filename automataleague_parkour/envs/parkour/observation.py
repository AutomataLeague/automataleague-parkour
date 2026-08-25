"""Assemble the policy observation from structured state + navigation info.

Layout (obs_dim = 3+3+3 + 3*n_joints + 4, robot-derived):
  base_linvel_base(3) | base_angvel(3) | proj_gravity(3)
  | joint_pos-home(nj) | joint_vel(nj) | prev_action(nj)
  | to_cp_base(2) | dist(1) | heading_error(1)
  [ | height_scan(SCAN_N) ]        # optional terrain sensor
  [ | track_preview(PREVIEW_N) ]   # optional path lookahead, appended last

Optional blocks are appended AFTER the proprioceptive block, in the fixed order
above, so the leading columns stay identical whether or not they're present. A
policy trained without one can be loaded into the sensored network by padding
just the new trailing input columns with zeros.

That padding is sound only while the observation is **append-only**. Two optional
blocks exist, so "the observation got wider" no longer implies "the old columns
still mean the same thing": `[proprio | height_scan]` (61) and
`[proprio | track_preview]` (66) differ in width *and* in meaning. `obs_layout`
names the blocks so a warm-start can be checked block-wise rather than by width
(see `training.models.check_obs_layout_compatible`).
"""

from __future__ import annotations

import torch
from torch import Tensor

from automataleague_parkour.envs.parkour.spatial import projected_gravity, quat_rotate_inverse
from automataleague_parkour.envs.parkour.state import ParkourState


def build_observation(
    state: ParkourState,
    to_cp_base: Tensor,
    dist: Tensor,
    heading_error: Tensor,
    prev_action: Tensor,
    home_joint_qpos: Tensor,
    height_scan: Tensor | None = None,
    track_preview: Tensor | None = None,
) -> Tensor:
    home = home_joint_qpos.to(state.joint_pos.device, state.joint_pos.dtype)
    lin_vel_base = quat_rotate_inverse(state.base_quat, state.base_linvel_world)
    grav = projected_gravity(state.base_quat)
    parts = [
        lin_vel_base,                       # 3
        state.base_angvel_local,            # 3
        grav,                               # 3
        state.joint_pos - home,             # nj
        state.joint_vel,                    # nj
        prev_action,                        # nj
        to_cp_base,                         # 2
        dist.unsqueeze(-1),                 # 1
        heading_error.unsqueeze(-1),        # 1
    ]
    if height_scan is not None:
        parts.append(height_scan)           # SCAN_N (terrain sensor)
    if track_preview is not None:
        parts.append(track_preview)         # PREVIEW_N (path lookahead, appended last)
    return torch.cat(parts, dim=-1)


def obs_layout(course, robot) -> tuple[tuple[str, int], ...]:
    """The ordered ``(block_name, width)`` layout the observation for `course` has.

    The single source of truth for observation width: the envs, `build_actor` and
    the warm-start check all read it, so a network can't be sized from arithmetic
    that has drifted from what `build_observation` actually concatenates.
    """
    from automataleague_parkour.envs.parkour.height_scan import SCAN_N
    from automataleague_parkour.envs.parkour.path_preview import (
        course_get,
        preview_dim,
        resolve_perception,
    )

    blocks: list[tuple[str, int]] = [("proprio", robot.obs_dim)]
    if bool(course_get(course, "height_scan", False)):
        blocks.append(("height_scan", SCAN_N))
    preview_on, mode = resolve_perception(course)
    if preview_on:
        dists = course_get(course, "preview_distances", (1.5, 3.0, 4.5, 6.0))
        blocks.append(("track_preview", preview_dim(dists, mode)))
    return tuple(blocks)


def obs_width(course, robot) -> int:
    """Total observation width for `course`, i.e. `sum` over `obs_layout` blocks."""
    return sum(width for _, width in obs_layout(course, robot))
