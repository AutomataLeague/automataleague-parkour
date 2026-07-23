"""Batched parkour reward.

Core signal: progress = reduction in distance to the current checkpoint. Plus
checkpoint/finish bonuses, upright/height/alive shaping, and action/joint-velocity
penalties. Terminal fall / off-path penalties applied via masks.
"""

from __future__ import annotations

import torch
from torch import Tensor

from automataleague.envs.parkour.config import RewardConfig
from automataleague.envs.parkour.spatial import tilt_angle
from automataleague.envs.parkour.state import ParkourState


def compute_reward(
    state: ParkourState,
    prev_dist: Tensor,
    cur_dist: Tensor,
    reached_intermediate: Tensor,
    reached_finish: Tensor,
    fell: Tensor,
    off_path: Tensor,
    action: Tensor,
    nominal_height: float,
    rc: RewardConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    progress = rc.progress * (prev_dist - cur_dist)
    alive = torch.full_like(progress, rc.alive)
    upright = rc.upright * torch.exp(-2.0 * tilt_angle(state.base_quat))
    height = rc.height * torch.exp(
        -10.0 * (state.base_pos[:, 2] - nominal_height) ** 2
    )
    action_pen = -rc.action * (action ** 2).sum(dim=-1)
    jointvel_pen = -rc.joint_vel * (state.joint_vel ** 2).sum(dim=-1)

    checkpoint_bonus = rc.checkpoint * reached_intermediate.float()
    success_bonus = rc.success * reached_finish.float()
    fall_pen = -rc.fall * fell.float()
    offpath_pen = -rc.off_path * off_path.float()

    components = {
        "progress": progress, "alive": alive, "upright": upright, "height": height,
        "action": action_pen, "joint_vel": jointvel_pen,
        "checkpoint": checkpoint_bonus, "success": success_bonus,
        "fall": fall_pen, "off_path": offpath_pen,
    }
    total = sum(components.values())
    return total.unsqueeze(-1), components
