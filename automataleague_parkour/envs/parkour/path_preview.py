"""Track path-preview sensor: K centerline points ahead of the agent, in its own frame,
plus the signed lateral offset. Batched torch (Warp + CPU). Appended to the observation
behind the optional height scan, gated by ParkourConfig.path_preview. Mirrors height_scan.
"""
from __future__ import annotations

import torch
from torch import Tensor

from automataleague_parkour.envs.parkour.height_scan import yaw_from_quat_torch
from automataleague_parkour.envs.parkour.navigation import centerline_project


def preview_dim(distances) -> int:
    """Observation width for a given set of lookahead distances: K points (x, y) plus
    the trailing signed lateral offset."""
    return 2 * len(distances) + 1


def cumulative_length(polyline: Tensor) -> Tensor:
    """Arc-length at each polyline vertex [M], starting at 0."""
    seg = torch.linalg.norm(polyline[1:] - polyline[:-1], dim=-1)
    return torch.cat([
        torch.zeros(1, device=polyline.device, dtype=polyline.dtype),
        torch.cumsum(seg, 0),
    ])


def _sample_at(cumlen: Tensor, polyline: Tensor, s: Tensor) -> Tensor:
    """Linear-interpolate the polyline at arc-lengths s [.] -> points [., 2]."""
    j = torch.searchsorted(cumlen, s.clamp(min=0.0), right=True).clamp(1, len(cumlen) - 1)
    s0, s1 = cumlen[j - 1], cumlen[j]
    w = ((s - s0) / (s1 - s0).clamp(min=1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    return polyline[j - 1] + w * (polyline[j] - polyline[j - 1])


def track_preview(
    base_xy: Tensor,
    base_quat: Tensor,
    centerline: Tensor,
    cumlen: Tensor,
    distances,
    closed: bool = True,
) -> Tensor:
    """K lookahead centerline points in the agent base frame (x ahead, y left),
    followed by the signed lateral offset. Returns [N, preview_dim(distances)].

    Closed tracks wrap arc-length modulo the total track length; open tracks clamp
    to the end so a lookahead past the finish still returns the last point.
    """
    dev = base_xy.device
    dists = torch.as_tensor(distances, device=dev, dtype=base_xy.dtype)  # [K]
    s0, signed_lat, _ = centerline_project(base_xy, centerline, cumlen)  # [N], [N]
    total = cumlen[-1]
    s = s0.unsqueeze(1) + dists.unsqueeze(0)  # [N, K]
    s = torch.remainder(s, total) if closed else s.clamp(max=total)
    samples = _sample_at(cumlen, centerline, s.reshape(-1)).reshape(s.shape[0], -1, 2)  # [N, K, 2]
    rel = samples - base_xy.unsqueeze(1)  # [N, K, 2] world-frame offset

    yaw = yaw_from_quat_torch(base_quat)  # [N]
    c, sn = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)  # [N, 1]
    lx = c * rel[..., 0] + sn * rel[..., 1]  # ahead
    ly = -sn * rel[..., 0] + c * rel[..., 1]  # left
    local = torch.stack([lx, ly], -1).reshape(s.shape[0], -1)  # [N, 2K]
    return torch.cat([local, signed_lat.unsqueeze(-1)], dim=-1)  # [N, 2K + 1]
