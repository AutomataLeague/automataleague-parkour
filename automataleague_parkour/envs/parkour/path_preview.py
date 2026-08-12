"""Track path-preview sensor: K lookahead points ahead of the agent, in its own frame,
plus the signed lateral offset. Batched torch (Warp + CPU). Appended to the observation
behind the optional height scan, gated by ParkourConfig.path_preview. Mirrors height_scan.

Two modes, selected by ParkourConfig.preview_mode:
  "centerline" (default): K centerline points, so the policy learns to track the
  centerline of the corridor.
  "boundaries": a LEFT and a RIGHT corridor-edge point per lookahead, so the policy
  perceives the drivable channel width and can find a tighter line instead of hugging
  the centerline.
"""
from __future__ import annotations

import torch
from torch import Tensor

from automataleague_parkour.envs.parkour.height_scan import yaw_from_quat_torch
from automataleague_parkour.envs.parkour.navigation import centerline_project


def preview_dim(distances, mode: str = "centerline") -> int:
    """Observation width for a given set of lookahead distances, plus the trailing
    signed lateral offset. "centerline": K points (x, y). "boundaries": K point pairs,
    a left and a right corridor-edge point (x, y) each."""
    n = len(distances)
    return (4 * n + 1) if mode == "boundaries" else (2 * n + 1)


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


def _sample_at_tangent(cumlen: Tensor, polyline: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
    """Linear-interpolate the polyline at arc-lengths s [.] -> points [., 2], plus the
    unit tangent of the polyline segment each sample falls in [., 2]. Used by the
    "boundaries" preview mode to place a left/right point at each lookahead."""
    j = torch.searchsorted(cumlen, s.clamp(min=0.0), right=True).clamp(1, len(cumlen) - 1)
    s0, s1 = cumlen[j - 1], cumlen[j]
    w = ((s - s0) / (s1 - s0).clamp(min=1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    seg = polyline[j] - polyline[j - 1]
    tangent = seg / torch.linalg.norm(seg, dim=-1, keepdim=True).clamp(min=1e-9)
    point = polyline[j - 1] + w * seg
    return point, tangent


def track_preview(
    base_xy: Tensor,
    base_quat: Tensor,
    centerline: Tensor,
    cumlen: Tensor,
    distances,
    closed: bool = True,
    mode: str = "centerline",
    half_width: float = 1.6,
) -> Tensor:
    """K lookahead points in the agent base frame (x ahead, y left), followed by the
    signed lateral offset. Returns [N, preview_dim(distances, mode)].

    "centerline" mode samples the centerline itself at each lookahead. "boundaries"
    mode samples the LEFT and RIGHT corridor edges instead: at each lookahead
    arc-length it takes the unit track tangent of the polyline segment the sample
    falls in, the left normal n = [-tangent_y, tangent_x], and places
    left = sample + half_width * n, right = sample - half_width * n, so the policy
    perceives the drivable channel rather than only its centre.

    Closed tracks wrap arc-length modulo the total track length; open tracks clamp
    to the end so a lookahead past the finish still returns the last point.
    """
    dev = base_xy.device
    dists = torch.as_tensor(distances, device=dev, dtype=base_xy.dtype)  # [K]
    s0, signed_lat, _ = centerline_project(base_xy, centerline, cumlen)  # [N], [N]
    total = cumlen[-1]
    s = s0.unsqueeze(1) + dists.unsqueeze(0)  # [N, K]
    s = torch.remainder(s, total) if closed else s.clamp(max=total)

    yaw = yaw_from_quat_torch(base_quat)  # [N]
    c, sn = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)  # [N, 1]

    def to_base_frame(rel):
        lx = c * rel[..., 0] + sn * rel[..., 1]  # ahead
        ly = -sn * rel[..., 0] + c * rel[..., 1]  # left
        return torch.stack([lx, ly], -1)

    if mode == "boundaries":
        samples, tangents = _sample_at_tangent(cumlen, centerline, s.reshape(-1))
        samples = samples.reshape(s.shape[0], -1, 2)    # [N, K, 2]
        tangents = tangents.reshape(s.shape[0], -1, 2)  # [N, K, 2]
        normals = torch.stack([-tangents[..., 1], tangents[..., 0]], -1)  # left normal [N, K, 2]
        left = samples + half_width * normals
        right = samples - half_width * normals
        local_left = to_base_frame(left - base_xy.unsqueeze(1))    # [N, K, 2]
        local_right = to_base_frame(right - base_xy.unsqueeze(1))  # [N, K, 2]
        pairs = torch.cat([local_left, local_right], dim=-1)  # [N, K, 4]: lx, ly, rx, ry
        local = pairs.reshape(s.shape[0], -1)  # [N, 4K]
        return torch.cat([local, signed_lat.unsqueeze(-1)], dim=-1)  # [N, 4K + 1]

    samples = _sample_at(cumlen, centerline, s.reshape(-1)).reshape(s.shape[0], -1, 2)  # [N, K, 2]
    rel = samples - base_xy.unsqueeze(1)  # [N, K, 2] world-frame offset
    local = to_base_frame(rel).reshape(s.shape[0], -1)  # [N, 2K]
    return torch.cat([local, signed_lat.unsqueeze(-1)], dim=-1)  # [N, 2K + 1]
