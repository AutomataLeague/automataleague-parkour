"""Forward height scan — a small terrain sensor for perceptive locomotion.

Blind (proprioceptive-only) policies stall at physical obstacles: the robot only
feels terrain once it collides with it. This sensor casts a grid of downward rays
*ahead* of the robot (in its heading frame) and reports how much higher/lower the
terrain is there, relative to the robot's own foot level — so the policy sees a
step / ramp / bank coming and can prepare.

The scan is expressed relative to foot level (``base_z - nominal_height``): on flat
ground every sample reads ~0; a 0.1 m step ahead reads +0.1; and when the robot is
standing *on* a raised platform, the platform surface ahead still reads ~0 (it is
level with the feet). This terrain-relative framing is what makes the signal
transferable across the track.

Rays are filtered to geom **group 0** (floor + obstacles); the robot's own geoms
live in groups 2-3, so the ray passes through the body and reads the ground beneath.
Both backends share the same offset grid: ``mj_ray`` on CPU, ``mjw.rays`` on GPU.
"""

from __future__ import annotations

import numpy as np

# Scan grid in the base heading frame: forward (x) distances x lateral (y) offsets.
# 4 forward x 3 lateral = 12 samples covering a ~0.75 m x 0.60 m patch ahead.
SCAN_FWD = (0.20, 0.45, 0.70, 0.95)
SCAN_LAT = (-0.30, 0.0, 0.30)
SCAN_N = len(SCAN_FWD) * len(SCAN_LAT)

_RAY_H = 3.0          # cast rays from 3 m up (above any obstacle)
_CLIP = 1.0           # clip terrain-relative height to +/- 1 m
# geomgroup mask: include only group 0 (floor + obstacles), skip the robot (groups 2-3)
_GROUP0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


def scan_offsets() -> np.ndarray:
    """(SCAN_N, 2) query offsets in the base heading frame (forward=x, left=y)."""
    return np.array([[fwd, lat] for fwd in SCAN_FWD for lat in SCAN_LAT], dtype=np.float64)


def world_query_points(base_xy, yaw: float, offsets: np.ndarray) -> np.ndarray:
    """Rotate the base-frame offsets by `yaw` and translate to world (x, y)."""
    c, s = np.cos(yaw), np.sin(yaw)
    fwd, lat = offsets[:, 0], offsets[:, 1]
    x = base_xy[0] + c * fwd - s * lat
    y = base_xy[1] + s * fwd + c * lat
    return np.stack([x, y], axis=1)


def scan_relative(terrain_z: np.ndarray, base_z: float, nominal_height: float) -> np.ndarray:
    """Terrain height at each sample relative to the robot's foot level, clipped."""
    foot_z = base_z - nominal_height
    return np.clip(terrain_z - foot_z, -_CLIP, _CLIP)


def yaw_from_quat_torch(quat):
    """Batched yaw (rotation about world z) from wxyz quaternions [N, 4]."""
    import torch

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def scan_offsets_torch(device):
    """(SCAN_N, 2) base-frame query offsets as a float32 tensor."""
    import torch

    return torch.tensor(scan_offsets(), dtype=torch.float32, device=device)


def world_query_points_torch(base_xy, yaw, offsets):
    """Rotate base-frame offsets [K,2] by per-env yaw [N] and translate to world.

    Returns world (x, y) query points [N, K, 2].
    """
    import torch

    c, s = torch.cos(yaw), torch.sin(yaw)                      # [N]
    fwd, lat = offsets[:, 0], offsets[:, 1]                    # [K]
    x = base_xy[:, 0:1] + c[:, None] * fwd[None] - s[:, None] * lat[None]
    y = base_xy[:, 1:2] + s[:, None] * fwd[None] + c[:, None] * lat[None]
    return torch.stack([x, y], dim=-1)                         # [N, K, 2]


def scan_relative_torch(terrain_z, base_z, nominal_height: float):
    """Terrain height [N,K] relative to per-env foot level [N], clipped to +/- 1 m."""
    foot_z = (base_z - nominal_height).unsqueeze(-1)          # [N, 1]
    return (terrain_z - foot_z).clamp(-_CLIP, _CLIP)


def cpu_terrain_heights(model, data, query_xy: np.ndarray):
    """Raycast straight down at each world (x, y); return (terrain_z, hit_points).

    terrain_z[i] is the world height of the first group-0 geom under query i
    (0 on the flat floor). hit_points is (N, 3) for rendering the rays.
    """
    import mujoco

    n = query_xy.shape[0]
    terrain_z = np.empty(n, dtype=np.float64)
    hits = np.empty((n, 3), dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gid = np.zeros(1, dtype=np.int32)
    for i in range(n):
        pnt = np.array([query_xy[i, 0], query_xy[i, 1], _RAY_H], dtype=np.float64)
        dist = mujoco.mj_ray(model, data, pnt, vec, _GROUP0, 1, -1, gid)
        tz = _RAY_H - dist if dist >= 0 else 0.0
        terrain_z[i] = tz
        hits[i] = [pnt[0], pnt[1], tz]
    return terrain_z, hits
