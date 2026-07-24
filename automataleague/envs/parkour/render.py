"""Multi-perspective rendering for parkour policies.

Design: *record once, render many*. Roll a policy through one episode and store the
qpos trajectory, then replay it into a fresh MjData and render from any number of
named cameras — synchronized views of the same run. Adding a camera never re-runs
physics or the policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class CameraPreset:
    """A tracking camera. lookat follows the robot base; azimuth may follow heading."""

    name: str
    azimuth: float          # degrees (added to robot yaw if heading_relative)
    elevation: float        # degrees (negative looks down)
    distance: float         # metres from lookat
    lookat_offset: tuple = (0.0, 0.0, 0.3)   # offset from base position (x,y,z)
    heading_relative: bool = False           # azimuth relative to robot yaw


# Registry of camera perspectives. Tuned for a robot facing +x down the corridor.
CAMERAS: dict[str, CameraPreset] = {
    # Classic side tracking (corridor runs left-right in frame).
    "side": CameraPreset("side", azimuth=90, elevation=-12, distance=4.0),
    # Over-the-shoulder: behind and above, looking forward down the course.
    # azimuth=0 (heading-relative) -> camera sits behind the robot looking along its
    # heading, so the robot runs *away* from the viewer (into the course ahead).
    "over_shoulder": CameraPreset(
        "over_shoulder", azimuth=0, elevation=-18, distance=2.6,
        lookat_offset=(0.6, 0.0, 0.35), heading_relative=True,
    ),
    # Drone: high aerial from directly behind, looking forward (see what's ahead).
    "drone": CameraPreset("drone", azimuth=0, elevation=-42, distance=6.0,
                          lookat_offset=(1.2, 0.0, 0.3), heading_relative=True),
    # Front: ahead of the robot, looking back at it (azimuth=180 -> camera in front).
    "front": CameraPreset("front", azimuth=180, elevation=-12, distance=3.2,
                          lookat_offset=(0.3, 0.0, 0.35), heading_relative=True),
    # Low chase from directly behind.
    "chase": CameraPreset("chase", azimuth=0, elevation=-6, distance=2.8,
                          lookat_offset=(0.4, 0.0, 0.35), heading_relative=True),
    # Top-down, oriented to the robot's heading (azimuth=0 -> forward = up in frame).
    "top": CameraPreset("top", azimuth=0, elevation=-89, distance=6.5,
                        heading_relative=True),
}


def _yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def make_camera(preset: CameraPreset, base_pos, base_yaw: float = 0.0) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    ox, oy, oz = preset.lookat_offset
    if preset.heading_relative:
        # Rotate the in-plane lookat offset into the robot's heading frame, and add
        # the yaw to the azimuth, so the framing stays fixed to the robot as it turns.
        c, s = math.cos(base_yaw), math.sin(base_yaw)
        ox, oy = ox * c - oy * s, ox * s + oy * c
        az = preset.azimuth + math.degrees(base_yaw)
    else:
        az = preset.azimuth
    cam.lookat = np.array([base_pos[0] + ox, base_pos[1] + oy, oz])
    cam.azimuth = az
    cam.elevation = preset.elevation
    cam.distance = preset.distance
    return cam


def record_trajectory(env, policy, max_steps, policy_device="cuda", loop_episodes=True):
    """Roll `policy` through the CPU env, recording qpos each step -> [T, nq]."""
    import torch
    from tensordict import TensorDict
    from torchrl.envs import ExplorationType, set_exploration_type

    obs = env.reset()
    qpos = [env.data.qpos.copy()]
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(max_steps):
            td = TensorDict(
                {"observation": torch.as_tensor(obs, dtype=torch.float32,
                                                device=policy_device).unsqueeze(0)},
                batch_size=[1],
            )
            td = policy(td)
            action = td["action"].squeeze(0).cpu().numpy()
            obs, _, term, trunc, _ = env.step(action)
            qpos.append(env.data.qpos.copy())
            if term or trunc:
                if not loop_episodes:
                    break
                obs = env.reset()
    return np.stack(qpos)


def render_trajectory(model, info, qpos_seq, camera="side", size=(720, 1280)):
    """Replay a qpos trajectory and render from one camera preset -> [T,H,W,3] uint8."""
    preset = CAMERAS[camera] if isinstance(camera, str) else camera
    data = mujoco.MjData(model)
    h, w = size
    renderer = mujoco.Renderer(model, height=h, width=w)
    base = info.base_qposadr
    frames = []
    for q in qpos_seq:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        yaw = _yaw_from_quat(q[base + 3:base + 7])
        cam = make_camera(preset, q[base:base + 3], yaw)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    renderer.close()
    return np.stack(frames)


def grid_frames(view_frames: dict[str, np.ndarray]) -> np.ndarray:
    """Tile multiple synchronized [T,H,W,3] view stacks into a 2-col grid [T,H',W',3]."""
    names = list(view_frames)
    stacks = [view_frames[n] for n in names]
    T = min(s.shape[0] for s in stacks)
    stacks = [s[:T] for s in stacks]
    ncol = 2
    nrow = (len(stacks) + ncol - 1) // ncol
    h, w = stacks[0].shape[1:3]
    out = np.zeros((T, nrow * h, ncol * w, 3), dtype=np.uint8)
    for i, s in enumerate(stacks):
        r, c = divmod(i, ncol)
        out[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = s
    return out
