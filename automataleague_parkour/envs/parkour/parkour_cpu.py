"""Single-environment parkour on classic MuJoCo (CPU). Steppable + renderable.

Runs the shared task logic at batch size 1. Used for local validation and for
rendering training videos (wrapped by TorchRL GymWrapper in training/utils_ppo.py).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from torchrl.data import Composite, Unbounded

from automataleague_parkour.envs.parkour.config import (
    ParkourConfig,
    RewardConfig,
    TerminationConfig,
)
from automataleague_parkour.envs.parkour.navigation import (
    advance_checkpoints,
    centerline_frame,
    checkpoint_geometry,
    forward_velocity,
    point_to_polyline_distance,
    race_nav,
)
from automataleague_parkour.envs.parkour import height_scan as hs
from automataleague_parkour.envs.parkour import path_preview
from automataleague_parkour.envs.parkour.observation import build_observation
from automataleague_parkour.envs.parkour.rewards import compute_reward
from automataleague_parkour.envs.parkour.scene import build_parkour_model
from automataleague_parkour.envs.parkour.state import extract_state
from automataleague_parkour.envs.parkour.termination import compute_termination


class ParkourEnvCPU:
    def __init__(self, robot="spot", cfg=None, reward_cfg=None, term_cfg=None,
                 render_size=(720, 1280), frame_skip=10, reward_fn=None):
        self.cfg = cfg or ParkourConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self._reward_fn = reward_fn or compute_reward   # custom reward fn or the default
        self.term_cfg = term_cfg or TerminationConfig()
        self.frame_skip = frame_skip
        self.model, self.info = build_parkour_model(robot, self.cfg)
        self.robot = self.info.robot
        self.data = mujoco.MjData(self.model)

        self._checkpoints = torch.tensor(self.info.checkpoints_xy, dtype=torch.float32)
        self._centerline = torch.tensor(self.info.centerline, dtype=torch.float32)
        self._cumlen = path_preview.cumulative_length(self._centerline)
        self._race_mode = bool(self.cfg.race_mode)
        self._cp_tangent = (centerline_frame(self._checkpoints, self._centerline)[1]
                            if self._race_mode else None)
        self._home_joint = torch.tensor(self.robot.home_joint_qpos, dtype=torch.float32)
        self._act_ids = np.asarray(self.info.actuator_ids)
        self._ctrl0 = np.asarray(self.robot.home_joint_qpos, dtype=np.float64)
        self._action_scale = (self.cfg.action_scale if self.cfg.action_scale is not None
                              else self.robot.action_scale)
        self._scan_offsets = hs.scan_offsets() if self.cfg.height_scan else None
        self._preview_on = bool(self.cfg.path_preview)
        self._preview_dist = tuple(self.cfg.preview_distances)

        self._obs_dim = (
            self.robot.obs_dim
            + (hs.SCAN_N if self.cfg.height_scan else 0)
            + (path_preview.preview_dim(self._preview_dist) if self._preview_on else 0)
        )
        self.observation_spec = Composite(
            observation=Unbounded(shape=(self._obs_dim,), dtype=torch.float32),
            shape=torch.Size([]),
        )

        self._renderer = None
        self._render_size = render_size

    # --- helpers ---
    def _state(self):
        qpos = torch.tensor(self.data.qpos, dtype=torch.float32).unsqueeze(0)
        qvel = torch.tensor(self.data.qvel, dtype=torch.float32).unsqueeze(0)
        return extract_state(qpos, qvel, self.info)

    def _scan(self):
        """12-point terrain height scan (relative to foot level) as a [1, SCAN_N] tensor."""
        if self._scan_offsets is None:
            return None
        ba = self.info.base_qposadr
        base_xy = self.data.qpos[ba:ba + 2]
        base_z = float(self.data.qpos[ba + 2])
        w, x, y, z = self.data.qpos[ba + 3:ba + 7]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        query = hs.world_query_points(base_xy, yaw, self._scan_offsets)
        terrain_z, _ = hs.cpu_terrain_heights(self.model, self.data, query)
        rel = hs.scan_relative(terrain_z, base_z, self.robot.nominal_height)
        return torch.tensor(rel, dtype=torch.float32).unsqueeze(0)

    def _preview(self, st):
        """Lookahead centerline preview [1, preview_dim] in the robot's base frame."""
        if not self._preview_on:
            return None
        return path_preview.track_preview(
            st.base_pos[:, :2], st.base_quat, self._centerline, self._cumlen,
            self._preview_dist, closed=True,
        )

    def _obs(self):
        st = self._state()
        to_cp, dist, herr = checkpoint_geometry(st, self._checkpoints, self.cp_idx)
        return build_observation(st, to_cp, dist, herr, self.prev_action,
                                 self._home_joint, height_scan=self._scan(),
                                 track_preview=self._preview(st))

    def reset(self):
        self.data.qpos[:] = self.info.home_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.cp_idx = torch.zeros(1, dtype=torch.long)
        self.prev_action = torch.zeros(1, self.robot.n_joints)
        self.step_count = torch.zeros(1, dtype=torch.long)
        _, dist, _ = checkpoint_geometry(self._state(), self._checkpoints, self.cp_idx)
        self.prev_dist = dist
        return self._obs().squeeze(0).numpy()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self._ctrl0 + self._action_scale * action
        self.data.ctrl[self._act_ids] = target
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        st = self._state()
        if self._race_mode:
            fwd_vel, lateral, new_idx, inter, fin = race_nav(
                st.base_pos[:, :2], st.base_linvel_world[:, :2], self._centerline,
                self._checkpoints, self._cp_tangent, self.cp_idx, len(self._checkpoints))
            cur_dist = torch.zeros_like(fwd_vel)
        else:
            _, cur_dist, _ = checkpoint_geometry(st, self._checkpoints, self.cp_idx)
            fwd_vel = forward_velocity(st, self._checkpoints, self.cp_idx, cur_dist)
            lateral = point_to_polyline_distance(st.base_pos[:, :2], self._centerline)
            new_idx, inter, fin = advance_checkpoints(
                cur_dist, self.cp_idx, self.cfg.checkpoint_radius, len(self._checkpoints))
        terminated, truncated, fell, off, outcome = compute_termination(
            st, self.step_count, fin, lateral, self.cfg.half_width, self.term_cfg)
        act_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        reward, comps = self._reward_fn(
            st, self.prev_dist, cur_dist, inter, fin, fell, off, act_t,
            self.robot.nominal_height, self.reward_cfg, forward_vel=fwd_vel)

        # advance bookkeeping: recompute prev_dist for the (possibly advanced) checkpoint
        self.cp_idx = new_idx
        _, self.prev_dist, _ = checkpoint_geometry(st, self._checkpoints, self.cp_idx)
        self.prev_action = act_t

        obs = self._obs().squeeze(0).numpy()
        info = {
            "outcome": int(outcome.item()),
            "reward_components": {k: float(v.item()) for k, v in comps.items()},
        }
        return obs, float(reward.item()), bool(terminated.item()), bool(truncated.item()), info

    def render(self):
        if self._renderer is None:
            h, w = self._render_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        cam = mujoco.MjvCamera()
        base = self.info.base_qposadr
        cam.lookat = np.array([self.data.qpos[base], self.data.qpos[base + 1], 0.3])
        cam.azimuth, cam.elevation, cam.distance = 90, -20, 4.0
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()
