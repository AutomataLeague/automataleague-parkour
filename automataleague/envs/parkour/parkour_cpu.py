"""Single-environment parkour on classic MuJoCo (CPU). Steppable + renderable.

Runs the shared task logic at batch size 1. Used for local validation and for
rendering training videos (wrapped by TorchRL GymWrapper in training/utils_ppo.py).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from automataleague.envs.parkour.config import (
    ParkourConfig,
    RewardConfig,
    TerminationConfig,
)
from automataleague.envs.parkour.navigation import (
    advance_checkpoints,
    checkpoint_geometry,
    forward_velocity,
)
from automataleague.envs.parkour.observation import build_observation
from automataleague.envs.parkour.rewards import compute_reward
from automataleague.envs.parkour.scene import build_parkour_model
from automataleague.envs.parkour.state import extract_state
from automataleague.envs.parkour.termination import compute_termination


class ParkourEnvCPU:
    def __init__(self, robot="spot", cfg=None, reward_cfg=None, term_cfg=None,
                 render_size=(720, 1280), frame_skip=10):
        self.cfg = cfg or ParkourConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.term_cfg = term_cfg or TerminationConfig()
        self.frame_skip = frame_skip
        self.model, self.info = build_parkour_model(robot, self.cfg)
        self.robot = self.info.robot
        self.data = mujoco.MjData(self.model)

        self._checkpoints = torch.tensor(self.info.checkpoints_xy, dtype=torch.float32)
        self._home_joint = torch.tensor(self.robot.home_joint_qpos, dtype=torch.float32)
        self._act_ids = np.asarray(self.info.actuator_ids)
        self._ctrl0 = np.asarray(self.robot.home_joint_qpos, dtype=np.float64)

        self._renderer = None
        self._render_size = render_size

    # --- helpers ---
    def _state(self):
        qpos = torch.tensor(self.data.qpos, dtype=torch.float32).unsqueeze(0)
        qvel = torch.tensor(self.data.qvel, dtype=torch.float32).unsqueeze(0)
        return extract_state(qpos, qvel, self.info)

    def _obs(self):
        st = self._state()
        to_cp, dist, herr = checkpoint_geometry(st, self._checkpoints, self.cp_idx)
        return build_observation(st, to_cp, dist, herr, self.prev_action, self._home_joint)

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
        target = self._ctrl0 + self.robot.action_scale * action
        self.data.ctrl[self._act_ids] = target
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        st = self._state()
        to_cp, cur_dist, herr = checkpoint_geometry(st, self._checkpoints, self.cp_idx)
        fwd_vel = forward_velocity(st, self._checkpoints, self.cp_idx, cur_dist)
        new_idx, inter, fin = advance_checkpoints(
            cur_dist, self.cp_idx, self.cfg.checkpoint_radius, len(self._checkpoints))
        terminated, truncated, fell, off, outcome = compute_termination(
            st, self.step_count, fin, self.cfg.half_width, self.term_cfg)
        act_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        reward, comps = compute_reward(
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
