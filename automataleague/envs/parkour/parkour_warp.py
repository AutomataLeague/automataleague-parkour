"""GPU-batched parkour environment using MuJoCo Warp.

Steps thousands of parallel Spot (or any registered robot) simulations on GPU via
NVIDIA Warp, with observation/reward/termination in batched PyTorch (shared with the
CPU env). Ported from smaller_projects/spaceX/env/rocket_landing_warp.py.

Usage with TorchRL:
    env = ParkourEnvWarp(robot="spot", num_envs=1024, device="cuda")
    td = env.reset()
    td = env.step(td)
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from automataleague.envs.parkour.config import (
    ParkourConfig,
    RewardConfig,
    TerminationConfig,
)
from automataleague.envs.parkour.navigation import advance_checkpoints, checkpoint_geometry
from automataleague.envs.parkour.observation import build_observation
from automataleague.envs.parkour.rewards import compute_reward
from automataleague.envs.parkour.scene import build_parkour_model
from automataleague.envs.parkour.state import extract_state
from automataleague.envs.parkour.termination import compute_termination


class ParkourEnvWarp(EnvBase):
    """Batched GPU parkour environment. All N envs step simultaneously."""

    def __init__(
        self,
        robot: str = "spot",
        num_envs: int = 1024,
        device: str = "cuda",
        cfg: ParkourConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        term_cfg: TerminationConfig | None = None,
        frame_skip: int = 10,
        # per-world constraint/contact buffer sizes for MuJoCo-Warp. Spot is
        # contact-rich (4 condim=6 feet); defaults must exceed the peak constraint
        # count or MuJoCo-Warp silently drops constraints ("nefc overflow").
        njmax: int = 200,
        nconmax: int = 40,
        # reset noise (small domain randomization on the start pose)
        reset_pos_noise: float = 0.05,
        reset_joint_noise: float = 0.05,
    ):
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._frame_skip = frame_skip
        self._reset_pos_noise = reset_pos_noise
        self._reset_joint_noise = reset_joint_noise

        self.cfg = cfg or ParkourConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.term_cfg = term_cfg or TerminationConfig()

        # Build the composed model + resolved scene info (Phase A).
        self._mjm, self.info = build_parkour_model(robot, self.cfg)
        self.robot = self.info.robot

        self._mjw_model = mjw.put_model(self._mjm)
        self._mjw_data = mjw.make_data(
            self._mjm, nworld=num_envs, nconmax=nconmax, njmax=njmax
        )

        self._nq = self._mjm.nq
        self._nv = self._mjm.nv
        self._nu = self._mjm.nu
        self._obs_dim = self.robot.obs_dim
        self._act_dim = self.robot.action_dim

        # Device tensors used every step.
        d = self._device
        self._checkpoints = torch.tensor(self.info.checkpoints_xy, dtype=torch.float32, device=d)
        self._num_cp = self._checkpoints.shape[0]
        self._home_joint = torch.tensor(self.robot.home_joint_qpos, dtype=torch.float32, device=d)
        self._home_qpos = torch.tensor(self.info.home_qpos, dtype=torch.float32, device=d)
        self._act_cols = torch.tensor(self.info.actuator_ids, dtype=torch.long, device=d)
        self._base_qadr = self.info.base_qposadr
        self._joint_qadr = torch.tensor(self.info.joint_qposadr, dtype=torch.long, device=d)

        # Per-env task state.
        self.cp_idx = torch.zeros(num_envs, dtype=torch.long, device=d)
        self.prev_dist = torch.zeros(num_envs, device=d)
        self.prev_action = torch.zeros(num_envs, self._act_dim, device=d)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=d)

        self._graph = None
        self._capture_cuda_graph()

        super().__init__(device=self._device, batch_size=torch.Size([num_envs]))
        self._make_spec()

    # ------------------------------------------------------------------ setup
    def _capture_cuda_graph(self):
        mjw.step(self._mjw_model, self._mjw_data)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(self._frame_skip):
                mjw.step(self._mjw_model, self._mjw_data)
        self._graph = capture.graph

    def _make_spec(self, td_params=None):
        d = self._device
        self.observation_spec = Composite(
            observation=Unbounded(shape=(self._num_envs, self._obs_dim), dtype=torch.float32, device=d),
            shape=(self._num_envs,),
        )
        self.action_spec = Composite(
            action=Bounded(
                low=-torch.ones(self._act_dim, device=d).expand(self._num_envs, -1),
                high=torch.ones(self._act_dim, device=d).expand(self._num_envs, -1),
                dtype=torch.float32,
                device=d,
            ),
            shape=(self._num_envs,),
        )
        self.reward_spec = Unbounded(shape=(self._num_envs, 1), dtype=torch.float32, device=d)
        self.done_spec = Composite(
            done=Unbounded(shape=(self._num_envs, 1), dtype=torch.bool, device=d),
            terminated=Unbounded(shape=(self._num_envs, 1), dtype=torch.bool, device=d),
            truncated=Unbounded(shape=(self._num_envs, 1), dtype=torch.bool, device=d),
            shape=(self._num_envs,),
        )

    # -------------------------------------------------------------- utilities
    def _get_state_tensors(self):
        qpos = wp.to_torch(self._mjw_data.qpos)   # [N, nq]
        qvel = wp.to_torch(self._mjw_data.qvel)   # [N, nv]
        return qpos, qvel

    def _write_ctrl(self, actions: torch.Tensor):
        target = self._home_joint.unsqueeze(0) + self.robot.action_scale * actions
        ctrl = wp.to_torch(self._mjw_data.ctrl)   # [N, nu]
        ctrl[:, self._act_cols] = target

    def _reset_envs(self, mask: torch.Tensor):
        """Reset the masked envs to the home pose (+ small noise)."""
        if not bool(mask.any()):
            return
        qpos, qvel = self._get_state_tensors()
        new_qpos = self._home_qpos.unsqueeze(0).expand(self._num_envs, -1).clone()
        if self._reset_pos_noise > 0:
            new_qpos[:, self._base_qadr:self._base_qadr + 2] += (
                torch.randn(self._num_envs, 2, device=self._device) * self._reset_pos_noise
            )
        if self._reset_joint_noise > 0:
            new_qpos[:, self._joint_qadr] += (
                torch.randn(self._num_envs, self._act_dim, device=self._device)
                * self._reset_joint_noise
            )
        qpos[mask] = new_qpos[mask]
        qvel[mask] = 0.0
        self.cp_idx[mask] = 0
        self.step_count[mask] = 0
        self.prev_action[mask] = 0.0

    # ------------------------------------------------------------------- step
    def _step(self, td: TensorDictBase) -> TensorDictBase:
        actions = td["action"].clamp(-1.0, 1.0)
        self._write_ctrl(actions)

        wp.capture_launch(self._graph)
        wp.synchronize()
        self.step_count += 1

        st = extract_state(*self._get_state_tensors(), self.info)
        _, cur_dist, _ = checkpoint_geometry(st, self._checkpoints, self.cp_idx)
        new_idx, inter, fin = advance_checkpoints(
            cur_dist, self.cp_idx, self.cfg.checkpoint_radius, self._num_cp
        )
        terminated, truncated, fell, off, outcome = compute_termination(
            st, self.step_count, fin, self.cfg.half_width, self.term_cfg
        )
        reward, _ = compute_reward(
            st, self.prev_dist, cur_dist, inter, fin, fell, off, actions,
            self.robot.nominal_height, self.reward_cfg,
        )

        self.cp_idx = new_idx
        self.prev_action = actions
        done = terminated | truncated

        # Auto-reset finished envs, then recompute obs + prev_dist for everyone.
        self._reset_envs(done)
        st_after = extract_state(*self._get_state_tensors(), self.info)
        to_cp, dist_after, herr = checkpoint_geometry(st_after, self._checkpoints, self.cp_idx)
        self.prev_dist = dist_after
        obs = build_observation(
            st_after, to_cp, dist_after, herr, self.prev_action, self._home_joint
        )

        return TensorDict(
            {
                "observation": obs,
                "reward": reward,
                "done": done.unsqueeze(-1),
                "terminated": terminated.unsqueeze(-1),
                "truncated": truncated.unsqueeze(-1),
                "outcome": outcome,
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _reset(self, td: TensorDictBase = None, **kwargs) -> TensorDictBase:
        if td is not None and "_reset" in td.keys():
            mask = td["_reset"].squeeze(-1)
        else:
            mask = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._reset_envs(mask)

        st = extract_state(*self._get_state_tensors(), self.info)
        to_cp, dist, herr = checkpoint_geometry(st, self._checkpoints, self.cp_idx)
        self.prev_dist = dist
        obs = build_observation(st, to_cp, dist, herr, self.prev_action, self._home_joint)

        return TensorDict(
            {
                "observation": obs,
                "done": torch.zeros(self._num_envs, 1, dtype=torch.bool, device=self._device),
                "terminated": torch.zeros(self._num_envs, 1, dtype=torch.bool, device=self._device),
                "truncated": torch.zeros(self._num_envs, 1, dtype=torch.bool, device=self._device),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)
