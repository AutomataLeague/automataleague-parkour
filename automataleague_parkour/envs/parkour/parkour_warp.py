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
from automataleague_parkour.envs.parkour.observation import build_observation
from automataleague_parkour.envs.parkour.rewards import compute_reward
from automataleague_parkour.envs.parkour.scene import build_parkour_model
from automataleague_parkour.envs.parkour.state import extract_state
from automataleague_parkour.envs.parkour.termination import compute_termination


def _qmul_b(a, b):
    """Batched Hamilton product of wxyz quaternions (broadcasting trailing dim 4)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def _qrot_b(q, v):
    """Batched rotation of vector v[...,3] by wxyz quaternion q[...,4]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return torch.stack([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx)], dim=-1)


class ParkourEnvWarp(EnvBase):
    """Batched GPU parkour environment. All N envs step simultaneously."""

    def __init__(
        self,
        robot: str = "spot",
        num_envs: int = 1024,
        device: str = "cuda",
        cfg: ParkourConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        reward_fn=None,   # custom reward fn (same signature as rewards.compute_reward);
                          # None -> the default compute_reward. Lets a robot bring its own reward.
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
        self._reward_fn = reward_fn or compute_reward
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
        self._scan_on = bool(self.cfg.height_scan)
        self._obs_dim = self.robot.obs_dim + (hs.SCAN_N if self._scan_on else 0)
        self._act_dim = self.robot.action_dim
        self._action_scale = (self.cfg.action_scale if self.cfg.action_scale is not None
                              else self.robot.action_scale)

        # Device tensors used every step.
        d = self._device
        self._checkpoints = torch.tensor(self.info.checkpoints_xy, dtype=torch.float32, device=d)
        self._num_cp = self._checkpoints.shape[0]
        self._centerline = torch.tensor(self.info.centerline, dtype=torch.float32, device=d)
        self._race_mode = bool(self.cfg.race_mode)
        # gate tangents (track direction at each checkpoint) for race-mode gate crossing
        self._cp_tangent = (centerline_frame(self._checkpoints, self._centerline)[1]
                            if self._race_mode else None)
        self._home_joint = torch.tensor(self.robot.home_joint_qpos, dtype=torch.float32, device=d)
        self._home_qpos = torch.tensor(self.info.home_qpos, dtype=torch.float32, device=d)
        self._act_cols = torch.tensor(self.info.actuator_ids, dtype=torch.long, device=d)
        self._base_qadr = self.info.base_qposadr
        self._joint_qadr = torch.tensor(self.info.joint_qposadr, dtype=torch.long, device=d)

        if self._scan_on:
            self._setup_height_scan()

        self._dr = None
        if self.cfg.randomize_obstacles and self.info.obstacle_dr:
            self._setup_obstacle_dr()

        # Per-env task state.
        self.cp_idx = torch.zeros(num_envs, dtype=torch.long, device=d)
        self.prev_dist = torch.zeros(num_envs, device=d)
        self.prev_action = torch.zeros(num_envs, self._act_dim, device=d)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=d)

        # Foot air-time gait reward (see reward_cfg.feet_air_time). Off unless enabled.
        fids = self.info.foot_geom_ids
        if len(fids) and self.reward_cfg.feet_air_time > 0:
            self._foot_geom_ids = torch.tensor(fids, dtype=torch.long, device=d)
            foot_r = torch.tensor(self._mjm.geom_size[fids, 0], dtype=torch.float32, device=d)
            self._foot_contact_h = foot_r + 0.03          # geom-center z below this = grounded
            self._feet_air_time = torch.zeros(num_envs, len(fids), device=d)
            self._ctrl_dt = self._frame_skip * float(self._mjm.opt.timestep)
        else:
            self._foot_geom_ids = None

        self._graph = None
        self._capture_cuda_graph()

        super().__init__(device=self._device, batch_size=torch.Size([num_envs]))
        self._make_spec()

    # ------------------------------------------------------------------ setup
    def _setup_height_scan(self):
        """Preallocate the batched raycast buffers for the terrain height scan.

        pnt/vec are torch buffers (updated in place each step) wrapped as zero-copy
        Warp vec3 arrays [N, K]; mjw.rays fills dist/geomid/normal [N, K]. The ray
        group mask includes only geom group 0 (floor + obstacles), so rays pass
        through the robot's own geoms (groups 2-3).
        """
        from mujoco_warp._src.types import vec6f

        d, N, K = self._device, self._num_envs, hs.SCAN_N
        self._scan_offsets_t = hs.scan_offsets_torch(d)                 # [K, 2]
        self._scan_pnt = torch.zeros(N, K, 3, device=d)
        self._scan_pnt[:, :, 2] = hs._RAY_H                            # cast from above
        self._scan_vec = torch.zeros(N, K, 3, device=d)
        self._scan_vec[:, :, 2] = -1.0                                # straight down
        self._scan_pnt_wp = wp.from_torch(self._scan_pnt, dtype=wp.vec3f)
        self._scan_vec_wp = wp.from_torch(self._scan_vec, dtype=wp.vec3f)
        self._scan_dist = wp.zeros((N, K), dtype=wp.float32, device=str(d))
        self._scan_geomid = wp.zeros((N, K), dtype=wp.int32, device=str(d))
        self._scan_normal = wp.zeros((N, K), dtype=wp.vec3f, device=str(d))
        self._scan_group = vec6f(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)         # group 0 only
        self._scan_bodyexclude = wp.full((N,), -1, dtype=wp.int32, device=str(d))

    def _setup_obstacle_dr(self):
        """Cache tensors for per-env obstacle domain randomization (height + angle)."""
        dr, d = self.info.obstacle_dr, self._device
        self._dr = dict(n_groups=int(dr["n_groups"]), H=float(dr["H"]),
                        low=float(self.cfg.dr_low), high=float(self.cfg.dr_high),
                        height=None, angle=None)
        if dr["height"] is not None:
            h = dr["height"]
            self._dr["height"] = dict(
                idx=torch.tensor(h["mocap_idx"], dtype=torch.long, device=d),
                h_nom=torch.tensor(h["h_nom"], dtype=torch.float32, device=d),
                group=torch.tensor(h["group"], dtype=torch.long, device=d))
        if dr["angle"] is not None:
            a = dr["angle"]
            self._dr["angle"] = dict(
                idx=torch.tensor(a["mocap_idx"], dtype=torch.long, device=d),
                group=torch.tensor(a["group"], dtype=torch.long, device=d),
                base_angle=torch.deg2rad(torch.tensor(a["base_angle"], device=d)),
                axis=torch.tensor(a["axis"], dtype=torch.float32, device=d),
                yaw=torch.tensor(a["yaw"], dtype=torch.float32, device=d),
                half=torch.tensor(a["half"], dtype=torch.float32, device=d),
                cz_mode=torch.tensor(a["cz_mode"], dtype=torch.bool, device=d),
                top_z=torch.tensor(a["top_z"], dtype=torch.float32, device=d))

    def _angle_pose(self, a, ang):
        """Batched mocap pose for angle-DR slabs: quats [N,B,4] + contact z [N,B]."""
        N, B = ang.shape
        ha = ang * 0.5
        s, c = torch.sin(ha), torch.cos(ha)
        axq = torch.cat([c.unsqueeze(-1), a["axis"].unsqueeze(0) * s.unsqueeze(-1)], dim=-1)
        yh = a["yaw"] * 0.5
        zeros = torch.zeros_like(yh)
        yq = torch.stack([torch.cos(yh), zeros, zeros, torch.sin(yh)], dim=-1)  # [B,4]
        q = _qmul_b(yq.unsqueeze(0).expand(N, B, 4), axq)                       # [N,B,4]
        hx, hy, hz = a["half"][:, 0], a["half"][:, 1], a["half"][:, 2]
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                v = torch.stack([sx * hx, sy * hy, hz], -1).unsqueeze(0).expand(N, B, 3)
                corners.append(_qrot_b(q, v)[..., 2])
        cz_ground = -torch.stack(corners, -1).min(-1).values                   # low corner on floor
        vtop = torch.stack([torch.zeros_like(hz), torch.zeros_like(hz), hz], -1)
        cz_top = a["top_z"].unsqueeze(0) - _qrot_b(q, vtop.unsqueeze(0).expand(N, B, 3))[..., 2]
        cz = torch.where(a["cz_mode"].unsqueeze(0), cz_top, cz_ground)
        return q, cz

    def _randomize_obstacles(self, mask):
        """Draw a per-env difficulty factor per obstacle and write mocap pose.

        One factor ~ U(low, high) per obstacle (shared across its bodies); only reset
        (masked) envs are updated, so each env keeps its difficulty for the episode.
        Height bodies move mocap_pos.z; angle bodies rewrite mocap_quat + contact z.
        """
        dr, N, d = self._dr, self._num_envs, self._device
        f = torch.empty(N, dr["n_groups"], device=d).uniform_(dr["low"], dr["high"])
        m = mask.unsqueeze(1)
        mp = wp.to_torch(self._mjw_data.mocap_pos)          # [N, nmocap, 3]
        if dr["height"] is not None:
            h = dr["height"]
            z = h["h_nom"] * f[:, h["group"]] - dr["H"]
            mp[:, h["idx"], 2] = torch.where(m, z, mp[:, h["idx"], 2])
        if dr["angle"] is not None:
            a = dr["angle"]
            q, cz = self._angle_pose(a, a["base_angle"].unsqueeze(0) * f[:, a["group"]])
            mq = wp.to_torch(self._mjw_data.mocap_quat)     # [N, nmocap, 4]
            mq[:, a["idx"]] = torch.where(m.unsqueeze(2), q, mq[:, a["idx"]])
            mp[:, a["idx"], 2] = torch.where(m, cz, mp[:, a["idx"], 2])

    def _compute_scan(self, st):
        """Batched forward terrain scan [N, SCAN_N], relative to each robot's feet."""
        base_xy, base_z = st.base_pos[:, :2], st.base_pos[:, 2]
        yaw = hs.yaw_from_quat_torch(st.base_quat)
        query = hs.world_query_points_torch(base_xy, yaw, self._scan_offsets_t)
        self._scan_pnt[:, :, 0:2] = query                              # in place -> wp view
        torch.cuda.synchronize()                                       # torch write visible to warp
        mjw.rays(self._mjw_model, self._mjw_data, self._scan_pnt_wp, self._scan_vec_wp,
                 self._scan_group, True, self._scan_bodyexclude,
                 self._scan_dist, self._scan_geomid, self._scan_normal)
        wp.synchronize()
        dist_t = wp.to_torch(self._scan_dist)                          # [N, K]
        terrain_z = torch.where(dist_t >= 0, hs._RAY_H - dist_t, torch.zeros_like(dist_t))
        return hs.scan_relative_torch(terrain_z, base_z, self.robot.nominal_height)

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
        target = self._home_joint.unsqueeze(0) + self._action_scale * actions
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
        if self._foot_geom_ids is not None:
            self._feet_air_time[mask] = 0.0
        if self._dr is not None:
            self._randomize_obstacles(mask)

    # ------------------------------------------------------------------- step
    def _step(self, td: TensorDictBase) -> TensorDictBase:
        actions = td["action"].clamp(-1.0, 1.0)
        self._write_ctrl(actions)

        wp.capture_launch(self._graph)
        wp.synchronize()
        self.step_count += 1

        st = extract_state(*self._get_state_tensors(), self.info)
        if self._race_mode:
            fwd_vel, lateral, new_idx, inter, fin = race_nav(
                st.base_pos[:, :2], st.base_linvel_world[:, :2], self._centerline,
                self._checkpoints, self._cp_tangent, self.cp_idx, self._num_cp)
            cur_dist = torch.zeros_like(fwd_vel)          # progress term is off in race mode
        else:
            _, cur_dist, _ = checkpoint_geometry(st, self._checkpoints, self.cp_idx)
            fwd_vel = forward_velocity(st, self._checkpoints, self.cp_idx, cur_dist)
            lateral = point_to_polyline_distance(st.base_pos[:, :2], self._centerline)
            new_idx, inter, fin = advance_checkpoints(
                cur_dist, self.cp_idx, self.cfg.checkpoint_radius, self._num_cp
            )
        terminated, truncated, fell, off, outcome = compute_termination(
            st, self.step_count, fin, lateral, self.cfg.half_width, self.term_cfg
        )
        reward, _ = self._reward_fn(
            st, self.prev_dist, cur_dist, inter, fin, fell, off, actions,
            self.robot.nominal_height, self.reward_cfg, forward_vel=fwd_vel,
        )

        # Foot air-time gait shaping: at each foot's touchdown, reward the swing time it
        # just completed (relative to the target), so a clean stepping gait beats a shuffle.
        # Only paid while moving forward, so it can't reward marching in place.
        if self._foot_geom_ids is not None:
            foot_z = wp.to_torch(self._mjw_data.geom_xpos)[:, self._foot_geom_ids, 2]
            contact = foot_z < self._foot_contact_h                     # [N, n_feet]
            first_contact = contact & (self._feet_air_time > 0)
            air_rew = ((self._feet_air_time - self.reward_cfg.feet_air_time_target)
                       * first_contact.float()).sum(-1)                        # [N]
            air_rew = air_rew * (fwd_vel.reshape(-1).clamp(min=0.0) > 0.1).float()
            reward = reward + self.reward_cfg.feet_air_time * air_rew.view_as(reward)
            self._feet_air_time = torch.where(
                contact, torch.zeros_like(self._feet_air_time),
                self._feet_air_time + self._ctrl_dt)

        # Task-progress diagnostics, captured PRE-reset so they reflect how far the
        # episode actually got (distance to the finish line + checkpoints reached).
        finish_xy = self._checkpoints[-1]
        dist_to_finish = torch.linalg.norm(st.base_pos[:, :2] - finish_xy, dim=-1)
        checkpoint_idx = new_idx.to(torch.float32)

        self.cp_idx = new_idx
        self.prev_action = actions
        done = terminated | truncated

        # Auto-reset finished envs, then recompute obs + prev_dist for everyone.
        self._reset_envs(done)
        st_after = extract_state(*self._get_state_tensors(), self.info)
        to_cp, dist_after, herr = checkpoint_geometry(st_after, self._checkpoints, self.cp_idx)
        self.prev_dist = dist_after
        scan = self._compute_scan(st_after) if self._scan_on else None
        obs = build_observation(
            st_after, to_cp, dist_after, herr, self.prev_action, self._home_joint,
            height_scan=scan,
        )

        return TensorDict(
            {
                "observation": obs,
                "reward": reward,
                "done": done.unsqueeze(-1),
                "terminated": terminated.unsqueeze(-1),
                "truncated": truncated.unsqueeze(-1),
                "outcome": outcome,
                "dist_to_finish": dist_to_finish.unsqueeze(-1),
                "checkpoint_idx": checkpoint_idx.unsqueeze(-1),
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
        scan = self._compute_scan(st) if self._scan_on else None
        obs = build_observation(st, to_cp, dist, herr, self.prev_action, self._home_joint,
                                height_scan=scan)

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
