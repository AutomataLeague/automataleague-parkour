"""Programmatic parkour scene: a polyline track + attached robot.

The track (a centerline polyline from ``tracks.py``) is painted onto the floor as
offset lane boundaries and perpendicular gate stripes, and the robot is grafted on
with ``MjSpec.attach`` at the spawn frame. Straight and curved tracks use the same
code — a curve is just a dense polyline.

``build_parkour_model`` returns a compiled ``MjModel`` plus a ``SceneInfo`` with the
indices, home pose, and centerline that a task/renderer needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from automataleague_parkour.envs.parkour.config import ParkourConfig
from automataleague_parkour.envs.parkour.obstacles import add_obstacles
from automataleague_parkour.envs.parkour.tracks import Track
from automataleague_parkour.robots import RobotSpec, get_robot

# Restrained palette (single warm accent = yellow checkpoints; calm green finish).
_YELLOW = [0.95, 0.82, 0.20, 1.0]       # checkpoint stripes
_GREEN = [0.20, 0.70, 0.35, 1.0]        # finish line
_START = [0.85, 0.85, 0.85, 1.0]        # start line
_LANE = [0.90, 0.90, 0.92, 1.0]         # path boundary lines painted on the floor

# Painted markers are flat decals: near-zero height, just above the floor.
_PAINT_HALF_Z = 0.001
_PAINT_Z = 0.0015
_STRIPE_HALF = 0.05  # half-thickness of start/finish/checkpoint stripes (~10 cm)


@dataclass
class SceneInfo:
    """Resolved handles into a compiled parkour model."""

    robot: RobotSpec
    prefix: str
    checkpoints_xy: np.ndarray   # (K, 2) gate waypoints, ending at the finish point
    checkpoint_radius: float
    centerline: np.ndarray       # (M, 2) path centerline (for off-path distance)
    spawn_xy: tuple[float, float]
    base_body_id: int
    actuator_ids: np.ndarray     # (n_joints,) in robot.joint_names order
    joint_qposadr: np.ndarray    # (n_joints,) qpos address of each actuated joint
    joint_dofadr: np.ndarray     # (n_joints,) dof address of each actuated joint
    base_qposadr: int            # qpos address of the free joint (base pose)
    base_dofadr: int             # dof address of the free joint (base velocity)
    home_qpos: np.ndarray        # full model qpos for the home stance at spawn
    obstacle_dr: dict | None = None   # per-world mocap obstacle DR (see _resolve_dr)
    foot_geom_ids: np.ndarray = field(  # (n_feet,) geom ids of the robot's feet (gait rewards)
        default_factory=lambda: np.array([], dtype=np.int64))


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _seg_yaw(a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    return math.atan2(d[1], d[0])


def _tangent_yaw_at(centerline: np.ndarray, p: np.ndarray) -> float:
    """Yaw of the centerline segment nearest to point p."""
    a, b = centerline[:-1], centerline[1:]
    ab = b - a
    ab2 = np.clip((ab * ab).sum(1), 1e-9, None)
    t = np.clip(((p - a) * ab).sum(1) / ab2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    i = int(np.linalg.norm(p - proj, axis=1).argmin())
    return _seg_yaw(a[i], b[i])


def _add_checker_floor(spec: mujoco.MjSpec, centerline: np.ndarray, margin: float) -> None:
    tex = spec.add_texture()
    tex.name = "grid"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.width = 300
    tex.height = 300
    tex.nchannel = 3
    tex.rgb1 = [0.20, 0.20, 0.22]
    tex.rgb2 = [0.28, 0.28, 0.30]

    mat = spec.add_material()
    mat.name = "grid"
    mat.texrepeat = [2, 2]
    mat.reflectance = 0.1
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"

    lo = centerline.min(0) - margin
    hi = centerline.max(0) + margin
    center = (lo + hi) / 2
    half = (hi - lo) / 2
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[float(half[0]), float(half[1]), 0.1],
        pos=[float(center[0]), float(center[1]), 0.0], material="grid",
    )


def _stripe(spec, name, pos, yaw, half_x, half_y, rgba):
    """A flat rotated box decal on the floor (local x along `yaw`)."""
    spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half_x, half_y, _PAINT_HALF_Z],
        pos=[float(pos[0]), float(pos[1]), _PAINT_Z],
        quat=_yaw_quat(yaw), rgba=rgba, contype=0, conaffinity=0,
    )


def _offset_polyline(cl: np.ndarray, dist: float) -> np.ndarray:
    """Offset a polyline sideways by `dist` (left = +) using per-vertex normals."""
    tang = np.zeros_like(cl)
    tang[1:-1] = cl[2:] - cl[:-2]
    tang[0] = cl[1] - cl[0]
    tang[-1] = cl[-1] - cl[-2]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)   # left perpendicular
    return cl + dist * normal


def _add_track_paint(spec: mujoco.MjSpec, track: Track, cfg: ParkourConfig,
                     checkpoints: np.ndarray) -> None:
    hw, lt = cfg.half_width, cfg.line_thickness
    cl = track.centerline

    # Lane boundary lines: draw each boundary as its own contiguous offset polyline
    # (no gaps on curves, unlike per-centerline-segment offsetting).
    for sign, tag in ((+1, "l"), (-1, "r")):
        off = _offset_polyline(cl, sign * hw)
        for i in range(len(off) - 1):
            a, b = off[i], off[i + 1]
            seg_len = float(np.linalg.norm(b - a))
            if seg_len < 1e-6:
                continue
            _stripe(spec, f"lane_{tag}_{i}", (a + b) / 2, _seg_yaw(a, b),
                    seg_len / 2, lt / 2, _LANE)

    # Start line (at centerline start) and finish line (at the last centerline vertex).
    _stripe(spec, "start_line", cl[0], _seg_yaw(cl[0], cl[1]), lt / 2, hw, _START)
    _stripe(spec, "finish_line", cl[-1], _seg_yaw(cl[-2], cl[-1]), lt / 2, hw, _GREEN)

    # Checkpoint gates (all but the last checkpoint, which is the finish success point).
    for j, cp in enumerate(checkpoints[:-1]):
        yaw = _tangent_yaw_at(cl, cp)
        _stripe(spec, f"checkpoint_{j + 1}", cp, yaw, lt / 2, hw, _YELLOW)


def _add_lights(spec: mujoco.MjSpec, centerline: np.ndarray) -> None:
    c = centerline.mean(0)
    spec.worldbody.add_light(
        pos=[float(c[0]), float(c[1]), 5.0], dir=[0, 0, -1],
        diffuse=[0.8, 0.8, 0.8], specular=[0.2, 0.2, 0.2], castshadow=True,
    )
    spec.worldbody.add_light(
        pos=[float(c[0]), float(c[1]) + 2.0, 4.0], dir=[0, -0.4, -1],
        diffuse=[0.4, 0.4, 0.4], castshadow=False,
    )


def _resolve_dr(model: mujoco.MjModel, dr_raw: list) -> dict | None:
    """Flatten obstacle DR descriptors into per-world arrays keyed by mocap index.

    One sampled factor per obstacle (group) scales all its bodies. Two body modes:
      * height — box top set via mocap_pos.z = h_nom * factor - H
      * angle  — slab tilt set via mocap_quat at angle = base_angle * factor, with z
                 recomputed for ground/top contact (cz_mode 0/1).
    """
    if not dr_raw:
        return None

    def mocapid(name):
        return int(model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)])

    H = {"mocap_idx": [], "h_nom": [], "group": []}
    A = {"mocap_idx": [], "group": [], "base_angle": [], "axis": [], "yaw": [],
         "half": [], "cz_mode": [], "top_z": []}
    for gi, obst in enumerate(dr_raw):
        for b in obst["bodies"]:
            if b["mode"] == "height":
                H["mocap_idx"].append(mocapid(b["body"]))
                H["h_nom"].append(b["h_nom"]); H["group"].append(gi)
            else:
                A["mocap_idx"].append(mocapid(b["body"])); A["group"].append(gi)
                A["base_angle"].append(b["base_angle"]); A["axis"].append(b["axis"])
                A["yaw"].append(b["yaw"]); A["half"].append(b["half"])
                A["cz_mode"].append(b["cz_mode"]); A["top_z"].append(b["top_z"])

    out = {"n_groups": len(dr_raw), "H": float(_DR_H_FROM(dr_raw)),
           "names": [obst["name"] for obst in dr_raw]}
    out["height"] = None if not H["mocap_idx"] else dict(
        mocap_idx=np.array(H["mocap_idx"], np.int64),
        h_nom=np.array(H["h_nom"], np.float32), group=np.array(H["group"], np.int64))
    out["angle"] = None if not A["mocap_idx"] else dict(
        mocap_idx=np.array(A["mocap_idx"], np.int64), group=np.array(A["group"], np.int64),
        base_angle=np.array(A["base_angle"], np.float32), axis=np.array(A["axis"], np.float32),
        yaw=np.array(A["yaw"], np.float32), half=np.array(A["half"], np.float32),
        cz_mode=np.array(A["cz_mode"], np.int64), top_z=np.array(A["top_z"], np.float32))
    return out


def _DR_H_FROM(dr_raw):
    for obst in dr_raw:
        for b in obst["bodies"]:
            if b["mode"] == "height":
                return b["H"]
    return 0.8


def build_parkour_model(
    robot: str | RobotSpec = "spot",
    cfg: ParkourConfig | None = None,
) -> tuple[mujoco.MjModel, SceneInfo]:
    """Compose track + robot into a compiled ``MjModel`` and a ``SceneInfo``."""
    cfg = cfg or ParkourConfig()
    robot_spec = robot if isinstance(robot, RobotSpec) else get_robot(robot)
    track = cfg.build_track()
    checkpoints = cfg.checkpoints_xy()

    spec = mujoco.MjSpec()
    spec.modelname = f"parkour_{robot_spec.name}_{track.name}"
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    # Brighter ambient so terrain/obstacles read clearly (the scene was quite dark).
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

    _add_checker_floor(spec, track.centerline, cfg.half_width + 1.5)
    _add_track_paint(spec, track, cfg, checkpoints)
    dr_raw = add_obstacles(spec, track.centerline, cfg)   # Stage-1 physical terrain
    _add_lights(spec, track.centerline)

    prefix = f"{robot_spec.name}/"
    sx, sy = track.spawn_xy
    frame = spec.worldbody.add_frame(pos=[sx, sy, 0.0])
    spec.attach(robot_spec.load_spec(), prefix=prefix, frame=frame)

    model = spec.compile()
    info = _resolve_scene_info(model, robot_spec, cfg, track, checkpoints, prefix)
    info.obstacle_dr = _resolve_dr(model, dr_raw)
    return model, info


def build_race_model(
    robot: str | RobotSpec = "spot", cfg: ParkourConfig | None = None,
    n: int = 2, lateral_gap: float = 0.6, robots=None,
) -> tuple[mujoco.MjModel, list[SceneInfo]]:
    """Same track, but with several robots attached side-by-side on the start line.

    Pass ``robots`` (a list of names/RobotSpecs) to race *different* robots against
    each other (e.g. ["spot", "go1"]); otherwise ``n`` copies of ``robot`` are placed.
    Returns the compiled model and one SceneInfo per robot (each with its own
    base/joint/actuator addresses; the checkpoints/centerline are shared).
    """
    cfg = cfg or ParkourConfig()
    if robots is not None:
        specs = [r if isinstance(r, RobotSpec) else get_robot(r) for r in robots]
    else:
        rs = robot if isinstance(robot, RobotSpec) else get_robot(robot)
        specs = [rs] * n
    track = cfg.build_track()
    checkpoints = cfg.checkpoints_xy()

    spec = mujoco.MjSpec()
    spec.modelname = f"race_{'_'.join(s.name for s in specs)}_{track.name}"
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

    _add_checker_floor(spec, track.centerline, cfg.half_width + 1.5)
    _add_track_paint(spec, track, cfg, checkpoints)
    add_obstacles(spec, track.centerline, cfg)
    _add_lights(spec, track.centerline)

    sx, sy = track.spawn_xy
    yaw = track.spawn_heading
    lat = (-math.sin(yaw), math.cos(yaw))          # perpendicular to the start heading
    placed = []
    for i, rspec in enumerate(specs):
        off = (i - (len(specs) - 1) / 2) * lateral_gap
        px, py = sx + lat[0] * off, sy + lat[1] * off
        prefix = f"{rspec.name}{i}/"
        frame = spec.worldbody.add_frame(pos=[px, py, 0.0])
        spec.attach(rspec.load_spec(), prefix=prefix, frame=frame)
        placed.append((prefix, rspec, (px, py)))

    model = spec.compile()
    home = mujoco.MjData(model).qpos.copy().astype(np.float32)
    infos = []
    for prefix, rspec, (px, py) in placed:
        info = _resolve_scene_info(model, rspec, cfg, track, checkpoints,
                                   prefix, spawn=(px, py))
        home[info.base_qposadr:info.base_qposadr + 3] = [px, py, rspec.nominal_height]
        home[info.base_qposadr + 3:info.base_qposadr + 7] = _yaw_quat(yaw)
        home[info.joint_qposadr] = rspec.home_joint_qpos
        infos.append(info)
    for info in infos:
        info.home_qpos = home                      # shared combined reset pose
    return model, infos


def _resolve_scene_info(
    model: mujoco.MjModel, robot: RobotSpec, cfg: ParkourConfig, track: Track,
    checkpoints: np.ndarray, prefix: str, spawn: tuple[float, float] | None = None,
) -> SceneInfo:
    def jid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + robot.base_body)
    base_jnt = int(model.body_jntadr[base_body_id])
    base_qposadr = int(model.jnt_qposadr[base_jnt])
    base_dofadr = int(model.jnt_dofadr[base_jnt])

    joint_ids = np.array([jid(n) for n in robot.joint_names], dtype=np.int64)
    joint_qposadr = model.jnt_qposadr[joint_ids].astype(np.int64)
    joint_dofadr = model.jnt_dofadr[joint_ids].astype(np.int64)
    foot_geom_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + g)
         for g in robot.foot_geom_names], dtype=np.int64)
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + a)
         for a in robot.actuator_names],
        dtype=np.int64,
    )

    sx, sy = spawn if spawn is not None else track.spawn_xy
    home = mujoco.MjData(model).qpos.copy().astype(np.float32)
    home[base_qposadr:base_qposadr + 3] = [sx, sy, robot.nominal_height]
    home[base_qposadr + 3:base_qposadr + 7] = _yaw_quat(track.spawn_heading)
    home[joint_qposadr] = robot.home_joint_qpos

    return SceneInfo(
        robot=robot,
        prefix=prefix,
        checkpoints_xy=checkpoints,
        checkpoint_radius=cfg.checkpoint_radius,
        centerline=track.centerline.astype(np.float32),
        spawn_xy=(sx, sy),
        base_body_id=base_body_id,
        actuator_ids=actuator_ids,
        joint_qposadr=joint_qposadr,
        joint_dofadr=joint_dofadr,
        base_qposadr=base_qposadr,
        base_dofadr=base_dofadr,
        home_qpos=home,
        foot_geom_ids=foot_geom_ids,
    )


if __name__ == "__main__":
    import sys
    track_name = sys.argv[1] if len(sys.argv) > 1 else "straight"
    m, info = build_parkour_model("spot", ParkourConfig(track=track_name))
    print(f"track={track_name} nq={m.nq} nu={m.nu} ngeom={m.ngeom}")
    print(f"checkpoints:\n{info.checkpoints_xy}")
    print(f"centerline:\n{info.centerline}")
