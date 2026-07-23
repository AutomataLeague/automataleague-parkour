"""Programmatic parkour scene: corridor + checkpoints + attached robot.

The corridor is built as an ``MjSpec`` from ``ParkourConfig`` and the chosen
robot is grafted on with ``MjSpec.attach`` at the spawn frame. Building the
corridor in code (rather than a static XML) keeps checkpoint coordinates a single
source of truth shared with the reward.

``build_parkour_model`` returns a compiled ``MjModel`` plus a ``SceneInfo`` with
the indices and home pose a task/renderer needs — so downstream code never has to
reason about name prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from automataleague.envs.parkour.config import ParkourConfig
from automataleague.robots import RobotSpec, get_robot

# Restrained palette (single warm accent = yellow checkpoints; calm green finish).
_YELLOW = [0.95, 0.82, 0.20, 1.0]       # checkpoint stripes
_GREEN = [0.20, 0.70, 0.35, 1.0]        # finish line
_START = [0.85, 0.85, 0.85, 1.0]        # start line
_LANE = [0.90, 0.90, 0.92, 1.0]         # path boundary lines painted on the floor

# Painted markers are essentially flat decals: near-zero height, sitting just
# above the floor so they read as tape, not raised curbs.
_PAINT_HALF_Z = 0.001
_PAINT_Z = 0.0015
_STRIPE_HALF_X = 0.05  # half-width of start/finish/checkpoint stripes (~10 cm)


@dataclass
class SceneInfo:
    """Resolved handles into a compiled parkour model."""

    robot: RobotSpec
    prefix: str
    checkpoints_xy: np.ndarray   # (K, 2) checkpoint centres, cp1..finish
    checkpoint_radius: float
    spawn_xy: tuple[float, float]
    base_body_id: int
    actuator_ids: np.ndarray     # (n_joints,) in robot.joint_names order
    joint_qposadr: np.ndarray    # (n_joints,) qpos address of each actuated joint
    joint_dofadr: np.ndarray     # (n_joints,) dof address of each actuated joint
    base_qposadr: int            # qpos address of the free joint (base pose)
    base_dofadr: int             # dof address of the free joint (base velocity)
    home_qpos: np.ndarray        # full model qpos for the home stance at spawn


def _add_checker_floor(spec: mujoco.MjSpec, cfg: ParkourConfig) -> None:
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
    mat.texrepeat = [int(max(2, cfg.length)), int(max(2, 2 * cfg.half_width))]
    mat.reflectance = 0.1
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"

    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[cfg.length / 2 + 1.5, cfg.half_width + 1.5, 0.1],
        pos=[cfg.length / 2, 0.0, 0.0],
        material="grid",
    )


def _add_corridor(spec: mujoco.MjSpec, cfg: ParkourConfig) -> None:
    """Paint the path onto the floor: two boundary lines + start/finish stripes.

    All markers are non-colliding (contype/conaffinity = 0) — the path is enforced
    by reward/termination (off-path = terminate + penalty), not by physical walls.
    """
    hw, L = cfg.half_width, cfg.length
    lt = cfg.line_thickness
    lane_end = L + cfg.finish_offset   # lanes continue to the success point past the line

    # Path boundary lines at y = ±half_width, running the length of the corridor.
    for sign, tag in ((+1, "left"), (-1, "right")):
        spec.worldbody.add_geom(
            name=f"lane_{tag}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[lane_end / 2, lt / 2, _PAINT_HALF_Z],
            pos=[lane_end / 2, sign * hw, _PAINT_Z],
            rgba=_LANE, contype=0, conaffinity=0,
        )

    # Start line (x=0) and finish line (x=L), stripes across the path.
    spec.worldbody.add_geom(
        name="start_line", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[_STRIPE_HALF_X, hw, _PAINT_HALF_Z], pos=[0.0, 0.0, _PAINT_Z],
        rgba=_START, contype=0, conaffinity=0,
    )
    spec.worldbody.add_geom(
        name="finish_line", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[_STRIPE_HALF_X, hw, _PAINT_HALF_Z], pos=[L, 0.0, _PAINT_Z],
        rgba=_GREEN, contype=0, conaffinity=0,
    )


def _add_checkpoints(spec: mujoco.MjSpec, cfg: ParkourConfig) -> None:
    hw = cfg.half_width
    cps = cfg.checkpoints_xy()
    # Intermediate checkpoints as yellow stripes across the path (gates to cross);
    # the final checkpoint is the green finish line.
    for i, (cx, _cy) in enumerate(cps[:-1]):
        spec.worldbody.add_geom(
            name=f"checkpoint_{i + 1}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[_STRIPE_HALF_X, hw, _PAINT_HALF_Z],
            pos=[float(cx), 0.0, _PAINT_Z],
            rgba=_YELLOW, contype=0, conaffinity=0,
        )


def _add_lights(spec: mujoco.MjSpec, cfg: ParkourConfig) -> None:
    spec.worldbody.add_light(
        pos=[cfg.length / 2, 0.0, 4.0], dir=[0, 0, -1],
        diffuse=[0.8, 0.8, 0.8], specular=[0.2, 0.2, 0.2], castshadow=True,
    )
    spec.worldbody.add_light(
        pos=[cfg.length * 0.25, 2.0, 3.0], dir=[0, -0.4, -1],
        diffuse=[0.4, 0.4, 0.4], castshadow=False,
    )


def build_parkour_model(
    robot: str | RobotSpec = "spot",
    cfg: ParkourConfig | None = None,
) -> tuple[mujoco.MjModel, SceneInfo]:
    """Compose corridor + robot into a compiled ``MjModel`` and a ``SceneInfo``."""
    cfg = cfg or ParkourConfig()
    robot_spec = robot if isinstance(robot, RobotSpec) else get_robot(robot)

    spec = mujoco.MjSpec()
    spec.modelname = f"parkour_{robot_spec.name}"

    # Physics options aligned with the Spot reference model.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0

    # Larger offscreen framebuffer so we can render HD screenshots/videos.
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    _add_checker_floor(spec, cfg)
    _add_corridor(spec, cfg)
    _add_checkpoints(spec, cfg)
    _add_lights(spec, cfg)

    # Attach the robot at the spawn frame.
    prefix = f"{robot_spec.name}/"
    frame = spec.worldbody.add_frame(pos=[cfg.spawn_x, cfg.spawn_y, 0.0])
    spec.attach(robot_spec.load_spec(), prefix=prefix, frame=frame)

    model = spec.compile()
    info = _resolve_scene_info(model, robot_spec, cfg, prefix)
    return model, info


def _resolve_scene_info(
    model: mujoco.MjModel, robot: RobotSpec, cfg: ParkourConfig, prefix: str
) -> SceneInfo:
    def jid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + robot.base_body)
    base_jnt = int(model.body_jntadr[base_body_id])   # free joint is the base's first joint
    base_qposadr = int(model.jnt_qposadr[base_jnt])
    base_dofadr = int(model.jnt_dofadr[base_jnt])

    joint_ids = np.array([jid(n) for n in robot.joint_names], dtype=np.int64)
    joint_qposadr = model.jnt_qposadr[joint_ids].astype(np.int64)
    joint_dofadr = model.jnt_dofadr[joint_ids].astype(np.int64)
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + a)
         for a in robot.actuator_names],
        dtype=np.int64,
    )

    # Home pose: default qpos with base placed at spawn/nominal height and legs at stance.
    home = mujoco.MjData(model).qpos.copy().astype(np.float32)
    home[base_qposadr:base_qposadr + 3] = [cfg.spawn_x, cfg.spawn_y, robot.nominal_height]
    home[base_qposadr + 3:base_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
    home[joint_qposadr] = robot.home_joint_qpos

    return SceneInfo(
        robot=robot,
        prefix=prefix,
        checkpoints_xy=cfg.checkpoints_xy(),
        checkpoint_radius=cfg.checkpoint_radius,
        spawn_xy=(cfg.spawn_x, cfg.spawn_y),
        base_body_id=base_body_id,
        actuator_ids=actuator_ids,
        joint_qposadr=joint_qposadr,
        joint_dofadr=joint_dofadr,
        base_qposadr=base_qposadr,
        base_dofadr=base_dofadr,
        home_qpos=home,
    )


if __name__ == "__main__":
    m, info = build_parkour_model("spot")
    print(f"Compiled parkour model: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
    print(f"Checkpoints (xy):\n{info.checkpoints_xy}")
    print(f"Home base pose: {info.home_qpos[info.base_qposadr:info.base_qposadr + 7]}")
