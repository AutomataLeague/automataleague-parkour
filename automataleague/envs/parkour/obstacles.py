"""Stage-1 physical obstacles placed along a track centerline.

Obstacles are collidable geoms (the feet actually contact them) built at positions
along the path, oriented to the local path direction. Magnitudes scale with
``ParkourConfig.level_difficulty`` (0 = flat, 4 = hardest) for a difficulty curriculum.

Feature kinds: "step" (raised platform, up + down), "staircase" (a few steps up then
down), "side_incline" (laterally-tilted crossing), "ramp" (up/down slope along path).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

_OBST = [0.62, 0.60, 0.66, 1.0]    # obstacle colour (light concrete, stands out on dark floor)
_OBST2 = [0.70, 0.68, 0.74, 1.0]   # alternating stair shade


def difficulty_params(level: int) -> dict:
    """Obstacle magnitudes for a difficulty level (0 = flat)."""
    lvl = max(0, int(level))
    return dict(
        paving_h=0.025 * lvl,     # paving-stone height jitter (m): L4 = 0.10
        hurdle_h=0.04 * lvl,      # low step-over bar height (m):   L4 = 0.16
        stair_h=0.028 * lvl,      # per-stair rise (m):             L4 = 0.11
        ramp_deg=5.0 * lvl,       # up/down ramp slope (deg):       L4 = 20
        incline_deg=3.0 * lvl,    # banked-traverse tilt (deg):     L4 = 12 (gentler)
    )


# --- quaternion helpers (w, x, y, z) ------------------------------------------
def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _axis_quat(axis, deg):
    a = math.radians(deg) / 2
    s = math.sin(a)
    return [math.cos(a), axis[0] * s, axis[1] * s, axis[2] * s]


def _yaw_quat(yaw_rad):
    return [math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)]


def _qrot(q, v):
    """Rotate vector v by quaternion q (w,x,y,z)."""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx)]


def _cz_ground(size, q):
    """Centre-z so the lowest corner of the (rotated) top face sits on the floor."""
    hx, hy, hz = size
    zs = [_qrot(q, (sx * hx, sy * hy, hz))[2] for sx in (-1, 1) for sy in (-1, 1)]
    return -min(zs)


def _sample_path(centerline: np.ndarray, s: float):
    """(x, y), yaw of the centerline at arc-length s."""
    seg = np.diff(centerline, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
    t = (s - cum[i]) / max(seglen[i], 1e-9)
    pos = centerline[i] + t * seg[i]
    yaw = math.atan2(seg[i][1], seg[i][0])
    return pos, yaw


def _straight_positions(centerline: np.ndarray, n: int, start_clear: float = 8.0,
                        end_clear: float = 2.5, min_gap: float = 5.0):
    """Arc-length midpoints of straight segments, in LAP ORDER (ascending), spaced.

    A straight is a single long segment (arcs are many short segments). Keeps the
    start (first ``start_clear`` m) and the finish (last ``end_clear`` m) clear, so
    the returned order runs from the first obstacle-straight to the one before the
    finish line.
    """
    seg = np.diff(centerline, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    mids = (cum[:-1] + cum[1:]) / 2
    total = float(cum[-1])
    cands = sorted(
        float(mids[i]) for i in range(len(seglen))
        if seglen[i] > 2.0 and start_clear < mids[i] < total - end_clear
    )
    picked = []
    for m in cands:                               # ascending = lap order
        if all(abs(m - q) > min_gap for q in picked):
            picked.append(m)
        if len(picked) == n:
            break
    return picked


# --- feature builders ---------------------------------------------------------
# Tilted slabs are made THICK and sunk below the floor so there's never an open
# gap the robot could crawl under; the floor plane hides the buried part.
_SLAB_T = 0.6


def _add_paving(spec, name, pos, yaw, hw, p):
    """A field of small uneven paving stones (cobbles) across the path."""
    jitter = p["paving_h"]
    if jitter <= 0:
        return
    cell = 0.35
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    lat = np.array([-math.sin(yaw), math.cos(yaw)])
    n_fwd, n_lat = 12, max(2, int(2 * hw / cell))       # longer paving field
    for i in range(n_fwd):
        for j in range(n_lat):
            fo = (i - (n_fwd - 1) / 2) * cell
            lo = (j - (n_lat - 1) / 2) * cell
            c = np.array(pos) + fwd * fo + lat * lo
            h = 0.03 + ((i * 7 + j * 13) % 5) / 4.0 * jitter    # deterministic jitter
            spec.worldbody.add_geom(
                name=f"{name}_{i}_{j}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[cell / 2 * 0.92, cell / 2 * 0.92, h / 2],
                pos=[float(c[0]), float(c[1]), h / 2], quat=_yaw_quat(yaw),
                rgba=_OBST if (i + j) % 2 else _OBST2,
            )


def _add_hurdle(spec, name, pos, yaw, hw, p):
    """A row of low bars across the path to step over (a simple early obstacle)."""
    h = p["hurdle_h"]
    if h <= 0:
        return
    bar_t = 0.14
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    for k, off in enumerate((-1.05, -0.35, 0.35, 1.05)):   # four hurdles
        c = np.array(pos) + fwd * off
        spec.worldbody.add_geom(
            name=f"{name}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[bar_t / 2, hw, h / 2], pos=[float(c[0]), float(c[1]), h / 2],
            quat=_yaw_quat(yaw), rgba=_OBST,
        )


def _add_staircase(spec, name, pos, yaw, hw, p):
    """Longer staircase: several steps up to a top platform, then back down."""
    h = p["stair_h"]
    if h <= 0:
        return
    tread = 0.5
    heights = [1, 2, 3, 4, 4, 4, 4, 3, 2, 1]           # longer: up, top plateau, down
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    start = np.array(pos) - fwd * (tread * len(heights) / 2)
    for k, mult in enumerate(heights):
        c = start + fwd * (tread * (k + 0.5))
        z = h * mult
        spec.worldbody.add_geom(
            name=f"{name}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tread / 2, hw, z / 2], pos=[float(c[0]), float(c[1]), z / 2],
            quat=_yaw_quat(yaw), rgba=_OBST if k % 2 else _OBST2,
        )


def _add_side_incline(spec, name, pos, yaw, hw, p):
    """A banked traverse: ramp UP -> banked to one side -> banked to the OPPOSITE side
    -> ramp DOWN. An off-camber twisting bridge."""
    ang = p["incline_deg"]
    if ang <= 0:
        return
    lp = 1.8                                   # length of each of the 4 pieces (stretched)
    h = 0.22                                   # raised platform height
    theta = math.degrees(math.atan2(h, lp))    # ramp pitch to reach h over lp
    fwd = np.array([math.cos(yaw), math.sin(yaw)])

    def slab(tag, foff, axis, tilt_deg, top_z, rgba):
        c = np.array(pos) + fwd * foff
        q = _qmul(_yaw_quat(yaw), _axis_quat(axis, tilt_deg))
        size = [lp / 2, hw, _SLAB_T / 2]
        cz = top_z - _qrot(q, (0.0, 0.0, _SLAB_T / 2))[2]   # top-face centre at top_z
        spec.worldbody.add_geom(
            name=f"{name}_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size, pos=[float(c[0]), float(c[1]), cz], quat=q, rgba=rgba,
        )

    slab("up",    -1.5 * lp, [0.0, 1.0, 0.0], -theta, h / 2, _OBST)    # ramp up to h
    slab("bankL", -0.5 * lp, [1.0, 0.0, 0.0], +ang,   h,     _OBST2)   # banked one way
    slab("bankR", +0.5 * lp, [1.0, 0.0, 0.0], -ang,   h,     _OBST)    # banked the other way
    slab("down",  +1.5 * lp, [0.0, 1.0, 0.0], +theta, h / 2, _OBST2)   # ramp down


def _add_ramp(spec, name, pos, yaw, hw, p):
    """A SOLID up-and-over hill: rises from the ground to a peak, then descends."""
    ang = p["ramp_deg"]
    if ang <= 0:
        return
    run = 2.2                                           # horizontal run of each slope
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    half = run / 2
    for sign, tag in ((-1, "up"), (+1, "down")):
        c = np.array(pos) + fwd * (sign * half)
        # pitch about the box's LOCAL lateral axis; up-slope (sign=-1) rises to the peak.
        q = _qmul(_yaw_quat(yaw), _axis_quat([0.0, 1.0, 0.0], sign * ang))
        size = [half, hw, _SLAB_T / 2]
        cz = _cz_ground(size, q)
        spec.worldbody.add_geom(
            name=f"{name}_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size, pos=[float(c[0]), float(c[1]), cz], quat=q, rgba=_OBST,
        )


_BUILDERS = {
    "paving": _add_paving,
    "hurdle": _add_hurdle,
    "staircase": _add_staircase,
    "side_incline": _add_side_incline,
    "ramp": _add_ramp,
}
# In LAP ORDER: gentle features first, the hard banked traverse just before the finish.
_LAYOUT = ["paving", "hurdle", "staircase", "ramp", "side_incline"]


def add_obstacles(spec: mujoco.MjSpec, centerline: np.ndarray, cfg) -> None:
    """Place the features along the track in lap order (banked traverse before finish)."""
    level = getattr(cfg, "level_difficulty", 0)
    if level <= 0:
        return
    p = difficulty_params(level)
    positions = _straight_positions(centerline, len(_LAYOUT))
    total = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    for kind, s in zip(_LAYOUT, positions):
        s = min(s, total - 3.6)                 # keep long features clear of the finish
        pos, yaw = _sample_path(centerline, s)
        _BUILDERS[kind](spec, f"obst_{kind}", pos, yaw, cfg.half_width, p)
