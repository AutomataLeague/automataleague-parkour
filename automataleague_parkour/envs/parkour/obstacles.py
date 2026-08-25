"""Stage-1 physical obstacles placed along a track centerline.

Obstacles are collidable geoms (the feet actually contact them) built at positions
along the path, oriented to the local path direction. Magnitudes scale with
``ParkourConfig.level_difficulty`` (0 = flat, 4 = hardest) for a difficulty curriculum.

Feature kinds (see _LAYOUT): "paving" (uneven cobbles), "hurdle" (low bars to step
over), "staircase" (steps up to a top plateau then down), "ramp" (up-and-over hill),
"side_incline" (banked off-camber traverse). Height/angle obstacles are mocap bodies
so their difficulty can be domain-randomized per episode (see scene.py / parkour envs).
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

# Domain-randomizable obstacles are built on MOCAP bodies: a fixed tall box whose
# exposed height above the floor is set per-env at reset via mocap_pos.z. The box
# half-height is generous so its base always stays below the floor across the DR
# range (max exposed ~= stair_h(L4) * 4 * dr_high ~= 0.67 m < _DR_H).
_DR_H = 0.8


def _mocap_box(spec, body_name, geoms, world_xy, yaw, h_nom):
    """Add a kinematic (mocap) body holding one or more tall boxes (HEIGHT DR).

    The body sits at z = h_nom - _DR_H so that, at the nominal mocap_pos, each box's
    top is at h_nom above the floor. Per-env DR shifts mocap_pos.z at reset. Returns
    the per-body DR record {mode, body, h_nom, H}.
    `geoms` is a list of (name, (local_fwd, local_lat), half_x, half_y, rgba).
    """
    body = spec.worldbody.add_body(name=body_name)
    body.mocap = True
    body.pos = [float(world_xy[0]), float(world_xy[1]), h_nom - _DR_H]
    body.quat = _yaw_quat(yaw)
    for gname, (lf, ll), hx, hy, rgba in geoms:
        body.add_geom(
            name=gname, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, _DR_H],
            pos=[lf, ll, 0.0], rgba=rgba,             # local (x=path-forward, y=lateral)
        )
    return {"mode": "height", "body": body_name, "h_nom": float(h_nom), "H": _DR_H}


def _mocap_slab(spec, body_name, world_xy, yaw, axis, angle_deg, half, cz_mode, top_z, rgba):
    """Add a tilted slab on a mocap body for ANGLE DR.

    The tilt lives in the body's quat (the geom is axis-aligned inside it), so per-env
    DR rewrites mocap_quat at reset and recomputes z for contact. Built at the nominal
    angle, so factor 1 reproduces the current geometry. cz_mode: "ground" sits the
    slab's low edge on the floor (ramps); "top" pins the top face at ``top_z`` (banks).
    """
    q = _qmul(_yaw_quat(yaw), _axis_quat(axis, angle_deg))
    cz = _cz_ground(half, q) if cz_mode == "ground" else top_z - _qrot(q, (0.0, 0.0, half[2]))[2]
    body = spec.worldbody.add_body(name=body_name)
    body.mocap = True
    body.pos = [float(world_xy[0]), float(world_xy[1]), cz]
    body.quat = q
    body.add_geom(name=f"{body_name}_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=list(half), pos=[0.0, 0.0, 0.0], rgba=rgba)
    return {"mode": "angle", "body": body_name, "yaw": float(yaw), "axis": list(axis),
            "base_angle": float(angle_deg), "half": [float(h) for h in half],
            "cz_mode": 0 if cz_mode == "ground" else 1, "top_z": float(top_z)}


def _add_paving(spec, name, pos, yaw, hw, p):
    """A field of small uneven paving stones (cobbles) across the path.

    Stones are grouped by their jitter height into a few mocap bodies (one per level),
    so a single DR factor scales the whole cobble field's roughness coherently.
    """
    jitter = p["paving_h"]
    if jitter <= 0:
        return None
    cell = 0.35
    n_fwd, n_lat = 12, max(2, int(2 * hw / cell))       # longer paving field
    levels = {}                                          # h_nom -> [(name, (fwd, lat), rgba)]
    for i in range(n_fwd):
        for j in range(n_lat):
            fo = (i - (n_fwd - 1) / 2) * cell
            lo = (j - (n_lat - 1) / 2) * cell
            h = 0.03 + ((i * 7 + j * 13) % 5) / 4.0 * jitter    # deterministic jitter
            levels.setdefault(round(h, 5), []).append(
                (f"{name}_{i}_{j}", (fo, lo), _OBST if (i + j) % 2 else _OBST2))
    bodies = []
    for gi, (h_nom, stones) in enumerate(sorted(levels.items())):
        geoms = [(gn, off, cell / 2 * 0.92, cell / 2 * 0.92, rgba) for gn, off, rgba in stones]
        bodies.append(_mocap_box(spec, f"{name}_g{gi}", geoms, pos, yaw, h_nom))
    return {"name": name, "bodies": bodies}


def _add_hurdle(spec, name, pos, yaw, hw, p):
    """A row of low bars across the path to step over (a simple early obstacle).

    All four bars share one mocap body (uniform height), DR-scaled together.
    """
    h = p["hurdle_h"]
    if h <= 0:
        return None
    bar_t = 0.14
    geoms = [(f"{name}_{k}", (off, 0.0), bar_t / 2, hw, _OBST)
             for k, off in enumerate((-1.05, -0.35, 0.35, 1.05))]
    rec = _mocap_box(spec, f"{name}_body", geoms, pos, yaw, h)
    return {"name": name, "bodies": [rec]}


def _add_staircase(spec, name, pos, yaw, hw, p):
    """Longer staircase: several steps up to a top platform, then back down.

    Each step is its own mocap body so a per-env DR factor scales the whole
    step-rise profile coherently (not just truncating the tall steps).
    """
    h = p["stair_h"]
    if h <= 0:
        return None
    tread = 0.5
    heights = [1, 2, 3, 4, 4, 4, 4, 3, 2, 1]           # longer: up, top plateau, down
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    start = np.array(pos) - fwd * (tread * len(heights) / 2)
    bodies = []
    for k, mult in enumerate(heights):
        c = start + fwd * (tread * (k + 0.5))
        geoms = [(f"{name}_{k}", (0.0, 0.0), tread / 2, hw, _OBST if k % 2 else _OBST2)]
        bodies.append(_mocap_box(spec, f"{name}_s{k}", geoms, c, yaw, h * mult))
    return {"name": name, "bodies": bodies}


def _add_side_incline(spec, name, pos, yaw, hw, p):
    """A banked traverse: ramp UP -> banked to one side -> banked to the OPPOSITE side
    -> ramp DOWN. An off-camber twisting bridge.

    The two approach ramps are static (they only carry the robot up to the platform
    height h); the two BANK slabs are angle-DR mocap bodies (pinned at top_z = h), so
    the DR factor scales the off-camber tilt — the difficulty of a banked traverse.
    """
    ang = p["incline_deg"]
    if ang <= 0:
        return None
    lp = 1.8                                   # length of each of the 4 pieces (stretched)
    h = 0.22                                   # raised platform height
    theta = math.degrees(math.atan2(h, lp))    # ramp pitch to reach h over lp
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    size = [lp / 2, hw, _SLAB_T / 2]

    def static_slab(tag, foff, tilt_deg, rgba):     # fixed approach ramp (top at h)
        c = np.array(pos) + fwd * foff
        q = _qmul(_yaw_quat(yaw), _axis_quat([0.0, 1.0, 0.0], tilt_deg))
        cz = h / 2 - _qrot(q, (0.0, 0.0, _SLAB_T / 2))[2]
        spec.worldbody.add_geom(
            name=f"{name}_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size, pos=[float(c[0]), float(c[1]), cz], quat=q, rgba=rgba)

    static_slab("up",   -1.5 * lp, -theta, _OBST)
    static_slab("down", +1.5 * lp, +theta, _OBST2)
    bankL = _mocap_slab(spec, f"{name}_bankL", np.array(pos) + fwd * (-0.5 * lp),
                        yaw, [1.0, 0.0, 0.0], +ang, size, "top", h, _OBST2)
    bankR = _mocap_slab(spec, f"{name}_bankR", np.array(pos) + fwd * (+0.5 * lp),
                        yaw, [1.0, 0.0, 0.0], -ang, size, "top", h, _OBST)
    return {"name": name, "bodies": [bankL, bankR]}


def _add_ramp(spec, name, pos, yaw, hw, p):
    """A SOLID up-and-over hill: rises from the ground to a peak, then descends.

    Both slopes are angle-DR mocap slabs (ground-contact); the DR factor scales the
    ramp pitch, so the hill gets steeper/shallower per episode.
    """
    ang = p["ramp_deg"]
    if ang <= 0:
        return None
    run = 2.2                                           # horizontal run of each slope
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    half = run / 2
    size = [half, hw, _SLAB_T / 2]
    bodies = []
    for sign, tag in ((-1, "up"), (+1, "down")):        # up-slope rises to the peak
        c = np.array(pos) + fwd * (sign * half)
        bodies.append(_mocap_slab(spec, f"{name}_{tag}", c, yaw, [0.0, 1.0, 0.0],
                                  sign * ang, size, "ground", 0.0, _OBST))
    return {"name": name, "bodies": bodies}


_BUILDERS = {
    "paving": _add_paving,
    "hurdle": _add_hurdle,
    "staircase": _add_staircase,
    "side_incline": _add_side_incline,
    "ramp": _add_ramp,
}
# In LAP ORDER: gentle features first, the hard banked traverse just before the finish.
_LAYOUT = ["paving", "hurdle", "staircase", "ramp", "side_incline"]


def add_obstacles(spec: mujoco.MjSpec, centerline: np.ndarray, cfg) -> list:
    """Place the features along the track in lap order (banked traverse before finish).

    Returns a list of domain-randomization descriptors (one per DR-capable obstacle:
    currently hurdle + staircase), each ``{"name", "bodies": [{body, h_nom, H}, ...]}``.
    Static features (paving/ramp/banked) return None and are skipped.
    """
    level = getattr(cfg, "level_difficulty", 0)
    if level <= 0:
        return []
    p = difficulty_params(level)
    positions = _straight_positions(centerline, len(_LAYOUT))
    total = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    dr = []
    for kind, s in zip(_LAYOUT, positions):
        s = min(s, total - 3.6)                 # keep long features clear of the finish
        pos, yaw = _sample_path(centerline, s)
        rec = _BUILDERS[kind](spec, f"obst_{kind}", pos, yaw, cfg.half_width, p)
        if rec is not None:
            dr.append(rec)
    return dr
