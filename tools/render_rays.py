"""Visualize the forward height-scan rays on an obstacle course.

Places Spot just before an obstacle, casts the 12 downward scan rays, and draws
them over the terrain (probe line + hit marker, coloured by terrain-relative
height) so we can see exactly what the sensor perceives before training on it.

    MUJOCO_GL=egl uv run python tools/render_rays.py --track circuit --level 2
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from automataleague.envs.parkour.config import ParkourConfig  # noqa: E402
from automataleague.envs.parkour import height_scan as hs  # noqa: E402
from automataleague.envs.parkour.scene import build_parkour_model  # noqa: E402
from automataleague.envs.parkour.tracks import TRACKS  # noqa: E402


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _obstacle_geoms(model):
    """World (x, y, top_z) of collidable terrain geoms (group-0 boxes above the floor)."""
    out = []
    for i in range(model.ngeom):
        if model.geom_group[i] != 0 or model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        pos, half = model.geom_pos[i], model.geom_size[i]
        top = pos[2] + half[2]
        if top > 0.04:                       # skip the ~2 mm painted floor decals
            out.append((pos[0], pos[1], top))
    return np.array(out) if out else np.zeros((0, 3))


def _place_before_obstacle(centerline, obstacles, back=0.55):
    """Place Spot `back` metres before the leading face of the tallest obstacle.

    Groups the geoms making up that obstacle, orients along the path, and finds the
    near edge (first face the robot meets) so the scan grid straddles the rise.
    """
    target = obstacles[np.argmax(obstacles[:, 2]), :2]
    j = int(np.argmin(np.linalg.norm(centerline - target, axis=1)))
    tangent = centerline[min(j + 1, len(centerline) - 1)] - centerline[max(j - 1, 0)]
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)

    cluster = obstacles[np.linalg.norm(obstacles[:, :2] - target, axis=1) < 2.5, :2]
    if np.dot(tangent, target - cluster.mean(0)) < 0:      # tangent must point into it
        tangent = -tangent
    proj = cluster @ tangent                                # path coordinate of each geom
    near_edge = cluster[int(np.argmin(proj))]              # first face the robot reaches
    spawn_xy = near_edge - back * tangent
    yaw = float(np.arctan2(tangent[1], tangent[0]))
    return spawn_xy.astype(float), yaw


def _add_sphere(scn, pos, r, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0.0]),
                        np.asarray(pos, float), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _add_line(scn, frm, to, width, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(frm, float), np.asarray(to, float))
    scn.ngeom += 1


def _height_color(rel):
    """Blue (low / at foot level) -> amber (high terrain ahead). No alarm reds."""
    t = float(np.clip((rel + 0.05) / 0.40, 0.0, 1.0))
    low, high = np.array([0.20, 0.55, 1.0]), np.array([1.0, 0.75, 0.20])
    return list((1 - t) * low + t * high) + [1.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", default="circuit", choices=list(TRACKS))
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--out", default="renders/height_scan_rays.png")
    args = p.parse_args()

    cfg = ParkourConfig(track=args.track, level_difficulty=args.level)
    model, info = build_parkour_model("spot", cfg)
    data = mujoco.MjData(model)

    obstacles = _obstacle_geoms(model)
    if len(obstacles) == 0:
        raise SystemExit(f"no obstacles at level {args.level} — try --level 2+")
    spawn_xy, yaw = _place_before_obstacle(info.centerline, obstacles)

    # place Spot at the chosen pose, feet on the ground, joints at home stance
    data.qpos[:] = info.home_qpos
    ba = info.base_qposadr
    data.qpos[ba:ba + 3] = [spawn_xy[0], spawn_xy[1], info.robot.nominal_height]
    data.qpos[ba + 3:ba + 7] = _yaw_quat(yaw)
    mujoco.mj_forward(model, data)

    # cast the scan
    offsets = hs.scan_offsets()
    query_xy = hs.world_query_points(spawn_xy, yaw, offsets)
    terrain_z, hits = hs.cpu_terrain_heights(model, data, query_xy)
    base_z = float(data.qpos[ba + 2])
    rel = hs.scan_relative(terrain_z, base_z, info.robot.nominal_height)

    print(f"track={args.track} level={args.level} | Spot at "
          f"({spawn_xy[0]:.1f},{spawn_xy[1]:.1f}) yaw={np.degrees(yaw):.0f}deg")
    print("scan (terrain height relative to foot level, metres), 4 rows fwd x 3 cols lat:")
    for r in range(len(hs.SCAN_FWD)):
        row = rel[r * len(hs.SCAN_LAT):(r + 1) * len(hs.SCAN_LAT)]
        print(f"  {hs.SCAN_FWD[r]:.2f} m ahead: " + "  ".join(f"{v:+.3f}" for v in row))

    # render with the rays drawn on top
    renderer = mujoco.Renderer(model, height=900, width=1400)
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([float(query_xy[:, 0].mean()),
                           float(query_xy[:, 1].mean()), 0.15])
    cam.azimuth = np.degrees(yaw) - 125     # 3/4 view from behind-left of the robot
    cam.elevation, cam.distance = -22, 3.4
    renderer.update_scene(data, camera=cam)
    scn = renderer.scene
    for i in range(len(hits)):
        c = _height_color(rel[i])
        origin = [query_xy[i, 0], query_xy[i, 1], base_z + 0.05]   # sensor height
        _add_line(scn, origin, hits[i], 0.004, c[:3] + [0.55])     # downward probe
        _add_sphere(scn, hits[i], 0.032, c)                        # terrain hit
    frame = renderer.render()
    renderer.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.imwrite(args.out, frame)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
