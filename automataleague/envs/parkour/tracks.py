"""Track shapes for parkour, as centerline polylines.

A track is a dense polyline (the path centerline) plus a spawn pose. Straight, L,
S, C are all just different polylines — a smooth curve is a densely-sampled arc.
Everything downstream (checkpoints, off-path distance, rendering) is polyline-based,
so new shapes are new entries here and nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Track:
    name: str
    centerline: np.ndarray          # (M, 2) dense polyline of the path centre
    spawn_xy: tuple[float, float]
    spawn_heading: float = 0.0      # initial yaw (radians)

    def _cumlen(self) -> np.ndarray:
        seg = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    def total_length(self) -> float:
        return float(self._cumlen()[-1])

    def resample(self, spacing: float) -> np.ndarray:
        """Points along the centerline at ~`spacing` metres, including both ends."""
        cum = self._cumlen()
        s = np.arange(0.0, cum[-1] + 1e-6, spacing)
        xs = np.interp(s, cum, self.centerline[:, 0])
        ys = np.interp(s, cum, self.centerline[:, 1])
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def final_heading_dir(self) -> np.ndarray:
        d = self.centerline[-1] - self.centerline[-2]
        return d / (np.linalg.norm(d) + 1e-9)

    def finish_point(self, finish_offset: float) -> np.ndarray:
        """The success point: `finish_offset` past the last waypoint (the visual line)."""
        return self.centerline[-1] + finish_offset * self.final_heading_dir()

    def checkpoints(self, spacing: float, finish_offset: float) -> np.ndarray:
        """Gate waypoints along the path (skip the start), ending at the finish point."""
        pts = self.resample(spacing)[1:]            # drop the spawn/start point
        finish = self.finish_point(finish_offset)
        # replace the last resampled point with the finish success point
        pts = np.vstack([pts[:-1], finish]) if len(pts) > 1 else finish[None]
        return pts.astype(np.float32)


def make_straight(length: float = 9.0, spawn_x: float = 1.0) -> Track:
    centerline = np.array([[0.0, 0.0], [length, 0.0]], dtype=np.float32)
    return Track("straight", centerline, (spawn_x, 0.0), 0.0)


def make_l_curved(
    leg1: float = 5.0, radius: float = 2.5, leg2: float = 5.0,
    spawn_x: float = 1.0, arc_pts: int = 40,
) -> Track:
    """East leg -> smooth quarter-circle (turning north) -> north leg."""
    center = np.array([leg1, radius])
    thetas = np.linspace(-np.pi / 2, 0.0, arc_pts)          # -90deg -> 0deg
    arc = center + radius * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    arc_end = arc[-1]                                        # (leg1+radius, radius)
    north_end = arc_end + np.array([0.0, leg2])
    centerline = np.vstack([[0.0, 0.0], arc, north_end]).astype(np.float32)
    return Track("l_curved", centerline, (spawn_x, 0.0), 0.0)


def build_centerline(segments: list[tuple], deg_per_pt: float = 4.0) -> np.ndarray:
    """Walk out a centerline from a sequence of moves (turtle-style).

    segments: list of ("straight", length) or ("arc", angle_deg, radius).
    Positive angle = left turn (CCW), negative = right. Start at (0,0) heading +x.
    """
    pos = np.array([0.0, 0.0])
    heading = 0.0
    pts = [pos.copy()]
    for seg in segments:
        if seg[0] == "straight":
            length = seg[1]
            pos = pos + length * np.array([np.cos(heading), np.sin(heading)])
            pts.append(pos.copy())
        elif seg[0] == "arc":
            angle_deg, radius = seg[1], seg[2]
            angle = np.radians(angle_deg)
            sign = 1.0 if angle >= 0 else -1.0
            perp = sign * np.array([-np.sin(heading), np.cos(heading)])   # toward centre
            center = pos + radius * perp
            start_ang = np.arctan2(pos[1] - center[1], pos[0] - center[0])
            n = max(2, int(abs(angle_deg) / deg_per_pt))
            for k in range(1, n + 1):
                a = start_ang + angle * k / n
                pts.append(center + radius * np.array([np.cos(a), np.sin(a)]))
            pos = pts[-1].copy()
            heading += angle
        else:
            raise ValueError(f"unknown segment {seg!r}")
    return np.array(pts, dtype=np.float32)


def _route_segments(straights: list[float], turns: list[float], radius: float) -> list[tuple]:
    segs: list[tuple] = []
    for i, length in enumerate(straights):
        segs.append(("straight", length))
        if i < len(turns):
            segs.append(("arc", turns[i], radius))
    return segs


def closed_route(turns: list[float], radius: float = 2.0, base: float = 3.0,
                 gap: float = 1.6, spawn_x: float = 1.0, name: str = "circuit") -> Track:
    """Build a closed winding loop from a turn sequence (signed degrees; + = left).

    The turns must net +/-360 deg (so the heading closes). Straight lengths enter the
    end position linearly, so we solve them with least-squares to return to the start,
    then trim the final straight by `gap` to leave a small start/finish opening.
    """
    n = len(turns) + 1
    headings = [0.0]
    for t in turns:
        headings.append(headings[-1] + np.radians(t))
    dirs = np.array([[np.cos(h), np.sin(h)] for h in headings])          # (n,2)
    arc_only = build_centerline(_route_segments([0.0] * n, turns, radius))
    offset = arc_only[-1] - arc_only[0]
    A = dirs.T                                                          # (2,n)
    for _ in range(20):
        b = -offset - A @ (base * np.ones(n))
        delta, *_ = np.linalg.lstsq(A, b, rcond=None)
        straights = base + delta
        if straights.min() >= 1.5 or base > 8.0:
            break
        base += 0.5
    straights = straights.copy()
    straights[-1] = max(0.4, straights[-1] - gap)   # small start/finish opening
    cl = build_centerline(_route_segments(list(straights), turns, radius))
    return Track(name, cl, (spawn_x, 0.0), 0.0)


def make_circuit(spawn_x: float = 1.0) -> Track:
    """A winding closed circuit from a right/left turn sequence that loops back to
    the start (net 360 deg; solved straights so it closes, no self-crossing)."""
    # net 360; single (non-adjacent) right-turn dents keep the path clear of itself
    # (no lane overlap) while still winding with left+right turns.
    turns = [90, 90, -90, 90, 90, -90, 90, 90]
    return closed_route(turns, radius=2.2, spawn_x=spawn_x)


def make_s_curved(
    leg1: float = 4.0, radius: float = 2.5, mid: float = 2.0, leg3: float = 4.0,
    spawn_x: float = 1.0, arc_pts: int = 40,
) -> Track:
    """East leg -> left arc (turn north) -> north leg -> right arc (turn east) -> east
    leg. An S made of two opposite quarter-circle bends."""
    # Arc 1: left turn east->north, centered at (leg1, radius).
    c1 = np.array([leg1, radius])
    th1 = np.linspace(-np.pi / 2, 0.0, arc_pts)
    arc1 = c1 + radius * np.stack([np.cos(th1), np.sin(th1)], axis=1)
    mid_end = arc1[-1] + np.array([0.0, mid])            # north leg
    # Arc 2: right turn north->east, centered to the east of mid_end.
    c2 = mid_end + np.array([radius, 0.0])
    th2 = np.linspace(np.pi, np.pi / 2, arc_pts)
    arc2 = c2 + radius * np.stack([np.cos(th2), np.sin(th2)], axis=1)
    end = arc2[-1] + np.array([leg3, 0.0])               # final east leg
    centerline = np.vstack([[0.0, 0.0], arc1, mid_end, arc2, end]).astype(np.float32)
    return Track("s_curved", centerline, (spawn_x, 0.0), 0.0)


TRACKS: dict[str, Callable[..., Track]] = {
    "straight": make_straight,
    "l_curved": make_l_curved,
    "s_curved": make_s_curved,
    "circuit": make_circuit,
}


def get_track(name: str, **kwargs) -> Track:
    if name not in TRACKS:
        raise ValueError(f"Unknown track '{name}'. Available: {sorted(TRACKS)}")
    return TRACKS[name](**kwargs)
