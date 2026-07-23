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


TRACKS: dict[str, Callable[..., Track]] = {
    "straight": make_straight,
    "l_curved": make_l_curved,
}


def get_track(name: str, **kwargs) -> Track:
    if name not in TRACKS:
        raise ValueError(f"Unknown track '{name}'. Available: {sorted(TRACKS)}")
    return TRACKS[name](**kwargs)
