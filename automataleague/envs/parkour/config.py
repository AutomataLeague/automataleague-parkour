"""Configuration for the parkour task.

These values are the single source of truth for both the *scene geometry*
(walls, lines, checkpoint markers) and the *reward* (checkpoint coordinates,
radius). The scene builder and the env read the same config, so they can't drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParkourConfig:
    # --- track shape (a centerline polyline; see tracks.py) ---
    track: str = "straight"      # "straight" | "l_curved" | ...
    length: float = 9.0          # length of the "straight" track
    spawn_x: float = 1.0         # spawn distance along the first leg

    # --- Stage-1 obstacles (see obstacles.py). 0 = flat; 1..4 scale up step
    # heights / ramp & incline angles for a difficulty curriculum. ---
    level_difficulty: int = 0

    # --- path geometry (metres) ---
    half_width: float = 1.6      # path half-width; boundary lines at ±half_width from the
                                 # centerline. Also the off-path tolerance: perpendicular
                                 # distance to the centerline beyond this terminates.
    line_thickness: float = 0.05  # painted line width, all markers (visual only)

    # --- checkpoints ---
    checkpoint_spacing: float = 2.0  # gate spacing along the centerline (metres)
    checkpoint_radius: float = 0.5   # within this xy-distance = checkpoint reached
    # The success trigger sits this far PAST the visual finish line, so the robot must
    # walk THROUGH the visible line to finish, not stop short of it.
    finish_offset: float = 0.75

    def build_track(self):
        """Construct the Track (centerline polyline) for this config."""
        from automataleague.envs.parkour.tracks import get_track

        if self.track == "straight":
            return get_track("straight", length=self.length, spawn_x=self.spawn_x)
        return get_track(self.track, spawn_x=self.spawn_x)

    def checkpoints_xy(self) -> np.ndarray:
        """(K, 2) gate waypoints along the centerline, ending at the finish point."""
        return self.build_track().checkpoints(self.checkpoint_spacing, self.finish_offset)


@dataclass
class RewardConfig:
    """Weights for the parkour reward (see envs/parkour/rewards.py, Phase B)."""

    # Forward-velocity reward: the DENSE locomotion driver. Rewards base velocity
    # toward the current checkpoint, capped at target_speed (so moving always beats
    # standing, without rewarding unstable lunges). This is what makes walking emerge.
    forward: float = 1.5         # weight on forward speed toward the goal
    target_speed: float = 1.0    # m/s; reward saturates here
    progress: float = 2.0        # potential-based goal shaping (distance reduction)
    checkpoint: float = 10.0     # bonus for reaching a checkpoint
    success: float = 100.0       # bonus for reaching the finish
    alive: float = 0.0           # per-step survival bonus (0: avoid loitering optimum)
    upright: float = 0.05        # exp-shaped uprightness (low, so standing isn't optimal)
    height: float = 0.05         # exp-shaped height-keeping
    action: float = 0.01         # penalty on action magnitude
    joint_vel: float = 0.001     # penalty on joint velocity (smoothness/energy)
    fall: float = 10.0           # one-off penalty (negative) when the robot falls
    off_path: float = 25.0       # one-off penalty (negative) + terminate when off the path


@dataclass
class TerminationConfig:
    fall_height: float = 0.25    # base below this = fallen
    max_tilt_deg: float = 60.0   # torso tilt beyond this = fallen
    off_path: bool = True        # terminate when the base strays off the path
    # off-path tolerance = ParkourConfig.half_width (lateral distance from centerline)
    max_episode_steps: int = 1000
