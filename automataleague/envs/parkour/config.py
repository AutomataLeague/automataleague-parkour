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
    # --- corridor geometry (metres) ---
    length: float = 9.0          # start line at x=0, finish line at x=length
    half_width: float = 1.6      # path half-width; boundary lines at y = ±half_width.
                                 # Also the off-path tolerance: |lateral offset| beyond
                                 # this terminates the episode (see TerminationConfig).
    line_thickness: float = 0.05  # painted line width, all markers (visual only)

    # --- checkpoints ---
    # Fractions of the corridor length for the INTERMEDIATE gates (the finish is
    # handled separately via finish_offset below).
    checkpoint_fracs: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
    checkpoint_radius: float = 0.5   # within this xy-distance = checkpoint reached
    # The success trigger sits this far PAST the visual finish line (drawn at x=length),
    # so the robot must walk THROUGH the visible line to finish, not stop short of it.
    finish_offset: float = 0.75

    # --- spawn ---
    spawn_x: float = 1.0         # robot base starts here (just past the start line)
    spawn_y: float = 0.0

    def checkpoints_xy(self) -> np.ndarray:
        """(K, 2) checkpoint centres: intermediate gates + the finish success point.

        The finish success point is `finish_offset` past the visual finish line, so
        the reward isn't satisfied until the robot crosses the drawn line.
        """
        xs = [f * self.length for f in self.checkpoint_fracs]
        xs.append(self.length + self.finish_offset)
        return np.array([[x, 0.0] for x in xs], dtype=np.float32)


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
