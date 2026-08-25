"""Configuration for the parkour task.

These values are the single source of truth for both the *scene geometry*
(walls, lines, checkpoint markers) and the *reward* (checkpoint coordinates,
radius). The scene builder and the env read the same config, so they can't drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Start-pose randomisation applied on reset, shared by both backends. Gaussian, in
# metres on the base xy and radians on each actuated joint. Evaluation has to use the
# SAME noise the policy trained under: a clean deterministic start is out of
# distribution and its numbers disagreed with real performance for days here
# (training-recipe.md, finding 5).
DEFAULT_RESET_POS_NOISE = 0.05
DEFAULT_RESET_JOINT_NOISE = 0.05


@dataclass
class ParkourConfig:
    # --- track shape (a centerline polyline; see tracks.py) ---
    track: str = "straight"      # "straight" | "l_curved" | ...
    length: float = 9.0          # length of the "straight" track
    spawn_x: float = 1.0         # spawn distance along the first leg

    # --- Stage-1 obstacles (see obstacles.py). 0 = flat; 1..4 scale up step
    # heights / ramp & incline angles for a difficulty curriculum. ---
    level_difficulty: int = 0

    # --- perception (see height_scan.py). When True, append a 12-point forward
    # terrain height scan to the observation (obs_dim += SCAN_N). Lets the policy
    # anticipate obstacles instead of feeling them only on contact. ---
    height_scan: bool = True

    # --- track perception (see path_preview.py). Appends K lookahead points ahead of
    # the agent (in its base frame) plus the signed lateral offset to the observation
    # (obs_dim += preview_dim(preview_distances, mode)), so the policy sees upcoming
    # curves instead of reacting to heading error alone. Off by default; racing turns
    # it on (see examples/config_race.yaml). Set via one knob:
    #   "none"       : disabled (blind to the track ahead). The default.
    #   "boundary"   : a LEFT and a RIGHT corridor-edge point per lookahead (at
    #                  ±half_width) -> the policy perceives the drivable channel and
    #                  can cut the apex. What racing uses.
    #   "centerline" : one centerline point per lookahead -> tracks the middle. ---
    track_perception: str = "boundary"
    preview_distances: tuple = (1.5, 3.0, 4.5, 6.0)  # lookahead distances (metres)

    # --- obstacle domain randomization (mocap-based, see obstacles.py). When True,
    # each episode draws a per-env difficulty factor ~ U(dr_low, dr_high) per DR
    # obstacle, with level_difficulty's value as the mean. Eval keeps factor = 1.
    randomize_obstacles: bool = True
    dr_low: float = 0.5
    dr_high: float = 1.5

    # --- race mode: navigate by along-track speed + gate-crossing (no forced dip to
    # checkpoint centres) so the agent can take the ideal line. Pair with the race
    # reward preset (progress=0, checkpoint=0, alive<0 time penalty, target_speed high)
    # to fine-tune a solved obstacle policy for minimal lap time. ---
    race_mode: bool = False

    # --- action range: q_target = home + action_scale * action (action in [-1,1]).
    # None => use the robot's default (Spot 0.3). Larger scale = more foot lift, needed
    # for taller obstacles. Max foot lift by scale (measured): 0.3->0.15m, 0.5->0.28m,
    # 0.7->0.40m, 0.79(hip-splay limit)->0.45m. Recommended per-level schedule:
    #   L1: 0.30 (clears 0.11m) | L2: 0.50 (0.22m) | L3: 0.70 (0.34m) | L4: ~0.79 (0.45m)
    action_scale: float | None = None

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
        from automataleague_parkour.envs.parkour.tracks import get_track

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
    # Foot air-time gait reward (Warp env only): rewards each foot for staying airborne
    # ~feet_air_time_target seconds before touchdown, which produces a clean stepping gait
    # instead of a shuffle/drag. Only paid while moving. Default 0 = off (Spot unaffected).
    feet_air_time: float = 0.0
    feet_air_time_target: float = 0.25   # seconds; target swing duration per step


@dataclass
class TerminationConfig:
    fall_height: float = 0.25    # base below this = fallen
    max_tilt_deg: float = 60.0   # torso tilt beyond this = fallen
    off_path: bool = True        # terminate when the base strays off the path
    # off-path tolerance = ParkourConfig.half_width (lateral distance from centerline)
    max_episode_steps: int = 1000
