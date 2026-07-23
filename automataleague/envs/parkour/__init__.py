"""Parkour task: walk a corridor from start to finish, hitting checkpoints."""

from automataleague.envs.parkour.config import (
    ParkourConfig,
    RewardConfig,
    TerminationConfig,
)
from automataleague.envs.parkour.scene import SceneInfo, build_parkour_model

__all__ = [
    "ParkourConfig",
    "RewardConfig",
    "TerminationConfig",
    "SceneInfo",
    "build_parkour_model",
]
