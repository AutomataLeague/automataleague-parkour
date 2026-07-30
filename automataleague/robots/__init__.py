"""Robot registry.

Add a robot by writing a ``make_<robot>() -> RobotSpec`` factory in a module here
and registering it in ``ROBOTS``. Tasks select a robot by name via ``get_robot``.
"""

from __future__ import annotations

from typing import Callable

from automataleague.robots.base import RobotSpec
from automataleague.robots.spot import make_spot
from automataleague.robots.unitree_go1 import make_go1

ROBOTS: dict[str, Callable[[], RobotSpec]] = {
    "spot": make_spot,
    "go1": make_go1,
}


def get_robot(name: str) -> RobotSpec:
    if name not in ROBOTS:
        raise ValueError(
            f"Unknown robot '{name}'. Registered robots: {sorted(ROBOTS)}"
        )
    return ROBOTS[name]()


__all__ = ["RobotSpec", "ROBOTS", "get_robot"]
