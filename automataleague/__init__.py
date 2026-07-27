"""automataleague — a league of robots competing at athletic tasks in MuJoCo."""

from automataleague.envs.registry import (
    EnvSpec,
    ENVIRONMENTS,
    get_env_spec,
    list_environments,
    make_env,
)

__all__ = ["make_env", "list_environments", "get_env_spec", "EnvSpec", "ENVIRONMENTS"]
