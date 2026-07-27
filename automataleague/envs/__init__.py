"""Task environments for the league (parkour now; boxing/sumo/... later)."""

from automataleague.envs.registry import (  # noqa: F401
    EnvSpec,
    ENVIRONMENTS,
    get_env_spec,
    list_environments,
    make_env,
)
