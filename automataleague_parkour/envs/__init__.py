"""Task environments for the league (parkour now; boxing/sumo/... later)."""

from automataleague_parkour.envs.registry import (  # noqa: F401
    ENVIRONMENTS,
    EnvSpec,
    get_env_spec,
    list_environments,
    make_env,
)
