"""Environment registry — named, versioned task environments.

An ENV is a registry id (``"parkour-1"``) mapped to an ``EnvSpec``: a default
``ParkourConfig`` factory over the shared ``envs/parkour`` engine, plus season
metadata. Users import an env by id:

    from automataleague_parkour import make_env, list_environments
    env = make_env("parkour-1", robot="spot", level=4)          # season-0, hardest

Add a future (harder) season by adding one ``EnvSpec`` entry — the engine is shared.
"""
from __future__ import annotations

from dataclasses import dataclass

from automataleague_parkour.envs.parkour.config import (
    ParkourConfig,
    RewardConfig,
    TerminationConfig,
)


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    season: int
    description: str
    track: str
    n_levels: int                              # difficulty levels 0..n_levels-1
    action_scale_by_level: tuple[float, ...]   # per-level q-target scale (radians)
    # Per-level episode budget, in control steps at 50 Hz. Harder levels are slower,
    # and an episode that ends before the finish never pays the success bonus and
    # never trains the last part of the course: the policy is optimising a lap it
    # is not allowed to complete. Sized from measured lap times with headroom.
    max_episode_steps_by_level: tuple[int, ...] = ()
    height_scan: bool = True
    track_perception: str = "boundary"   # none | centerline | boundary
    # Per-episode obstacle scaling ~ U(dr_low, dr_high) around the level's nominal
    # height. On by default: it is an implicit within-level curriculum (some episodes
    # come out easier), which is what lets a policy get a first success to learn from.
    # Evaluation forces it off, so scores are always on the nominal course.
    randomize_obstacles: bool = True

    def max_episode_steps(self, level: int) -> int:
        """Episode budget for `level`, falling back to the last entry (then 3000)."""
        if not self.max_episode_steps_by_level:
            return 3000
        idx = min(int(level), len(self.max_episode_steps_by_level) - 1)
        return int(self.max_episode_steps_by_level[idx])

    def config(self, level: int, **overrides) -> ParkourConfig:
        """Default ParkourConfig for this env at `level`, before any hydra overrides."""
        if not 0 <= level < self.n_levels:
            raise ValueError(
                f"{self.env_id}: level {level} out of range 0..{self.n_levels - 1}")
        cfg = ParkourConfig(
            track=self.track,
            level_difficulty=level,
            height_scan=self.height_scan,
            track_perception=self.track_perception,
            randomize_obstacles=self.randomize_obstacles,
            action_scale=self.action_scale_by_level[level],
        )
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown ParkourConfig field '{k}'")
            setattr(cfg, k, v)
        return cfg


ENVIRONMENTS: dict[str, EnvSpec] = {
    "parkour-1": EnvSpec(
        env_id="parkour-1",
        season=0,
        description="Season 0 — winding closed circuit, 5 difficulty levels (flat -> L4).",
        track="circuit",
        n_levels=5,
        # L2 is 0.70, not the 0.50 the geometry alone suggests: 0.70 is what the
        # runs that actually cleared level 2 used.
        action_scale_by_level=(0.30, 0.30, 0.70, 0.70, 0.79),
        # Measured laps: L0 ~1465 steps, obstacle levels 1600-2200 and slower still
        # while learning. 3000 is the ceiling; flat needs less.
        max_episode_steps_by_level=(2000, 3000, 3000, 3000, 3000),
        height_scan=True,
        track_perception="boundary",
        randomize_obstacles=True,
    ),
}


def get_env_spec(env_id: str) -> EnvSpec:
    if env_id not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment '{env_id}'. Registered: {sorted(ENVIRONMENTS)}")
    return ENVIRONMENTS[env_id]


def list_environments(season: int | None = None) -> list[EnvSpec]:
    specs = list(ENVIRONMENTS.values())
    if season is not None:
        specs = [s for s in specs if s.season == season]
    return sorted(specs, key=lambda s: (s.season, s.env_id))


def make_env(env_id, robot="spot", level=None, backend="warp", num_envs=None,
             reward_cfg=None, reward_fn=None, term_cfg=None, **cfg_overrides):
    """Instantiate a registered env. backend: 'warp' (GPU, batched) | 'cpu' (single).

    reward_cfg tunes the default reward's weights; reward_fn (optional) replaces the
    reward entirely with a custom callable (same signature as rewards.compute_reward,
    returning ``(reward, components)``). If reward_fn is None the default is used.
    """
    spec = get_env_spec(env_id)
    lvl = spec.n_levels - 1 if level is None else int(level)
    course = spec.config(lvl, **cfg_overrides)
    rc = reward_cfg if reward_cfg is not None else RewardConfig()
    tc = term_cfg if term_cfg is not None else TerminationConfig()
    if backend == "cpu":
        from automataleague_parkour.envs.parkour.parkour_cpu import ParkourEnvCPU
        return ParkourEnvCPU(robot=robot, cfg=course, reward_cfg=rc,
                             reward_fn=reward_fn, term_cfg=tc)
    if backend == "warp":
        from automataleague_parkour.envs.parkour.parkour_warp import ParkourEnvWarp  # GPU-only
        return ParkourEnvWarp(
            robot=robot, num_envs=num_envs or 2048, cfg=course,
            reward_cfg=rc, reward_fn=reward_fn, term_cfg=tc)
    raise ValueError(f"Unknown backend '{backend}' (use 'warp' or 'cpu')")
