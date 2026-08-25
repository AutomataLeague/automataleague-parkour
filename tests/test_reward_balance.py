"""Preflight: does the shipped reward pay for the behaviour it asks for?

Recipe stage 0. For a time trial the finish bonus has to outweigh the accumulated
per-step time cost, or finishing is net-negative and the optimal policy is to stall
while every training curve still looks like progress. This is checked here rather
than left as prose, because the episode budget and the reward weights live in
different files and only interact at runtime.
"""
import pytest

pytest.importorskip("omegaconf", reason="hydra/omegaconf ship in the `train` extra")

from omegaconf import OmegaConf  # noqa: E402

from automataleague_parkour.envs.registry import get_env_spec  # noqa: E402
from automataleague_parkour.training.env import configs_from_cfg  # noqa: E402

RACE = "examples/config_race.yaml"
PPO = "examples/config_ppo.yaml"


def _cfg(path, level):
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:                      # config_race composes config_ppo
        cfg = OmegaConf.merge(OmegaConf.load(PPO), cfg)
        del cfg["defaults"]
    cfg.env.course.level_difficulty = level
    return cfg


@pytest.mark.parametrize("level", range(5))
def test_race_finish_bonus_beats_the_worst_case_time_cost(level):
    cfg = _cfg(RACE, level)
    course, rc, tc = configs_from_cfg(cfg)
    worst_case_time_cost = abs(rc.alive) * tc.max_episode_steps
    assert rc.success > worst_case_time_cost, (
        f"level {level}: success={rc.success} does not beat "
        f"|alive|={abs(rc.alive)} x {tc.max_episode_steps} steps = {worst_case_time_cost}. "
        "Finishing is net-negative, so the optimal policy is to stall.")


@pytest.mark.parametrize("level", range(5))
def test_episode_budget_allows_a_lap(level):
    """An episode that cannot contain a lap never pays the finish bonus and never
    trains the end of the course. Measured laps are ~1465 steps flat and up to ~2200
    on obstacles, so the budget must comfortably exceed that."""
    _, _, tc = configs_from_cfg(_cfg(PPO, level))
    assert tc.max_episode_steps >= 2000, (
        f"level {level}: {tc.max_episode_steps} steps is under a measured lap")


def test_registry_budget_is_capped_at_3000():
    spec = get_env_spec("parkour-1")
    for level in range(spec.n_levels):
        assert 2000 <= spec.max_episode_steps(level) <= 3000
