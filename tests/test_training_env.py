import pytest

pytest.importorskip("omegaconf", reason="hydra/omegaconf ship in the `train` extra")

from omegaconf import OmegaConf  # noqa: E402

from automataleague_parkour.training.env import configs_from_cfg  # noqa: E402


def _cfg(**course):
    base = {"name": "parkour-1", "robot": "spot", "max_episode_steps": 1000,
            "course": {"level_difficulty": 2, "action_scale": None, "height_scan": True}}
    base["course"].update(course)
    return OmegaConf.create({"env": base})


def test_registry_backed_defaults():
    course, rc, tc = configs_from_cfg(_cfg())
    assert course.track == "circuit"            # from registry (env.name=parkour-1)
    assert course.level_difficulty == 2
    assert course.action_scale == pytest.approx(0.70)   # null in cfg -> registry schedule
    assert tc.max_episode_steps == 1000


def test_explicit_override_wins():
    course, _, _ = configs_from_cfg(_cfg(action_scale=0.79, race_mode=True))
    assert course.action_scale == pytest.approx(0.79)
    assert course.race_mode is True


def test_null_level_difficulty_defaults_to_zero():
    course, _, _ = configs_from_cfg(_cfg(level_difficulty=None))
    assert course.level_difficulty == 0
    assert course.track == "circuit"
    assert course.action_scale == pytest.approx(0.30)
