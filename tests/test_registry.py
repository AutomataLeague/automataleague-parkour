import pytest
from automataleague import make_env, list_environments, get_env_spec
from automataleague.envs.registry import EnvSpec


def test_parkour1_registered():
    ids = [s.env_id for s in list_environments()]
    assert "parkour-1" in ids
    spec = get_env_spec("parkour-1")
    assert spec.season == 0 and spec.track == "circuit" and spec.n_levels == 5


def test_unknown_env_raises():
    with pytest.raises(ValueError):
        get_env_spec("nope")


def test_config_resolves_per_level():
    spec = get_env_spec("parkour-1")
    c = spec.config(2)
    assert c.track == "circuit"
    assert c.level_difficulty == 2
    assert c.height_scan is True
    assert c.action_scale == pytest.approx(0.50)
    # override wins
    assert spec.config(2, race_mode=True).race_mode is True


def test_make_env_cpu():
    env = make_env("parkour-1", robot="spot", level=3, backend="cpu")
    assert env.cfg.level_difficulty == 3
    assert env.cfg.action_scale == pytest.approx(0.70)
    assert env.robot.name == "spot"
