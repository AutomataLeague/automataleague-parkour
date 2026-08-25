import pytest

pytest.importorskip("omegaconf", reason="hydra/omegaconf ship in the `train` extra")

from omegaconf import OmegaConf  # noqa: E402

from automataleague_parkour.training.curriculum import (  # noqa: E402
    CurriculumConfig,
    curriculum_from_cfg,
)


def test_curriculum_from_cfg():
    cfg = OmegaConf.create({"curriculum": {
        "levels": [1, 2, 3, 4],
        "frames_per_level": [10, 10, 10, 20],
        "action_scale_per_level": [0.30, 0.50, 0.70, 0.79],
        "warm_start": True, "eval_per_level": True}})
    cur = curriculum_from_cfg(cfg)
    assert isinstance(cur, CurriculumConfig)
    assert cur.levels == [1, 2, 3, 4]
    assert cur.action_scale_per_level[-1] == 0.79
    assert cur.warm_start is True


def test_curriculum_length_validation():
    import pytest
    cfg = OmegaConf.create({"curriculum": {
        "levels": [1, 2], "frames_per_level": [10],           # mismatched
        "action_scale_per_level": [0.3, 0.5], "warm_start": True}})
    with pytest.raises(ValueError):
        curriculum_from_cfg(cfg)
