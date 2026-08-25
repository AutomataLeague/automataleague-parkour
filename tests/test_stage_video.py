"""The per-stage video hook the training entry points call.

A video is a nice-to-have that runs *after* a training stage has already succeeded,
so the one behaviour that matters is that it can never turn a finished run into a
failed one.
"""
import importlib.util

import pytest

pytest.importorskip("omegaconf", reason="hydra/omegaconf ship in the `train` extra")

from omegaconf import OmegaConf  # noqa: E402

torchrl_only = pytest.mark.skipif(
    importlib.util.find_spec("torchrl") is None,
    reason="torchrl ships in the `train` extra",
)


@torchrl_only
def test_stage_video_is_off_unless_asked(tmp_path):
    from automataleague_parkour.training import render_stage_video
    cfg = OmegaConf.create({"logger": {"stage_video": False}})
    assert render_stage_video(cfg, "nonexistent.pt", "run", videos_dir=str(tmp_path)) is None
    assert list(tmp_path.iterdir()) == []


@torchrl_only
def test_stage_video_absent_key_is_off(tmp_path):
    from automataleague_parkour.training import render_stage_video
    cfg = OmegaConf.create({"logger": {}})
    assert render_stage_video(cfg, "nonexistent.pt", "run", videos_dir=str(tmp_path)) is None


@torchrl_only
def test_stage_video_failure_does_not_raise(tmp_path):
    """A broken checkpoint must not take down a training run that already finished."""
    from automataleague_parkour.training import render_stage_video
    cfg = OmegaConf.create({"logger": {"stage_video": True}})
    assert render_stage_video(cfg, str(tmp_path / "missing.pt"), "run",
                              videos_dir=str(tmp_path)) is None
