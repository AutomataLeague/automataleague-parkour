"""The learning curve must survive a run with no logger backend.

Episode return is the metric that says whether training converged; without it you
cannot distinguish a finished run from one whose reward was still climbing. Six runs
were completed here with `logger.backend=""` and left no curve at all, so the JSONL
sink is written unconditionally and this test pins that.
"""
import json

import pytest

pytest.importorskip("torchrl", reason="torchrl ships in the `train` extra")

from automataleague_parkour.training.env import log_metrics  # noqa: E402


def test_metrics_are_written_without_a_logger(tmp_path):
    path = tmp_path / "metrics.jsonl"
    log_metrics(None, {"train/reward": 12.5, "train/episode_length": 900.0}, 1024,
                jsonl_path=str(path))
    rec = json.loads(path.read_text().strip())
    assert rec == {"frames": 1024, "train/reward": 12.5, "train/episode_length": 900.0}


def test_metrics_append_one_row_per_call(tmp_path):
    path = tmp_path / "metrics.jsonl"
    for step, reward in ((1000, 1.0), (2000, 4.0), (3000, 9.0)):
        log_metrics(None, {"train/reward": reward}, step, jsonl_path=str(path))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["frames"] for r in rows] == [1000, 2000, 3000]
    assert [r["train/reward"] for r in rows] == [1.0, 4.0, 9.0]


def test_non_numeric_values_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "metrics.jsonl"
    log_metrics(None, {"train/reward": 3.0, "train/note": "text"}, 10, jsonl_path=str(path))
    rec = json.loads(path.read_text().strip())
    assert rec["train/reward"] == 3.0 and "train/note" not in rec


def test_jsonl_path_none_is_a_noop(tmp_path):
    log_metrics(None, {"train/reward": 1.0}, 10, jsonl_path=None)   # must not raise
    assert list(tmp_path.iterdir()) == []
