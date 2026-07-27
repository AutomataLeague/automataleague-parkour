"""Curriculum config: train across difficulty levels, warm-starting each stage."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    levels: list[int]
    frames_per_level: list[int]
    action_scale_per_level: list[float]
    warm_start: bool = True
    eval_per_level: bool = True


def curriculum_from_cfg(cfg) -> CurriculumConfig:
    c = cfg.curriculum
    cur = CurriculumConfig(
        levels=[int(x) for x in c.levels],
        frames_per_level=[int(x) for x in c.frames_per_level],
        action_scale_per_level=[float(x) for x in c.action_scale_per_level],
        warm_start=bool(getattr(c, "warm_start", True)),
        eval_per_level=bool(getattr(c, "eval_per_level", True)),
    )
    n = len(cur.levels)
    if not (len(cur.frames_per_level) == len(cur.action_scale_per_level) == n):
        raise ValueError(
            "curriculum lists must be equal length: levels/frames_per_level/"
            f"action_scale_per_level ({n}/{len(cur.frames_per_level)}/"
            f"{len(cur.action_scale_per_level)})")
    return cur
