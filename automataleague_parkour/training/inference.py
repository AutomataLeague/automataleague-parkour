"""Load a trained PPO checkpoint into an eval-ready actor.

`run_ppo` saves ``{actor_state_dict, critic_state_dict, config}``; the config records
the robot, the perception settings and the network shape, so the actor rebuilds without
a live env. Consumed by the render / eval tools in ``tools/``.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from automataleague_parkour.robots import get_robot
from automataleague_parkour.training.models import build_actor


def load_policy(checkpoint_path, device="cpu"):
    """Rebuild the PPO policy stored in `checkpoint_path`.

    Returns ``(actor, cfg)``. The actor is in eval mode; roll it out under
    ``ExplorationType.DETERMINISTIC`` to take its mean action (see
    ``envs.parkour.render.record_trajectory``).

    The returned `cfg` is normalised so the env built from it matches the network:
    see `_pin_legacy_perception`.
    """
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    _pin_legacy_perception(cfg)
    robot = get_robot(cfg.env.robot)
    actor = build_actor(cfg, robot, device)          # obs/action dims come from cfg
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    _check_env_matches_policy(cfg, robot, ckpt, checkpoint_path)
    return actor, cfg


def _pin_legacy_perception(cfg):
    """Pin `track_perception` on a checkpoint that predates the sensor.

    `configs_from_cfg` fills unset course fields from the env registry, so a
    checkpoint whose config never mentioned perception picks up whatever the
    registry defaults to *today*. When that default changed to "boundary", replaying
    an older policy silently built a 78-wide env for a 61-wide network. Absence of
    every perception key means the run predates the feature: it trained blind.
    """
    course = getattr(cfg.env, "course", None)
    if course is None:
        return
    if not any(k in course for k in ("track_perception", "path_preview", "preview_mode")):
        course.track_perception = "none"


def _check_env_matches_policy(cfg, robot, ckpt, checkpoint_path):
    """Fail loudly if the env this cfg builds is not the one the policy trained on.

    A width mismatch surfaces as an opaque matmul shape error deep in the rollout, and
    a mismatch that happens to share a width would not surface at all.
    """
    from automataleague_parkour.envs.parkour.observation import obs_layout
    from automataleague_parkour.training.env import configs_from_cfg

    stored = ckpt.get("obs_layout")
    if not stored:
        return                                   # pre-layout checkpoint; nothing to check
    course, _, _ = configs_from_cfg(cfg)
    actual = obs_layout(course, robot)
    expected = tuple((str(n), int(w)) for n, w in stored)
    if actual != expected:
        def fmt(lay):
            return "[" + " | ".join(f"{n} {w}" for n, w in lay) + f"] = {sum(w for _, w in lay)}"
        raise ValueError(
            f"{checkpoint_path} was trained on a different observation than the env its "
            f"config now builds.\n  trained: {fmt(expected)}\n  env now: {fmt(actual)}\n"
            f"The env registry defaults have changed since this checkpoint was written. "
            f"Set the sensors explicitly (env.course.height_scan / track_perception) to "
            f"match how it trained.")
