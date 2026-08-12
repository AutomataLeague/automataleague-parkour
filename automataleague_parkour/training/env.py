"""Env construction for training/eval — resolves configs through the env registry."""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import Compose, ExplorationType, TransformedEnv, set_exploration_type
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter

from automataleague_parkour.envs.registry import get_env_spec

_COURSE_KEYS = ("track", "length", "half_width", "checkpoint_spacing", "finish_offset",
                "level_difficulty", "height_scan", "randomize_obstacles", "dr_low",
                "dr_high", "action_scale", "race_mode", "path_preview", "preview_distances",
                "preview_mode")


def log_metrics(logger, metrics, step):
    for name, value in metrics.items():
        logger.log_scalar(name, value, step)


def configs_from_cfg(cfg):
    """ParkourConfig/RewardConfig/TerminationConfig from a Hydra cfg, registry-backed."""
    from automataleague_parkour.envs.parkour.config import RewardConfig, TerminationConfig

    level = 0
    if hasattr(cfg.env, "course"):
        _lvl = getattr(cfg.env.course, "level_difficulty", 0)
        level = 0 if _lvl is None else int(_lvl)
    course = get_env_spec(cfg.env.name).config(level)   # registry defaults (track, scan, scale)
    if hasattr(cfg.env, "course"):
        for k in _COURSE_KEYS:
            if hasattr(cfg.env.course, k):
                v = getattr(cfg.env.course, k)
                if v is not None:            # null keeps the registry/schedule value
                    setattr(course, k, v)

    rc = RewardConfig()
    if hasattr(cfg.env, "reward_weights"):
        for k in list(vars(rc).keys()):
            if hasattr(cfg.env.reward_weights, k):
                setattr(rc, k, getattr(cfg.env.reward_weights, k))

    tc = TerminationConfig()
    if hasattr(cfg.env, "termination"):
        for k in ("fall_height", "max_tilt_deg", "off_path"):
            if hasattr(cfg.env.termination, k):
                setattr(tc, k, getattr(cfg.env.termination, k))
    tc.max_episode_steps = cfg.env.max_episode_steps
    return course, rc, tc


def env_maker(cfg, num_envs=None, eval_mode=False):
    """Create a ParkourEnvWarp (already a TorchRL EnvBase).

    eval_mode forces obstacle DR off, so eval measures the clean nominal difficulty
    (factor 1.0) — the comparable, official course — while training randomizes.
    """
    from automataleague_parkour.envs.parkour.parkour_warp import ParkourEnvWarp

    device = cfg.network.device or "cuda"
    course, rc, tc = configs_from_cfg(cfg)
    if eval_mode:
        course.randomize_obstacles = False
    return ParkourEnvWarp(
        robot=cfg.env.robot,
        num_envs=num_envs if num_envs is not None else cfg.env.num_envs,
        device=device,
        cfg=course,
        reward_cfg=rc,
        term_cfg=tc,
        frame_skip=getattr(cfg.env, "frame_skip", 10),
    )


def apply_env_transforms(env, max_episode_steps):
    return TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            RewardSum(),
        ),
    )


def make_environment(cfg):
    """Make train and eval environments."""
    train_env = env_maker(cfg)
    train_env = apply_env_transforms(train_env, cfg.env.max_episode_steps)
    eval_env = env_maker(cfg, num_envs=1, eval_mode=True)   # eval on the clean nominal course
    eval_env = apply_env_transforms(eval_env, cfg.env.max_episode_steps)
    return train_env, eval_env


def rollout_video(policy, cfg, max_steps=None, policy_device="cuda", render_size=(480, 854),
                  stop_at_done=True):
    """Roll the deterministic policy on a single CPU env and return frames [T,H,W,3].

    With stop_at_done (default), the video ends at the episode's natural conclusion
    (finish / fall / off-path), so a completing policy yields a full-lap clip that
    isn't chopped mid-course by the step cap. `max_steps` is then just an upper bound.
    """
    from automataleague_parkour.envs.parkour.parkour_cpu import ParkourEnvCPU

    course, rc, tc = configs_from_cfg(cfg)
    env = ParkourEnvCPU(
        robot=cfg.env.robot, cfg=course, reward_cfg=rc, term_cfg=tc,
        render_size=render_size, frame_skip=getattr(cfg.env, "frame_skip", 10),
    )
    steps = max_steps or cfg.env.max_episode_steps
    obs = env.reset()
    frames = []
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(steps):
            td = TensorDict(
                {"observation": torch.as_tensor(obs, dtype=torch.float32,
                                                device=policy_device).unsqueeze(0)},
                batch_size=[1],
            )
            td = policy(td)
            action = td["action"].squeeze(0).cpu().numpy()
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc:
                if stop_at_done:
                    break
                obs = env.reset()
    return np.stack(frames)
