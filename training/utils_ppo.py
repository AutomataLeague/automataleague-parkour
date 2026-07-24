"""Utilities for GPU-parallel PPO training with ParkourEnvWarp.

Ported from smaller_projects/spaceX/training/utils_ppo.py. The actor/critic builders
are task-agnostic (sized from env specs). Env construction targets the parkour Warp
env; eval videos are rendered by looping the CPU env with the policy (no gym dep).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn
from tensordict import TensorDict
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import Compose, ExplorationType, TransformedEnv, set_exploration_type
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator


# ====================================================================
# General utils
# ====================================================================


def log_metrics(logger, metrics, step):
    for name, value in metrics.items():
        logger.log_scalar(name, value, step)


def get_activation(cfg):
    return {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "leaky_relu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
    }[cfg.network.activation]


# ====================================================================
# Environment utils
# ====================================================================


def _configs_from_cfg(cfg):
    """Build ParkourConfig / RewardConfig / TerminationConfig from a Hydra cfg."""
    from automataleague.envs.parkour.config import (
        ParkourConfig,
        RewardConfig,
        TerminationConfig,
    )

    course = ParkourConfig()
    if hasattr(cfg.env, "course"):
        for k in ("track", "length", "half_width", "checkpoint_spacing",
                  "finish_offset", "level_difficulty", "height_scan",
                  "randomize_obstacles", "dr_low", "dr_high", "action_scale"):
            if hasattr(cfg.env.course, k):
                setattr(course, k, getattr(cfg.env.course, k))

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
    from automataleague.envs.parkour.parkour_warp import ParkourEnvWarp

    device = cfg.network.device or "cuda"
    course, rc, tc = _configs_from_cfg(cfg)
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
    from automataleague.envs.parkour.parkour_cpu import ParkourEnvCPU

    course, rc, tc = _configs_from_cfg(cfg)
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


# ====================================================================
# PPO models
# ====================================================================


def make_ppo_models(cfg, train_env, device):
    """Actor (MLP -> TanhNormal ProbabilisticActor) and critic (MLP -> ValueOperator)."""
    input_shape = train_env.observation_spec["observation"].shape
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]

    num_outputs = action_spec.shape[-1]
    activation_class = get_activation(cfg)
    hidden_sizes = cfg.network.hidden_sizes

    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=activation_class,
        out_features=num_outputs,
        num_cells=hidden_sizes,
        device=device,
    )
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(num_outputs, scale_lb=1e-8).to(device),
    )

    policy_module = ProbabilisticActor(
        TensorDictModule(module=policy_mlp, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=activation_class,
        out_features=1,
        num_cells=hidden_sizes,
        device=device,
    )
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    value_module = ValueOperator(value_mlp, in_keys=["observation"])
    return policy_module, value_module
