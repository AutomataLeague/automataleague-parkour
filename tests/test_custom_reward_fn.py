"""A custom reward_fn passed to make_env replaces the default reward."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

from automataleague_parkour import make_env


def _const_reward(state, prev_dist, cur_dist, reached_intermediate, reached_finish,
                  fell, off_path, action, nominal_height, rc, forward_vel=None):
    # ignore everything, return a fixed sentinel reward + empty components
    r = torch.full_like(prev_dist, 42.0)
    return r, {}


def test_custom_reward_fn_overrides_default():
    env = make_env("parkour-1", robot="spot", level=0, backend="cpu", reward_fn=_const_reward)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(env.robot.n_joints, dtype=np.float32))
    assert float(reward) == 42.0


def test_default_reward_when_none():
    env = make_env("parkour-1", robot="spot", level=0, backend="cpu")   # no reward_fn
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(env.robot.n_joints, dtype=np.float32))
    assert float(reward) != 42.0        # default reward, not the sentinel
