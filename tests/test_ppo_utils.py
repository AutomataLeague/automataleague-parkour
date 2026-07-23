import os
import sys

import torch

# train_ppo.py imports utils_ppo flat (run from training/ via hydra); match that here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))


class _Cfg:
    class network:
        activation = "elu"


def test_get_activation_supports_elu():
    from utils_ppo import get_activation

    assert get_activation(_Cfg()) is torch.nn.ELU


def test_utils_ppo_imports():
    import utils_ppo as u

    assert hasattr(u, "make_ppo_models") and hasattr(u, "make_environment")
    assert hasattr(u, "rollout_video")
