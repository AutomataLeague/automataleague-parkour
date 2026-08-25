"""Warm-start compatibility: a policy trained WITHOUT the track-preview sensor
must load into the preview-augmented network, with the new input columns
zero-padded so the warm-started policy initially ignores the preview signal
and keeps its learned gait.
"""
import pytest

pytest.importorskip("torchrl", reason="torchrl ships in the `train` extra")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torchrl.data import Bounded, Composite, Unbounded  # noqa: E402

from automataleague_parkour.robots import get_robot  # noqa: E402
from automataleague_parkour.training.models import (  # noqa: E402
    _pad_obs_input,
    build_actor,
    make_ppo_models,
)

HIDDEN_SIZES = [512, 256, 128]
SMALL_OBS = 49  # spot, height scan off, no path preview
PREVIEW_DIM = 9  # 4 lookahead points (x, y) plus the signed lateral offset
BIG_OBS = SMALL_OBS + PREVIEW_DIM  # 58, preview appended behind the base obs
FIRST_LAYER_WEIGHT_KEY = "module.0.module.0.0.weight"

CFG = OmegaConf.create({
    "network": {"hidden_sizes": HIDDEN_SIZES, "activation": "elu", "device": "cpu"},
    "env": {"robot": "spot", "course": {"height_scan": False}},
})


def _build_big_actor(device):
    """Build an actor at BIG_OBS width via a stub env spec, mirroring how
    build_actor fakes a spec, but with the preview-sized observation."""

    class _Stub:
        pass

    stub = _Stub()
    stub.batch_size = torch.Size([1])
    stub.observation_spec = Composite(
        observation=Unbounded(shape=(1, BIG_OBS), device=device), shape=(1,)
    )
    action_spec = Composite(
        action=Bounded(
            low=-torch.ones(1, 12, device=device),
            high=torch.ones(1, 12, device=device),
            device=device,
        ),
        shape=(1,),
    )
    stub.action_spec = action_spec["action"]
    actor, _ = make_ppo_models(CFG, stub, device)
    return actor


def test_pre_preview_policy_warmstarts_into_preview_network():
    device = torch.device("cpu")

    small = build_actor(CFG, get_robot("spot"), device)
    small_sd = small.state_dict()
    small_weight = small_sd[FIRST_LAYER_WEIGHT_KEY]
    assert small_weight.shape == (512, SMALL_OBS)

    padded = _pad_obs_input(small_sd, cur_obs=BIG_OBS, hidden_sizes=HIDDEN_SIZES)

    big = _build_big_actor(device)
    big.load_state_dict(padded)  # must not raise: shapes now line up

    loaded_weight = big.state_dict()[FIRST_LAYER_WEIGHT_KEY]
    assert loaded_weight.shape == (512, BIG_OBS)

    # The original 49 columns carry over unchanged...
    assert torch.equal(loaded_weight[:, :SMALL_OBS], small_weight)
    # ...and the 9 new preview columns are zero, so the warm-started policy
    # initially ignores the preview and keeps its learned gait.
    new_cols = loaded_weight[:, SMALL_OBS:BIG_OBS]
    assert new_cols.shape == (512, PREVIEW_DIM)
    assert torch.all(new_cols == 0)
