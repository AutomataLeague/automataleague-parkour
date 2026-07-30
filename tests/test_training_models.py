import torch
from omegaconf import OmegaConf
from automataleague_parkour.robots import get_robot
from automataleague_parkour.training.models import build_actor, _pad_obs_input


def _cfg(scan):
    return OmegaConf.create({
        "env": {"robot": "spot", "course": {"height_scan": scan}},
        "network": {"activation": "elu", "hidden_sizes": [64, 64], "device": "cpu"},
    })


def test_build_actor_dims_and_forward():
    robot = get_robot("spot")
    dev = torch.device("cpu")
    actor = build_actor(_cfg(True), robot, dev)          # scan on -> obs 61
    from tensordict import TensorDict
    td = actor(TensorDict({"observation": torch.zeros(1, robot.obs_dim + 12)}, [1]))
    assert td["action"].shape == (1, robot.action_dim)


def test_pad_obs_input_zero_pads_first_layer():
    sd = {"0.weight": torch.ones(8, 49), "2.weight": torch.ones(8, 64)}
    out = _pad_obs_input(sd, 61, hidden_sizes=[64])
    assert out["0.weight"].shape == (8, 61)          # padded 49 -> 61
    assert out["0.weight"][:, 49:].sum() == 0        # new cols zeroed
    assert out["2.weight"].shape == (8, 64)          # hidden width untouched


def test_get_activation_maps_elu():
    import torch
    from automataleague_parkour.training.models import get_activation
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"network": {"activation": "elu"}})
    assert get_activation(cfg) is torch.nn.ELU
