import math

import torch

from automataleague.envs.parkour.config import TerminationConfig
from automataleague.envs.parkour.termination import compute_termination
from automataleague.envs.parkour.state import ParkourState


def _state(z=0.46, y=0.0, roll=0.0, N=1):
    q = torch.tensor([[math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]]).repeat(N, 1)
    return ParkourState(
        base_pos=torch.tensor([[1.0, y, z]]).repeat(N, 1), base_quat=q,
        base_linvel_world=torch.zeros(N, 3), base_angvel_local=torch.zeros(N, 3),
        joint_pos=torch.zeros(N, 12), joint_vel=torch.zeros(N, 12),
    )


def test_fall_by_height():
    t, _, fell, _, oc = compute_termination(
        _state(z=0.1), torch.tensor([10]), torch.tensor([False]), 1.6, TerminationConfig())
    assert t.item() and fell.item() and oc.item() == 2


def test_fall_by_tilt():
    t, _, fell, _, oc = compute_termination(
        _state(roll=math.radians(80)), torch.tensor([10]), torch.tensor([False]), 1.6,
        TerminationConfig())
    assert fell.item() and oc.item() == 2


def test_off_path():
    t, _, _, off, oc = compute_termination(
        _state(y=2.0), torch.tensor([10]), torch.tensor([False]), 1.6, TerminationConfig())
    assert t.item() and off.item() and oc.item() == 3


def test_success_overrides():
    t, _, _, _, oc = compute_termination(
        _state(), torch.tensor([10]), torch.tensor([True]), 1.6, TerminationConfig())
    assert t.item() and oc.item() == 1


def test_truncation():
    _, trunc, _, _, _ = compute_termination(
        _state(), torch.tensor([1000]), torch.tensor([False]), 1.6, TerminationConfig())
    assert trunc.item()
