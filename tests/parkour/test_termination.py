import math

import torch

from automataleague_parkour.envs.parkour.config import TerminationConfig
from automataleague_parkour.envs.parkour.state import ParkourState
from automataleague_parkour.envs.parkour.termination import compute_termination


def _state(z=0.46, y=0.0, roll=0.0, N=1):
    q = torch.tensor([[math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]]).repeat(N, 1)
    return ParkourState(
        base_pos=torch.tensor([[1.0, y, z]]).repeat(N, 1), base_quat=q,
        base_linvel_world=torch.zeros(N, 3), base_angvel_local=torch.zeros(N, 3),
        joint_pos=torch.zeros(N, 12), joint_vel=torch.zeros(N, 12),
    )


def _term(state, step, finish, lateral):
    return compute_termination(
        state, torch.tensor([step]), torch.tensor([finish]),
        torch.tensor([lateral]), 1.6, TerminationConfig())


def test_fall_by_height():
    t, _, fell, _, oc = _term(_state(z=0.1), 10, False, 0.0)
    assert t.item() and fell.item() and oc.item() == 2


def test_fall_by_tilt():
    t, _, fell, _, oc = _term(_state(roll=math.radians(80)), 10, False, 0.0)
    assert fell.item() and oc.item() == 2


def test_off_path():
    t, _, _, off, oc = _term(_state(y=2.0), 10, False, 2.0)
    assert t.item() and off.item() and oc.item() == 3


def test_success_overrides():
    t, _, _, _, oc = _term(_state(), 10, True, 0.0)
    assert t.item() and oc.item() == 1


def test_truncation():
    _, trunc, _, _, _ = _term(_state(), 1000, False, 0.0)
    assert trunc.item()
