import torch

from automataleague.envs.parkour import navigation as nav
from automataleague.envs.parkour.state import ParkourState


def _state_at(x, y, yaw=0.0, N=1):
    import math
    q = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]).repeat(N, 1)
    return ParkourState(
        base_pos=torch.tensor([[x, y, 0.46]]).repeat(N, 1),
        base_quat=q,
        base_linvel_world=torch.zeros(N, 3),
        base_angvel_local=torch.zeros(N, 3),
        joint_pos=torch.zeros(N, 12),
        joint_vel=torch.zeros(N, 12),
    )


def test_distance_and_heading_straight_ahead():
    cps = torch.tensor([[3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
    st = _state_at(1.0, 0.0, yaw=0.0)
    to_cp, dist, herr = nav.checkpoint_geometry(st, cps, torch.tensor([0]))
    assert abs(dist.item() - 2.0) < 1e-5           # 3.0 - 1.0
    assert abs(herr.item()) < 1e-5                 # checkpoint dead ahead
    assert to_cp[0, 0].item() > 0                  # forward is +x in base frame


def test_advance_on_reach():
    cps = torch.tensor([[3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
    dist = torch.tensor([0.3])                     # within radius 0.5
    idx, inter, fin = nav.advance_checkpoints(dist, torch.tensor([0]), 0.5, num_cp=3)
    assert idx.item() == 1 and inter.item() and not fin.item()


def test_finish_detected():
    dist = torch.tensor([0.2])
    idx, inter, fin = nav.advance_checkpoints(dist, torch.tensor([2]), 0.5, num_cp=3)
    assert fin.item() and idx.item() == 2          # index stays clamped at last


def test_no_advance_when_far():
    dist = torch.tensor([1.0])
    idx, inter, fin = nav.advance_checkpoints(dist, torch.tensor([0]), 0.5, num_cp=3)
    assert idx.item() == 0 and not inter.item() and not fin.item()
