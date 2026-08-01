import torch

from automataleague_parkour.envs.parkour.navigation import centerline_project


def _cumlen(pl):
    seg = torch.linalg.norm(pl[1:] - pl[:-1], dim=-1)
    return torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])


def test_project_straight_x_axis():
    pl = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    cum = _cumlen(pl)
    pts = torch.tensor([[3.0, 0.5], [7.0, -0.25]])  # left of travel, right of travel
    s0, lat, tang = centerline_project(pts, pl, cum)
    assert torch.allclose(s0, torch.tensor([3.0, 7.0]), atol=1e-4)
    assert torch.allclose(lat, torch.tensor([0.5, -0.25]), atol=1e-4)  # signed
    assert torch.allclose(tang, torch.tensor([[1.0, 0.0], [1.0, 0.0]]), atol=1e-4)


def test_project_bent_polyline_nonfirst_segment():
    # A=(0,0) -> B=(10,0) -> C=(10,10): the nearest projection for both points below
    # lands on segment B->C (index 1, not 0), so s0 must accumulate seg0's full
    # length (10) before adding the partial distance into seg1.
    pl = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    cum = _cumlen(pl)
    pts = torch.tensor([[12.0, 5.0], [8.0, 5.0]])  # right of travel, left of travel
    s0, lat, tang = centerline_project(pts, pl, cum)
    assert torch.allclose(s0, torch.tensor([15.0, 15.0]), atol=1e-4)
    assert torch.allclose(lat, torch.tensor([-2.0, 2.0]), atol=1e-4)  # signed
    assert torch.allclose(tang, torch.tensor([[0.0, 1.0], [0.0, 1.0]]), atol=1e-4)
