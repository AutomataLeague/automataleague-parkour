import math

import torch

from automataleague_parkour.envs.parkour import spatial


def test_projected_gravity_upright(identity_quat):
    g = spatial.projected_gravity(identity_quat)
    assert torch.allclose(g, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6)


def test_projected_gravity_rolled_90deg():
    # 90-degree roll about x: world-down maps to body +y (or -y). |z| ~ 0.
    q = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])
    g = spatial.projected_gravity(q)
    assert abs(g[0, 2].item()) < 1e-6
    assert abs(g[0, 1].abs().item() - 1.0) < 1e-6


def test_tilt_angle_upright_is_zero(identity_quat):
    assert spatial.tilt_angle(identity_quat).item() < 1e-6


def test_tilt_angle_90deg():
    q = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])
    assert abs(spatial.tilt_angle(q).item() - math.pi / 2) < 1e-5


def test_quat_rotate_inverse_identity(identity_quat):
    v = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(spatial.quat_rotate_inverse(identity_quat, v), v, atol=1e-6)


def test_yaw_from_quat_90deg():
    q = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
    assert abs(spatial.yaw_from_quat(q).item() - math.pi / 2) < 1e-5
