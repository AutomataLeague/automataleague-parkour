import torch

from automataleague_parkour.envs.parkour.observation import build_observation
from automataleague_parkour.envs.parkour.state import ParkourState


def _state(n=2, nj=12):
    z3 = torch.zeros(n, 3)
    q = torch.zeros(n, 4)
    q[:, 0] = 1
    return ParkourState(base_pos=z3, base_quat=q, base_linvel_world=z3,
                         base_angvel_local=z3, joint_pos=torch.zeros(n, nj),
                         joint_vel=torch.zeros(n, nj))


def test_preview_appended_last_and_leading_cols_stable():
    st = _state()
    nj = 12
    home = torch.zeros(nj)
    pa = torch.zeros(2, nj)
    to_cp = torch.zeros(2, 2)
    dist = torch.zeros(2)
    herr = torch.zeros(2)
    base = build_observation(st, to_cp, dist, herr, pa, home)
    prev = torch.arange(9.).repeat(2, 1)  # PREVIEW_N=9
    full = build_observation(st, to_cp, dist, herr, pa, home, track_preview=prev)
    assert full.shape[-1] == base.shape[-1] + 9
    assert torch.allclose(full[:, :base.shape[-1]], base)  # leading columns unchanged
    assert torch.allclose(full[:, -9:], prev)
