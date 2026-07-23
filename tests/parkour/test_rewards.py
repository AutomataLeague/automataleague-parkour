import torch

from automataleague.envs.parkour.config import RewardConfig
from automataleague.envs.parkour.rewards import compute_reward
from automataleague.envs.parkour.state import ParkourState


def _upright_state(N=1):
    return ParkourState(
        base_pos=torch.tensor([[1.0, 0.0, 0.46]]).repeat(N, 1),
        base_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(N, 1),
        base_linvel_world=torch.zeros(N, 3),
        base_angvel_local=torch.zeros(N, 3),
        joint_pos=torch.zeros(N, 12),
        joint_vel=torch.zeros(N, 12),
    )


def _kw(**over):
    base = dict(
        state=_upright_state(),
        prev_dist=torch.tensor([2.0]),
        cur_dist=torch.tensor([2.0]),
        reached_intermediate=torch.tensor([False]),
        reached_finish=torch.tensor([False]),
        fell=torch.tensor([False]),
        off_path=torch.tensor([False]),
        action=torch.zeros(1, 12),
        nominal_height=0.46,
        rc=RewardConfig(),
    )
    base.update(over)
    return base


def test_progress_positive_when_distance_decreases():
    r, _ = compute_reward(**_kw(prev_dist=torch.tensor([2.0]), cur_dist=torch.tensor([1.5])))
    r0, _ = compute_reward(**_kw())
    assert r.item() > r0.item()


def test_checkpoint_bonus_applied():
    r_hit, _ = compute_reward(**_kw(reached_intermediate=torch.tensor([True])))
    r_no, _ = compute_reward(**_kw())
    assert r_hit.item() - r_no.item() >= RewardConfig().checkpoint - 1e-3


def test_fall_penalty_negative():
    r, _ = compute_reward(**_kw(fell=torch.tensor([True])))
    assert r.item() < 0


def test_off_path_penalty_negative():
    r, _ = compute_reward(**_kw(off_path=torch.tensor([True])))
    assert r.item() < 0


def test_reward_is_finite_and_shaped():
    r, comps = compute_reward(**_kw())
    assert torch.isfinite(r).all()
    assert r.shape == (1, 1)
    assert "progress" in comps and "upright" in comps
