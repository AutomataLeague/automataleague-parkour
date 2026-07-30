import torch

from automataleague_parkour.envs.parkour.config import RewardConfig
from automataleague_parkour.envs.parkour.rewards import compute_reward
from automataleague_parkour.envs.parkour.state import ParkourState


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


def test_forward_velocity_reward_increases_reward():
    r_move, comps = compute_reward(**_kw(forward_vel=torch.tensor([1.0])))
    r_still, _ = compute_reward(**_kw(forward_vel=torch.tensor([0.0])))
    assert r_move.item() > r_still.item()
    assert "forward" in comps


def test_forward_velocity_reward_saturates_at_target_speed():
    # Beyond target_speed the forward reward should not grow (no lunge incentive).
    r_at, _ = compute_reward(**_kw(forward_vel=torch.tensor([1.0])))
    r_over, _ = compute_reward(**_kw(forward_vel=torch.tensor([5.0])))
    assert abs(r_at.item() - r_over.item()) < 1e-5


def test_reward_is_finite_and_shaped():
    r, comps = compute_reward(**_kw())
    assert torch.isfinite(r).all()
    assert r.shape == (1, 1)
    assert "progress" in comps and "upright" in comps
