import torch

from automataleague_parkour.envs.parkour import build_parkour_model
from automataleague_parkour.envs.parkour.navigation import checkpoint_geometry
from automataleague_parkour.envs.parkour.observation import build_observation
from automataleague_parkour.envs.parkour.state import extract_state


def test_observation_shape_matches_robot_obs_dim():
    model, info = build_parkour_model("spot")
    N = 4
    qpos = torch.tensor(info.home_qpos, dtype=torch.float32).repeat(N, 1)
    qvel = torch.zeros(N, model.nv, dtype=torch.float32)
    st = extract_state(qpos, qvel, info)
    cps = torch.tensor(info.checkpoints_xy)
    cp_idx = torch.zeros(N, dtype=torch.long)
    to_cp, dist, herr = checkpoint_geometry(st, cps, cp_idx)
    home = torch.tensor(info.robot.home_joint_qpos)
    prev_action = torch.zeros(N, info.robot.n_joints)

    obs = build_observation(st, to_cp, dist, herr, prev_action, home)
    assert obs.shape == (N, info.robot.obs_dim)      # 49 for Spot
    assert torch.isfinite(obs).all()


def test_observation_joint_pos_relative_to_home_is_zero_at_home():
    model, info = build_parkour_model("spot")
    qpos = torch.tensor(info.home_qpos, dtype=torch.float32).unsqueeze(0)
    qvel = torch.zeros(1, model.nv, dtype=torch.float32)
    st = extract_state(qpos, qvel, info)
    cps = torch.tensor(info.checkpoints_xy)
    to_cp, dist, herr = checkpoint_geometry(st, cps, torch.zeros(1, dtype=torch.long))
    home = torch.tensor(info.robot.home_joint_qpos)
    obs = build_observation(st, to_cp, dist, herr, torch.zeros(1, 12), home)
    # joint_pos - home occupies indices [9 : 9+12]; should be ~0 at home stance.
    assert torch.allclose(obs[0, 9:21], torch.zeros(12), atol=1e-4)
