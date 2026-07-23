import torch

from automataleague.envs.parkour import build_parkour_model
from automataleague.envs.parkour.state import extract_state


def test_extract_state_home_pose():
    model, info = build_parkour_model("spot")
    qpos = torch.tensor(info.home_qpos, dtype=torch.float32).unsqueeze(0)  # [1, nq]
    qvel = torch.zeros(1, model.nv, dtype=torch.float32)
    st = extract_state(qpos, qvel, info)

    assert st.base_pos.shape == (1, 3)
    assert st.base_quat.shape == (1, 4)
    assert st.joint_pos.shape == (1, info.robot.n_joints)
    # Base at spawn/nominal height.
    assert torch.allclose(st.base_pos[0], torch.tensor([1.0, 0.0, 0.46]), atol=1e-5)
    # Joints at home stance (fl=0,1.04,-1.8 ...).
    assert torch.allclose(st.joint_pos[0, :3], torch.tensor([0.0, 1.04, -1.8]), atol=1e-4)


def test_extract_state_batched():
    model, info = build_parkour_model("spot")
    N = 8
    qpos = torch.tensor(info.home_qpos, dtype=torch.float32).repeat(N, 1)
    qvel = torch.zeros(N, model.nv, dtype=torch.float32)
    st = extract_state(qpos, qvel, info)
    assert st.base_pos.shape == (N, 3)
    assert st.joint_vel.shape == (N, info.robot.n_joints)
