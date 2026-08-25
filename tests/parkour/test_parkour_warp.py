import pytest
import torch


@pytest.mark.gpu
def test_warp_env_reset_and_step():
    from automataleague_parkour.envs.parkour.parkour_warp import ParkourEnvWarp

    env = ParkourEnvWarp(robot="spot", num_envs=64, device="cuda")
    td = env.reset()
    assert td["observation"].shape == (64, env.robot.obs_dim)
    td["action"] = torch.zeros(64, env.robot.n_joints, device="cuda")
    td = env.step(td)
    assert td["next", "observation"].shape == (64, env.robot.obs_dim)
    assert torch.isfinite(td["next", "reward"]).all()


@pytest.mark.gpu
def test_warp_env_throughput():
    import time

    from automataleague_parkour.envs.parkour.parkour_warp import ParkourEnvWarp

    N = 512
    env = ParkourEnvWarp(robot="spot", num_envs=N, device="cuda")
    td = env.reset()
    act = torch.zeros(N, env.robot.n_joints, device="cuda")
    for _ in range(10):
        td["action"] = act
        td = env.step(td)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        td["action"] = act
        td = env.step(td)
    torch.cuda.synchronize()
    sps = 100 * N / (time.time() - t0)
    print(f"throughput: {sps:,.0f} steps/sec ({N} envs)")
    # Deliberately far below any GPU we have measured: this asserts the batched env
    # still steps, not that a particular device is fast.
    assert sps > 2_000
