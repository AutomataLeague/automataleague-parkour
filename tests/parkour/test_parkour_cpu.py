import numpy as np

from automataleague_parkour.envs.parkour.parkour_cpu import ParkourEnvCPU


def test_reset_returns_correct_obs_dim():
    env = ParkourEnvCPU(robot="spot")
    obs = env.reset()
    assert obs.shape == (env.robot.obs_dim,)
    assert np.isfinite(obs).all()


def test_step_runs_and_returns_finite():
    env = ParkourEnvCPU(robot="spot")
    env.reset()
    obs, rew, term, trunc, info = env.step(np.zeros(env.robot.n_joints, dtype=np.float32))
    assert obs.shape == (env.robot.obs_dim,)
    assert np.isfinite(rew)
    assert "outcome" in info


def test_episode_terminates_within_max_steps():
    env = ParkourEnvCPU(robot="spot")
    env.reset()
    done = False
    for _ in range(env.term_cfg.max_episode_steps + 5):
        _, _, term, trunc, _ = env.step(np.zeros(env.robot.n_joints, dtype=np.float32))
        if term or trunc:
            done = True
            break
    assert done  # zero action -> Spot holds/collapses then truncates or falls


def test_render_produces_image():
    env = ParkourEnvCPU(robot="spot", render_size=(240, 320))
    env.reset()
    img = env.render()
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
