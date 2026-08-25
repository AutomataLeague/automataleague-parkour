import numpy as np

from automataleague_parkour.envs.parkour.config import ParkourConfig
from automataleague_parkour.envs.parkour.parkour_cpu import ParkourEnvCPU

# Both optional sensors off, so the observation is the bare robot.obs_dim (49). This is
# NOT the default any more (parkour-1 ships both sensors on), so it is stated explicitly:
# these tests pin the proprioceptive block, independent of what the defaults do.
_BARE = ParkourConfig(track_perception="none", height_scan=False)


def test_reset_returns_correct_obs_dim():
    env = ParkourEnvCPU(robot="spot", cfg=_BARE)
    obs = env.reset()
    assert obs.shape == (env.robot.obs_dim,)
    assert np.isfinite(obs).all()


def test_step_runs_and_returns_finite():
    env = ParkourEnvCPU(robot="spot", cfg=_BARE)
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


# --- seeded, noisy reset ----------------------------------------------------
# Evaluation has to start from the same distribution training resets into, and until
# `reset` took a seed the only way to get a noisy start was for the caller to reach in
# and mutate `env.info.home_qpos`. See training-recipe.md, finding 5.

def test_reset_is_deterministic_by_default():
    env = ParkourEnvCPU(robot="spot", cfg=_BARE)
    assert np.array_equal(env.reset(), env.reset())


def test_reset_with_the_same_seed_repeats():
    env = ParkourEnvCPU(robot="spot", cfg=_BARE)
    a = env.reset(seed=7, pos_noise=0.05, joint_noise=0.05)
    b = env.reset(seed=7, pos_noise=0.05, joint_noise=0.05)
    assert np.array_equal(a, b)


def test_reset_with_different_seeds_differs():
    env = ParkourEnvCPU(robot="spot", cfg=_BARE)
    a = env.reset(seed=1, pos_noise=0.05, joint_noise=0.05)
    b = env.reset(seed=2, pos_noise=0.05, joint_noise=0.05)
    assert not np.array_equal(a, b)
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_reset_noise_perturbs_the_actuated_joints_and_base():
    env = ParkourEnvCPU(robot="spot", cfg=_BARE)
    env.reset()
    clean = env.data.qpos.copy()
    env.reset(seed=3, pos_noise=0.05, joint_noise=0.05)
    noisy = env.data.qpos.copy()
    ja, ba = env.info.joint_qposadr, env.info.base_qposadr
    assert not np.array_equal(clean[ja], noisy[ja])          # joints moved
    assert not np.array_equal(clean[ba:ba + 2], noisy[ba:ba + 2])   # base xy moved
    assert np.array_equal(clean[ba + 2], noisy[ba + 2])      # base height untouched


def test_reset_noise_is_opt_in_per_axis():
    env = ParkourEnvCPU(robot="spot", cfg=_BARE)
    env.reset()
    clean = env.data.qpos.copy()
    env.reset(seed=4, joint_noise=0.05)                       # no pos_noise
    noisy = env.data.qpos.copy()
    ba = env.info.base_qposadr
    assert np.array_equal(clean[ba:ba + 3], noisy[ba:ba + 3])
    assert not np.array_equal(clean[env.info.joint_qposadr], noisy[env.info.joint_qposadr])
