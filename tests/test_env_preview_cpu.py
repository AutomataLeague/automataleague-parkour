import torch
from automataleague_parkour import make_env
from automataleague_parkour.envs.parkour.path_preview import preview_dim


def test_cpu_obs_dim_grows_with_centerline_preview():
    off = make_env("parkour-1", robot="spot", level=0, backend="cpu", track_perception="none")
    on = make_env("parkour-1", robot="spot", level=0, backend="cpu",
                  track_perception="centerline", preview_distances=(1.5, 3.0, 4.5, 6.0))
    d_off = off.observation_spec["observation"].shape[-1]
    d_on = on.observation_spec["observation"].shape[-1]
    assert d_on == d_off + 9
    # ParkourEnvCPU.reset() is Gym-style (plain numpy, not a TensorDict); see
    # tests/parkour/test_parkour_cpu.py::test_reset_returns_correct_obs_dim.
    obs = on.reset()
    assert obs.shape[-1] == d_on
    assert torch.isfinite(torch.as_tensor(obs)).all()


def test_cpu_obs_dim_grows_with_boundary_preview():
    dists = (1.5, 3.0, 4.5, 6.0)
    off = make_env("parkour-1", robot="spot", level=0, backend="cpu", track_perception="none")
    on = make_env("parkour-1", robot="spot", level=0, backend="cpu",
                  track_perception="boundary", preview_distances=dists)
    d_off = off.observation_spec["observation"].shape[-1]
    d_on = on.observation_spec["observation"].shape[-1]
    assert d_on == d_off + preview_dim(dists, "boundary")   # 4K + 1
    obs = on.reset()
    assert obs.shape[-1] == d_on
    assert torch.isfinite(torch.as_tensor(obs)).all()


def test_cpu_boundary_is_the_default_perception():
    # parkour-1 defaults to track_perception="boundary"; asking for it explicitly must
    # not change obs_dim.
    dists = (1.5, 3.0, 4.5, 6.0)
    default = make_env("parkour-1", robot="spot", level=0, backend="cpu",
                       preview_distances=dists)
    explicit = make_env("parkour-1", robot="spot", level=0, backend="cpu",
                        track_perception="boundary", preview_distances=dists)
    assert (default.observation_spec["observation"].shape[-1]
            == explicit.observation_spec["observation"].shape[-1])


def test_perception_none_is_scan_only_baseline():
    # With perception off, parkour-1 is robot.obs_dim (49) + height scan (12) = 61.
    # parkour-1's registry default turns the height scan on, so the perception-off
    # baseline for this env_id is 61, not the bare robot.obs_dim of 49.
    off = make_env("parkour-1", robot="spot", level=0, backend="cpu", track_perception="none")
    assert off.observation_spec["observation"].shape[-1] == 61
