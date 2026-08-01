import torch
from automataleague_parkour import make_env


def test_cpu_obs_dim_grows_with_preview():
    off = make_env("parkour-1", robot="spot", level=0, backend="cpu")
    on = make_env("parkour-1", robot="spot", level=0, backend="cpu",
                  path_preview=True, preview_distances=(1.5, 3.0, 4.5, 6.0))
    d_off = off.observation_spec["observation"].shape[-1]
    d_on = on.observation_spec["observation"].shape[-1]
    assert d_on == d_off + 9
    # ParkourEnvCPU.reset() is Gym-style (plain numpy, not a TensorDict); see
    # tests/parkour/test_parkour_cpu.py::test_reset_returns_correct_obs_dim.
    obs = on.reset()
    assert obs.shape[-1] == d_on
    assert torch.isfinite(torch.as_tensor(obs)).all()


def test_flag_off_is_unchanged():
    off = make_env("parkour-1", robot="spot", level=0, backend="cpu")
    # parkour-1's registry default already turns height_scan on (robot.obs_dim=49 +
    # SCAN_N=12), so the byte-identical-to-before baseline for this env_id is 61, not
    # the bare robot.obs_dim=49 (that figure is what a raw ParkourEnvCPU(robot="spot")
    # gets, since its default ParkourConfig has height_scan=False).
    assert off.observation_spec["observation"].shape[-1] == 61   # baseline preserved
