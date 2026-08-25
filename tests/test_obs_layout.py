"""The observation layout, and the warm-start guard built on it.

`_pad_obs_input` zero-pads a warm-start's first layer so a policy trained with a
smaller observation loads into a sensored network. That is only sound while the
observation is *append-only*: the old columns have to still mean what they meant.

The moment a second optional block existed, "obs got wider" stopped implying
"the leading columns are unchanged". Going from
`[proprio | height_scan]` (61) to `[proprio | track_preview]` (66) is wider *and*
scrambled, and width-only padding accepted it silently. These tests pin the
layout as the thing that must match.
"""
import importlib.util

import pytest
import torch

from automataleague_parkour.envs.parkour.observation import obs_layout, obs_width
from automataleague_parkour.envs.parkour.path_preview import resolve_perception
from automataleague_parkour.robots import get_robot

SPOT = get_robot("spot")
DISTS = (1.5, 3.0, 4.5, 6.0)


def _course(scan, perception):
    return {"height_scan": scan, "track_perception": perception, "preview_distances": DISTS}


# --- the layout itself ------------------------------------------------------

@pytest.mark.parametrize("scan,perception,expected", [
    (False, "none",       (("proprio", 49),)),
    (True,  "none",       (("proprio", 49), ("height_scan", 12))),
    (False, "centerline", (("proprio", 49), ("track_preview", 9))),
    (False, "boundary",   (("proprio", 49), ("track_preview", 17))),
    (True,  "boundary",   (("proprio", 49), ("height_scan", 12), ("track_preview", 17))),
])
def test_obs_layout_blocks(scan, perception, expected):
    assert obs_layout(_course(scan, perception), SPOT) == expected
    assert obs_width(_course(scan, perception), SPOT) == sum(w for _, w in expected)


@pytest.mark.parametrize("level", [0, 2, 4])
@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("perception", ["none", "centerline", "boundary"])
def test_obs_layout_matches_the_live_env(level, scan, perception):
    """The layout is only useful if it is the same width the env actually emits."""
    from automataleague_parkour import make_env
    env = make_env("parkour-1", robot="spot", level=level, backend="cpu",
                   height_scan=scan, track_perception=perception,
                   preview_distances=DISTS)
    assert obs_width(_course(scan, perception), SPOT) == len(env.reset())


# --- the warm-start guard ---------------------------------------------------

def test_appending_a_sensor_is_a_valid_warm_start():
    from automataleague_parkour.training.models import check_obs_layout_compatible
    check_obs_layout_compatible(                       # must not raise
        obs_layout(_course(False, "none"), SPOT),
        obs_layout(_course(True, "none"), SPOT),
    )


def test_identical_layouts_are_a_valid_warm_start():
    from automataleague_parkour.training.models import check_obs_layout_compatible
    lay = obs_layout(_course(True, "boundary"), SPOT)
    check_obs_layout_compatible(lay, lay)              # must not raise


def test_swapping_a_sensor_is_rejected():
    """The regression: L0 completion (scan) -> racing (preview) is 61 -> 66, wider
    but scrambled. Width-only padding accepted it; the layout check must not."""
    from automataleague_parkour.training.models import check_obs_layout_compatible
    src = obs_layout(_course(True, "none"), SPOT)      # [proprio | height_scan] = 61
    dst = obs_layout(_course(False, "boundary"), SPOT)  # [proprio | track_preview] = 66
    assert sum(w for _, w in src) < sum(w for _, w in dst)   # wider, so padding "works"
    with pytest.raises(ValueError, match="height_scan"):
        check_obs_layout_compatible(src, dst)


def test_dropping_a_sensor_is_rejected():
    from automataleague_parkour.training.models import check_obs_layout_compatible
    with pytest.raises(ValueError):
        check_obs_layout_compatible(
            obs_layout(_course(True, "boundary"), SPOT),
            obs_layout(_course(True, "none"), SPOT),
        )


def test_changing_a_preview_mode_is_rejected():
    """centerline (9) -> boundary (17) is the same block, different meaning."""
    from automataleague_parkour.training.models import check_obs_layout_compatible
    with pytest.raises(ValueError, match="track_preview"):
        check_obs_layout_compatible(
            obs_layout(_course(False, "centerline"), SPOT),
            obs_layout(_course(False, "boundary"), SPOT),
        )


# --- perception knob: no silent fallbacks -----------------------------------

@pytest.mark.parametrize("value,expected", [
    ("none",       (False, "centerline")),
    ("centerline", (True, "centerline")),
    ("boundary",   (True, "boundary")),
    ("Boundary",   (True, "boundary")),
    ("boundaries", (True, "boundary")),   # the legacy spelling stays accepted
])
def test_resolve_perception_accepted_values(value, expected):
    assert resolve_perception({"track_perception": value}) == expected


@pytest.mark.parametrize("typo", ["bounary", "boundry", "centreline", "yes", "true"])
def test_resolve_perception_rejects_unknown_values(typo):
    """A typo used to fall back to centerline, silently training the wrong sensor."""
    with pytest.raises(ValueError, match="track_perception"):
        resolve_perception({"track_perception": typo})


def test_legacy_preview_mode_pair_still_resolves():
    """Checkpoints predating the `track_perception` rename must keep loading."""
    assert resolve_perception({"path_preview": True, "preview_mode": "boundaries"}) \
        == (True, "boundary")
    assert resolve_perception({"path_preview": False}) == (False, "centerline")


def test_unspecified_perception_is_off():
    assert resolve_perception({}) == (False, "centerline")


# --- padding still does its job ---------------------------------------------

def test_pad_obs_input_zero_pads_only_the_new_trailing_columns():
    from automataleague_parkour.training.models import _pad_obs_input
    sd = {"0.weight": torch.ones(8, 49), "2.weight": torch.ones(8, 64)}
    out = _pad_obs_input(sd, 61, hidden_sizes=[64])
    assert out["0.weight"].shape == (8, 61)
    assert torch.all(out["0.weight"][:, :49] == 1)
    assert torch.all(out["0.weight"][:, 49:] == 0)
    assert out["2.weight"].shape == (8, 64)      # hidden width untouched


# --- what a checkpoint carries ----------------------------------------------

torchrl_only = pytest.mark.skipif(
    importlib.util.find_spec("torchrl") is None,
    reason="torchrl ships in the `train` extra",
)


@torchrl_only
def test_layout_is_reconstructed_from_a_legacy_checkpoint_config():
    """Checkpoints predating the stored layout only carry their hydra config. That is
    enough to rebuild the layout, so existing runs stay warm-startable AND checked."""
    from automataleague_parkour.training.ppo import checkpoint_obs_layout
    state = {"config": {"env": {"robot": "spot",
                                "course": {"height_scan": True, "track_perception": "none"}}}}
    assert checkpoint_obs_layout(state, SPOT) == (("proprio", 49), ("height_scan", 12))


@torchrl_only
def test_layout_is_reconstructed_from_a_pre_rename_checkpoint():
    """Older still: the `path_preview` + `preview_mode` pair, before track_perception."""
    from automataleague_parkour.training.ppo import checkpoint_obs_layout
    state = {"config": {"env": {"robot": "spot",
                                "course": {"path_preview": True, "preview_mode": "boundaries"}}}}
    assert checkpoint_obs_layout(state, SPOT) == (("proprio", 49), ("track_preview", 17))


@torchrl_only
def test_stored_layout_wins_over_the_config():
    from automataleague_parkour.training.ppo import checkpoint_obs_layout
    state = {"obs_layout": [["proprio", 49], ["height_scan", 12]],
             "config": {"env": {"robot": "spot", "course": {}}}}
    assert checkpoint_obs_layout(state, SPOT) == (("proprio", 49), ("height_scan", 12))


@torchrl_only
def test_the_recipe_stage3_warm_start_is_caught():
    """End to end on the real configs: a level-0 completion checkpoint warm-started
    into the racing preset. 61 -> 66 is wider, so padding alone would accept it."""
    from automataleague_parkour.training.models import check_obs_layout_compatible
    from automataleague_parkour.training.ppo import checkpoint_obs_layout

    completion = {"config": {"env": {"robot": "spot", "course": {
        "height_scan": True, "track_perception": "none"}}}}
    racing = _course(False, "boundary")

    src = checkpoint_obs_layout(completion, SPOT)
    dst = obs_layout(racing, SPOT)
    assert sum(w for _, w in src) == 61 and sum(w for _, w in dst) == 66
    with pytest.raises(ValueError) as exc:
        check_obs_layout_compatible(src, dst)
    assert "height_scan" in str(exc.value) and "track_preview" in str(exc.value)


# --- replaying a checkpoint after the registry defaults moved ----------------

@torchrl_only
def test_legacy_checkpoint_perception_is_pinned_to_none():
    """A checkpoint predating the preview sensor must not pick up today's registry
    default, or its env is built wider than the network it trained."""
    from omegaconf import OmegaConf

    from automataleague_parkour.training.inference import _pin_legacy_perception
    cfg = OmegaConf.create({"env": {"robot": "spot", "name": "parkour-1",
                                    "course": {"height_scan": True, "level_difficulty": 2}}})
    _pin_legacy_perception(cfg)
    assert cfg.env.course.track_perception == "none"


@torchrl_only
def test_explicit_perception_is_left_alone():
    from omegaconf import OmegaConf

    from automataleague_parkour.training.inference import _pin_legacy_perception
    cfg = OmegaConf.create({"env": {"robot": "spot", "name": "parkour-1",
                                    "course": {"track_perception": "boundary"}}})
    _pin_legacy_perception(cfg)
    assert cfg.env.course.track_perception == "boundary"


@torchrl_only
def test_legacy_preview_pair_is_left_alone():
    from omegaconf import OmegaConf

    from automataleague_parkour.training.inference import _pin_legacy_perception
    cfg = OmegaConf.create({"env": {"robot": "spot", "name": "parkour-1",
                                    "course": {"path_preview": True,
                                               "preview_mode": "boundaries"}}})
    _pin_legacy_perception(cfg)
    assert "track_perception" not in cfg.env.course
