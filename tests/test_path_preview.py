import torch

from automataleague_parkour.envs.parkour.path_preview import (
    cumulative_length,
    preview_dim,
    track_preview,
)

DIST = (1.0, 2.0, 3.0)
YAW0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity quat (w,x,y,z) -> yaw 0, facing +x


def test_preview_dim():
    assert preview_dim(DIST) == 2 * 3 + 1


def test_cumulative_length_accumulates_segment_lengths():
    # 3-4-5 triangle then a vertical drop: lengths 5, then 4.
    pl = torch.tensor([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
    cum = cumulative_length(pl)
    assert torch.allclose(cum, torch.tensor([0.0, 5.0, 9.0]), atol=1e-5)


def test_straight_ahead_points_are_on_axis():
    pl = torch.tensor([[0.0, 0.0], [20.0, 0.0]])
    cum = cumulative_length(pl)
    base_xy = torch.tensor([[5.0, 0.0]])
    out = track_preview(base_xy, YAW0, pl, cum, DIST, closed=False)
    pts = out[0, :6].reshape(3, 2)
    assert torch.allclose(pts[:, 0], torch.tensor([1.0, 2.0, 3.0]), atol=1e-3)  # x = distance ahead
    assert torch.allclose(pts[:, 1], torch.zeros(3), atol=1e-3)  # y = 0 on a straight
    assert torch.allclose(out[0, -1], torch.tensor(0.0), atol=1e-3)  # centered -> lateral 0


def test_left_curve_points_bend_positive_y():
    # Quarter circle turning left; lookahead points should have y > 0 in the base frame.
    th = torch.linspace(0, torch.pi / 2, 50)
    pl = torch.stack([torch.sin(th) * 10, (1 - torch.cos(th)) * 10], -1)
    cum = cumulative_length(pl)
    out = track_preview(pl[0:1], YAW0, pl, cum, (2.0, 4.0, 6.0), closed=False)
    pts = out[0, :6].reshape(3, 2)
    assert (pts[:, 1] > 0).all()


def test_closed_track_wrap_matches_manual_arc_length():
    # Closed square, perimeter 40. Base sits at arc-length 39 (on the final edge,
    # 1 unit from the finish); a 3 m lookahead must wrap to arc-length 2 on the
    # first edge, i.e. world point (2, 0).
    pl = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]])
    cum = cumulative_length(pl)
    base_xy = torch.tensor([[0.0, 1.0]])  # arc-length 39, exactly on the centerline
    out = track_preview(base_xy, YAW0, pl, cum, (3.0,), closed=True)
    # world point (2, 0) minus base (0, 1), expressed in the identity (yaw=0) frame.
    assert torch.allclose(out[0, :2], torch.tensor([2.0, -1.0]), atol=1e-3)
    assert torch.allclose(out[0, -1], torch.tensor(0.0), atol=1e-3)  # base is on the centerline


def test_open_track_clamps_past_the_end():
    pl = torch.tensor([[0.0, 0.0], [20.0, 0.0]])
    cum = cumulative_length(pl)
    base_xy = torch.tensor([[19.0, 0.0]])  # 1 m from the end
    out = track_preview(base_xy, YAW0, pl, cum, (5.0,), closed=False)
    # A 5 m lookahead overshoots the 20 m track; it must clamp to the final point
    # (20, 0), not wrap or extrapolate past it.
    assert torch.allclose(out[0, :2], torch.tensor([1.0, 0.0]), atol=1e-3)


def test_lap_wrap_samples_past_start():
    pl = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]])  # closed square
    cum = cumulative_length(pl)
    base_xy = torch.tensor([[0.0, 0.5]])  # just before the finish/start
    out = track_preview(
        base_xy, torch.tensor([[0.7071, 0, 0, -0.7071]]), pl, cum, (2.0,), closed=True
    )
    assert out.shape == (1, 3)  # runs without index error at the wrap


def test_batched_equals_single():
    pl = torch.tensor([[0.0, 0.0], [20.0, 0.0]])
    cum = cumulative_length(pl)
    b = torch.tensor([[3.0, 0.2], [8.0, -0.1]])
    q = YAW0.repeat(2, 1)
    out = track_preview(b, q, pl, cum, DIST, closed=False)
    s0 = track_preview(b[0:1], q[0:1], pl, cum, DIST, closed=False)
    assert torch.allclose(out[0], s0[0], atol=1e-5)
