"""Render a trained checkpoint to an mp4.

Used by the `examples/` entry points to drop an eval video after every training stage
(so a curriculum leaves one clip per level), and by `tools/render_policy.py`.

Frames are written to the encoder in chunks rather than accumulated: a full lap at
720p is ~2500 frames, and holding those in a list is several GB before it ever
reaches the muxer.
"""
from __future__ import annotations

import os

import numpy as np

DEFAULT_CAMERA = "chase"


def render_checkpoint(checkpoint_path, out_path, *, camera=DEFAULT_CAMERA, max_steps=2500,
                      device="cpu", fps=30, size=(720, 1280), chunk=200):
    """Roll the policy in `checkpoint_path` through its own course and write `out_path`.

    The level, sensors and reward come from the checkpoint's stored config, so the
    policy is always replayed on the course it trained on. Returns the number of
    simulated steps written, or None if rendering failed (never raises: a video is
    a nice-to-have and must not kill a training run that already succeeded).
    """
    import imageio.v2 as imageio

    from automataleague_parkour.envs.parkour.parkour_cpu import ParkourEnvCPU
    from automataleague_parkour.envs.parkour.render import record_trajectory, render_trajectory
    from automataleague_parkour.training.env import configs_from_cfg
    from automataleague_parkour.training.inference import load_policy

    actor, cfg = load_policy(checkpoint_path, device)
    course, rc, tc = configs_from_cfg(cfg)
    tc.max_episode_steps = max_steps
    env = ParkourEnvCPU(robot=cfg.env.robot, cfg=course, reward_cfg=rc,
                        term_cfg=tc, frame_skip=10)

    qpos = np.asarray(record_trajectory(env, actor, max_steps,
                                        policy_device=device, loop_episodes=False))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with imageio.get_writer(out_path, fps=fps) as writer:
        for i in range(0, len(qpos), chunk):
            for frame in render_trajectory(env.model, env.info, qpos[i:i + chunk],
                                           camera=camera, size=size):
                writer.append_data(frame)
    return len(qpos)


def render_stage_video(cfg, checkpoint_path, run_name, *, videos_dir="videos"):
    """Post-stage hook for the `examples/` scripts: render `checkpoint_path` if the
    config asks for it, and never let a rendering problem fail the run.

    Controlled by `logger.stage_video` (bool) and `logger.stage_video_camera`.
    """
    logger_cfg = getattr(cfg, "logger", None)
    if not bool(getattr(logger_cfg, "stage_video", False)):
        return None

    out_path = os.path.join(videos_dir, f"{run_name}.mp4")
    camera = str(getattr(logger_cfg, "stage_video_camera", DEFAULT_CAMERA))
    max_steps = int(getattr(logger_cfg, "stage_video_steps", 2500))
    try:
        steps = render_checkpoint(checkpoint_path, out_path, camera=camera,
                                  max_steps=max_steps)
    except Exception as exc:                      # noqa: BLE001 - best effort by design
        print(f"[video] {run_name}: rendering failed ({type(exc).__name__}: {exc})")
        return None
    print(f"[video] {run_name}: wrote {out_path} ({steps} steps, {steps / 50:.1f}s, {camera})")
    return out_path
