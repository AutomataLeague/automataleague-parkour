"""1v1 parkour eval: race two trained agents on the same track.

Two modes (eval only — agents are trained independently):
  * time-trial   — each agent runs the track solo; report time-to-finish.
  * head-to-head — both agents run the SAME track together (two robots in one
                   scene) so they can collide / block each other; first to the
                   finish wins. A race video is rendered.

Agents are "blind" to each other (they use their solo 49-dim observation); they
interact only through physics. Works for any two checkpoints (PPO vs PPO, etc.).

Usage:
    MUJOCO_GL=egl uv run python training/eval_1v1.py --track circuit \
        --agents runA.pt runB.pt --names Alice Bob --level 0 --out results/race.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from automataleague.envs.parkour.navigation import (  # noqa: E402
    advance_checkpoints,
    checkpoint_geometry,
)
from automataleague.envs.parkour.observation import build_observation  # noqa: E402
from automataleague.envs.parkour.parkour_cpu import ParkourEnvCPU  # noqa: E402
from automataleague.envs.parkour.scene import build_race_model  # noqa: E402
from automataleague.envs.parkour.spatial import tilt_angle  # noqa: E402
from automataleague.envs.parkour.state import extract_state  # noqa: E402
from automataleague.robots import get_robot  # noqa: E402

from demo_render import _build_actor  # noqa: E402
from utils_ppo import _configs_from_cfg  # noqa: E402


def load_agent(path, track, level):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    cfg.env.course.track = track            # race on the requested track/difficulty
    cfg.env.course.level_difficulty = level
    robot = get_robot(cfg.env.robot)
    actor = _build_actor(cfg, robot, torch.device("cpu"))
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor, cfg, robot


def _policy_action(actor, obs):
    td = actor(TensorDict({"observation": obs}, batch_size=[1]))
    return td["action"].squeeze(0).numpy()


def _respawn(data, info, robot):
    """Teleport one robot back to its start pose (used when it falls over)."""
    ba, da = info.base_qposadr, info.base_dofadr
    data.qpos[ba:ba + 7] = info.home_qpos[ba:ba + 7]
    data.qpos[info.joint_qposadr] = robot.home_joint_qpos
    data.qvel[da:da + 6] = 0.0
    data.qvel[info.joint_dofadr] = 0.0


def time_trial(actor, cfg, robot, max_steps, dt):
    """Run the agent solo; on a fall it respawns at the start and keeps trying (the
    lost time counts). Returns (finish_time_s or None, checkpoints_reached, respawns)."""
    course, rc, tc = _configs_from_cfg(cfg)
    tc.max_episode_steps = max_steps + 10        # only falls/off-path end a run, not truncation
    env = ParkourEnvCPU(robot=cfg.env.robot, cfg=course, reward_cfg=rc, term_cfg=tc,
                        frame_skip=getattr(cfg.env, "frame_skip", 10))
    obs = env.reset()
    respawns = 0
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for step in range(max_steps):
            action = _policy_action(actor, torch.as_tensor(obs, dtype=torch.float32)[None])
            obs, _, term, trunc, info = env.step(action)
            if info["outcome"] == 1:
                return (step + 1) * dt, int(env.cp_idx.item()), respawns
            if term or trunc:                     # fell / off-path -> respawn at the start
                respawns += 1
                obs = env.reset()
    return None, int(env.cp_idx.item()), respawns


def head_to_head(actors, cfg, robot, max_steps, dt, render_size=(720, 1280)):
    """Race all agents together in one scene. Returns finish times + frames."""
    course, _, tc = _configs_from_cfg(cfg)
    fall_h, max_tilt = tc.fall_height, math.radians(tc.max_tilt_deg)
    model, infos = build_race_model(cfg.env.robot, course, n=len(actors))
    data = mujoco.MjData(model)
    data.qpos[:] = infos[0].home_qpos
    mujoco.mj_forward(model, data)

    checkpoints = torch.tensor(infos[0].checkpoints_xy, dtype=torch.float32)
    n_cp = checkpoints.shape[0]
    radius = course.checkpoint_radius
    home_joint = torch.tensor(robot.home_joint_qpos, dtype=torch.float32)
    ctrl0 = np.asarray(robot.home_joint_qpos, dtype=np.float64)
    scale = robot.action_scale
    fs = getattr(cfg.env, "frame_skip", 10)

    cp_idx = [0] * len(actors)
    prev_action = [torch.zeros(1, robot.action_dim) for _ in actors]
    finish_step = [None] * len(actors)
    respawns = [0] * len(actors)

    renderer = mujoco.Renderer(model, height=render_size[0], width=render_size[1])
    cam = mujoco.MjvCamera()
    frames = []

    def state_of(i):
        qpos = torch.as_tensor(data.qpos, dtype=torch.float32)[None]
        qvel = torch.as_tensor(data.qvel, dtype=torch.float32)[None]
        return extract_state(qpos, qvel, infos[i])

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for step in range(max_steps):
            # each agent acts on its own observation (blind to the opponent)
            for i, (actor, info) in enumerate(zip(actors, infos)):
                st = state_of(i)
                to_cp, dist, herr = checkpoint_geometry(st, checkpoints, torch.tensor([cp_idx[i]]))
                obs = build_observation(st, to_cp, dist, herr, prev_action[i], home_joint)
                action = _policy_action(actor, obs)
                data.ctrl[info.actuator_ids] = ctrl0 + scale * action
                prev_action[i] = torch.as_tensor(action, dtype=torch.float32)[None]

            for _ in range(fs):
                mujoco.mj_step(model, data)

            # respawn fallers at the start; otherwise advance checkpoints + detect finish
            for i, info in enumerate(infos):
                if finish_step[i] is not None:
                    continue
                st = state_of(i)
                fell = (float(st.base_pos[0, 2]) < fall_h
                        or float(tilt_angle(st.base_quat)[0]) > max_tilt)
                if fell:
                    _respawn(data, info, robot)
                    cp_idx[i] = 0
                    prev_action[i] = torch.zeros(1, robot.action_dim)
                    respawns[i] += 1
                    continue
                _, dist, _ = checkpoint_geometry(st, checkpoints, torch.tensor([cp_idx[i]]))
                new_idx, _, fin = advance_checkpoints(dist, torch.tensor([cp_idx[i]]), radius, n_cp)
                cp_idx[i] = int(new_idx.item())
                if bool(fin.item()):
                    finish_step[i] = step + 1

            # follow the pack: camera looks at the mean base position
            bases = np.array([data.qpos[info.base_qposadr:info.base_qposadr + 2] for info in infos])
            cam.lookat = np.array([bases[:, 0].mean(), bases[:, 1].mean(), 0.3])
            cam.azimuth, cam.elevation, cam.distance = 50, -25, 7.0
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

            if all(f is not None for f in finish_step):
                break
    renderer.close()
    times = [None if f is None else f * dt for f in finish_step]
    return times, cp_idx, respawns, np.stack(frames)


def main():
    p = argparse.ArgumentParser(description="1v1 parkour race between two agents.")
    p.add_argument("--track", default="circuit")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--agents", nargs=2, required=True, metavar=("A", "B"))
    p.add_argument("--names", nargs=2, default=["agent_a", "agent_b"])
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--out", default="results/race.json")
    p.add_argument("--video", default="videos/race.mp4")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    agents = [load_agent(pth, args.track, args.level) for pth in args.agents]
    robot = agents[0][2]
    dt = 0.002 * getattr(agents[0][1].env, "frame_skip", 10)

    print(f"=== Track: {args.track} (level {args.level}) | dt={dt:.3f}s ===\n-- time trials --")
    trials = {}
    for name, (actor, cfg, rob) in zip(args.names, agents):
        t, cps, r = time_trial(actor, cfg, rob, args.max_steps, dt)
        trials[name] = {"time_s": t, "checkpoints": cps, "respawns": r, "finished": t is not None}
        print(f"  {name}: {'DNF' if t is None else f'{t:.2f}s'}  "
              f"({cps} checkpoints, {r} respawns)")

    print("\n-- head-to-head --")
    actors = [a[0] for a in agents]
    times, cps, resp, frames = head_to_head(actors, agents[0][1], robot, args.max_steps, dt)
    h2h = {}
    for name, t, c, r in zip(args.names, times, cps, resp):
        h2h[name] = {"time_s": t, "checkpoints": c, "respawns": r, "finished": t is not None}
        print(f"  {name}: {'DNF' if t is None else f'{t:.2f}s'}  "
              f"({c} checkpoints, {r} respawns)")
    finishers = [(n, v["time_s"]) for n, v in h2h.items() if v["time_s"] is not None]
    if finishers:
        winner = min(finishers, key=lambda x: x[1])[0]
    else:  # nobody finished -> furthest along wins
        winner = max(h2h.items(), key=lambda kv: kv[1]["checkpoints"])[0]
    print(f"  WINNER: {winner}")

    if not args.no_video and frames is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=int(1 / dt))
        print(f"\n  race video: {args.video} ({len(frames)} frames)")

    result = {
        "track": args.track, "level": args.level, "dt_s": dt,
        "agents": dict(zip(args.names, args.agents)),
        "time_trial": trials, "head_to_head": h2h, "h2h_winner": winner,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  results: {args.out}")


if __name__ == "__main__":
    main()
