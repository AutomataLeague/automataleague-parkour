"""Parkour leaderboard — rank agents by lap time on a track/difficulty level.

Algorithm-agnostic by design: an ENTRY is a checkpoint + metadata (algorithm, name,
author, notes). A per-algorithm loader turns it into an obs->action policy, so PPO /
SAC / TD3 (and community submissions) all compete on equal footing. Each entry is
time-trialed over N noisy starts on the SAME course (race-nav, so the ideal line isn't
penalized) and ranked by finish rate, then median lap time.

    MUJOCO_GL=egl uv run python tools/leaderboard.py \
        --entries tools/leaderboard_entries.json --track circuit --level 2 --seeds 5

Adding a new algorithm = implement its branch in `load_policy` (map its checkpoint to
an actor with an actor(TensorDict{observation})->action interface).
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import mujoco  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from automataleague.envs.parkour.config import ParkourConfig, TerminationConfig  # noqa: E402
from automataleague.envs.parkour.parkour_cpu import ParkourEnvCPU  # noqa: E402
from automataleague.robots import get_robot  # noqa: E402

from automataleague.training.models import build_actor as _build_actor  # noqa: E402

FRAME_DT = 0.002


def load_policy(entry, device):
    """Return (actor, cfg) for an entry. actor(TensorDict{'observation'})->'action'."""
    algo = entry.get("algorithm", "PPO").upper()
    ckpt = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    robot = get_robot(cfg.env.robot)
    if algo == "PPO":
        actor = _build_actor(cfg, robot, device)
        actor.load_state_dict(ckpt["actor_state_dict"])
        actor.eval()
        return actor, cfg
    # SAC / TD3 / etc.: add a loader that reconstructs the actor and loads its weights.
    raise NotImplementedError(
        f"No policy loader for algorithm '{algo}' yet — only PPO. Add a branch here.")


def time_trials(actor, cfg, track, level, n_seeds, max_steps=3500):
    """Run n_seeds noisy time-trials on (track, level). Returns per-seed (finished, time_s,
    checkpoints). The policy keeps its own action_scale / height_scan (a trained property);
    the course is race-nav + nominal (no obstacle DR) so timing is fair and comparable."""
    ec = cfg.env.course
    dt = FRAME_DT * int(getattr(cfg.env, "frame_skip", 10))
    course = ParkourConfig(
        track=track, level_difficulty=level,
        height_scan=bool(getattr(ec, "height_scan", False)),
        action_scale=getattr(ec, "action_scale", None),
        race_mode=True, randomize_obstacles=False,
    )
    # let a full lap run (default cap is 1000 steps ~= 20s; a lap takes ~35s)
    tc = TerminationConfig(max_episode_steps=max_steps)
    env = ParkourEnvCPU(robot=cfg.env.robot, cfg=course, term_cfg=tc,
                        frame_skip=int(getattr(cfg.env, "frame_skip", 10)))
    robot = env.robot
    ncp = env._checkpoints.shape[0]
    ba = env.info.base_qposadr
    rows = []
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for seed in range(n_seeds):
            np.random.seed(seed)
            obs = env.reset()
            env.data.qpos[ba:ba + 2] += np.random.randn(2) * 0.05
            env.data.qpos[env.info.joint_qposadr] += np.random.randn(robot.n_joints) * 0.05
            mujoco.mj_forward(env.model, env.data)
            obs = env._obs().squeeze(0).numpy()
            steps = max_steps
            for t in range(max_steps):
                td = actor(TensorDict({"observation": torch.as_tensor(obs, dtype=torch.float32)[None]}, [1]))
                obs, _, term, trunc, info = env.step(td["action"].squeeze(0).numpy())
                if term or trunc:
                    steps = t + 1
                    break
            reached = int(env.cp_idx.item())
            finished = reached >= ncp - 1 and steps < max_steps - 1
            rows.append((finished, steps * dt if finished else None, reached, ncp))
    return rows


def summarize(rows):
    fins = [r for r in rows if r[0]]
    times = sorted(r[1] for r in fins)
    reaches = [r[2] for r in rows]
    ncp = rows[0][3]
    return {
        "finish_rate": len(fins) / len(rows),
        "finishes": len(fins), "runs": len(rows),
        "best_time_s": round(times[0], 2) if times else None,
        "median_time_s": round(times[len(times) // 2], 2) if times else None,
        "mean_checkpoints": round(float(np.mean(reaches)), 1), "n_checkpoints": ncp,
    }


def rank_key(s):
    # finishers first (higher finish rate), then faster median time; non-finishers by reach
    t = s["median_time_s"] if s["median_time_s"] is not None else 1e9
    return (-s["finish_rate"], t, -s["mean_checkpoints"])


def render_markdown(board, track, level, n_seeds):
    lines = [f"# parkour1 leaderboard — track `{track}`, level {level}  ({n_seeds} noisy starts each)",
             "", "| # | Agent | Algo | Finish | Best | Median | Reach |",
             "|---|---|---|---|---|---|---|"]
    for i, e in enumerate(board, 1):
        s = e["result"]
        bt = f"{s['best_time_s']:.1f}s" if s["best_time_s"] else "—"
        mt = f"{s['median_time_s']:.1f}s" if s["median_time_s"] else "—"
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, str(i))
        lines.append(f"| {medal} | {e['name']} | {e['algorithm']} | "
                     f"{s['finishes']}/{s['runs']} | {bt} | {mt} | "
                     f"{s['mean_checkpoints']:.0f}/{s['n_checkpoints']} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Rank parkour agents by lap time.")
    p.add_argument("--entries", default="tools/leaderboard_entries.json")
    p.add_argument("--track", default="circuit")
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="results/leaderboard.json")
    p.add_argument("--md", default="results/leaderboard.md")
    args = p.parse_args()

    device = torch.device("cpu")
    entries = json.load(open(args.entries))
    board = []
    for e in entries:
        if not os.path.exists(e["checkpoint"]):
            print(f"  skip {e['name']}: missing {e['checkpoint']}"); continue
        try:
            actor, cfg = load_policy(e, device)
        except NotImplementedError as exc:
            print(f"  skip {e['name']}: {exc}"); continue
        rows = time_trials(actor, cfg, args.track, args.level, args.seeds)
        s = summarize(rows)
        board.append({**e, "result": s})
        print(f"  {e['name']:28s} [{e['algorithm']}]: "
              f"{s['finishes']}/{s['runs']} finish, median "
              f"{s['median_time_s'] if s['median_time_s'] else '--'}s, "
              f"reach {s['mean_checkpoints']:.0f}/{s['n_checkpoints']}")

    board.sort(key=lambda e: rank_key(e["result"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"track": args.track, "level": args.level, "seeds": args.seeds,
               "ranking": board}, open(args.out, "w"), indent=2)
    md = render_markdown(board, args.track, args.level, args.seeds)
    open(args.md, "w").write(md)
    print("\n" + md + f"\n\nwrote {args.out} + {args.md}")


if __name__ == "__main__":
    main()
