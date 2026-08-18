"""PPO time-trial racing on parkour-1 — the fastest lap, not just reaching the finish.

Same PPO loop as `ppo_single.py`, but with the racing preset (`config_race.yaml`):
`race_mode` navigation, boundary track perception, and a lap-time reward. The flat
circuit (level 0, no obstacles) is the default; race an obstacle course by raising the
level and turning the height scan on.

    uv run python examples/ppo_race.py                        # race the flat circuit (level 0)
    uv run python examples/ppo_race.py env.course.level_difficulty=2 env.course.height_scan=true
"""
import os

import hydra
import numpy as np
import torch

from automataleague_parkour.training import run_ppo


@hydra.main(version_base="1.1", config_path="", config_name="config_race")
def main(cfg):  # noqa: F821
    os.chdir(hydra.utils.get_original_cwd())   # restore launch dir so checkpoints/ land there
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    best = run_ppo(
        cfg,
        level=cfg.env.course.level_difficulty,
        total_frames=cfg.collector.total_frames,
        action_scale=cfg.env.course.action_scale,
        init_ckpt=cfg.network.init_checkpoint,
        run_name=f"race_L{cfg.env.course.level_difficulty}",
    )
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
