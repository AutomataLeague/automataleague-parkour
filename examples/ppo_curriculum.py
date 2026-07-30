"""PPO curriculum across parkour-1 difficulty levels, warm-starting each stage.

    uv run python examples/ppo_curriculum.py
"""
import os

import hydra
import numpy as np
import torch

from automataleague_parkour.training import curriculum_from_cfg, run_ppo


@hydra.main(version_base="1.1", config_path="", config_name="config_ppo")
def main(cfg):  # noqa: F821
    os.chdir(hydra.utils.get_original_cwd())   # Hydra chdirs into outputs/; restore so checkpoints/ land at the launch dir
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    cur = curriculum_from_cfg(cfg)
    prev_best = cfg.network.init_checkpoint     # optional warm-start into level 1
    for i, level in enumerate(cur.levels):
        print(f"=== curriculum level {level} "
              f"({cur.frames_per_level[i]:,} frames, scale {cur.action_scale_per_level[i]}) ===")
        best = run_ppo(
            cfg,
            level=level,
            total_frames=cur.frames_per_level[i],
            action_scale=cur.action_scale_per_level[i],
            init_ckpt=prev_best if cur.warm_start else None,
            run_name=f"parkour1_curriculum_L{level}",
        )
        print(f"level {level} best -> {best}")
        prev_best = best
    print(f"curriculum complete; final policy: {prev_best}")


if __name__ == "__main__":
    main()
