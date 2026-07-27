"""PPO on a single parkour-1 difficulty level (standard TorchRL scheme).

    uv run python examples/ppo_single.py env.course.level_difficulty=2
"""
import hydra
import numpy as np
import torch

from automataleague.training import run_ppo


@hydra.main(version_base="1.1", config_path="", config_name="config_ppo")
def main(cfg):  # noqa: F821
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    best = run_ppo(
        cfg,
        level=cfg.env.course.level_difficulty,
        total_frames=cfg.collector.total_frames,
        action_scale=cfg.env.course.action_scale,
        init_ckpt=cfg.network.init_checkpoint,
        run_name=f"parkour1_L{cfg.env.course.level_difficulty}",
    )
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
