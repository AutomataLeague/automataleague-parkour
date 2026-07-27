"""Reusable PPO training loop (on-policy, GPU-parallel Warp envs)."""
from __future__ import annotations

import os, time
from collections import defaultdict

import numpy as np, torch, tqdm, wandb
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name, get_logger

from automataleague.training.env import make_environment, rollout_video, log_metrics
from automataleague.training.models import make_ppo_models, _pad_obs_input

OUTCOME_NAMES = {0: "ongoing", 1: "success", 2: "fell", 3: "off_path"}


def aggregate_outcomes(codes):
    stats = defaultdict(int)
    for code in codes:
        stats[OUTCOME_NAMES.get(int(code), "unknown")] += 1
    return dict(stats)


def run_ppo(cfg, *, level, total_frames, action_scale=None, init_ckpt=None,
            run_name="ppo", checkpoints_root="checkpoints") -> str:
    """Train one PPO run at `level` for `total_frames`; return the best checkpoint path."""
    # Parameterize the run into the shared cfg (mutating the DictConfig is fine here).
    cfg.env.course.level_difficulty = int(level)
    if action_scale is not None:
        cfg.env.course.action_scale = float(action_scale)
    cfg.collector.total_frames = int(total_frames)
    if init_ckpt is not None:
        cfg.network.init_checkpoint = init_ckpt

    device = cfg.network.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    checkpoint_dir = os.path.join(checkpoints_root, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}_{run_name}")
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="ppo_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    log_outcomes = getattr(cfg.logger, "log_outcome_breakdown", True)

    train_env, eval_env = make_environment(cfg)
    actor, critic = make_ppo_models(cfg, train_env, device)

    # Warm-start: initialise from a previously-trained policy (e.g. flat -> obstacles).
    init_ckpt = getattr(cfg.network, "init_checkpoint", None)
    if init_ckpt:
        state = torch.load(init_ckpt, map_location=device, weights_only=False)
        cur_obs = train_env.observation_spec["observation"].shape[-1]
        a_sd = _pad_obs_input(state["actor_state_dict"], cur_obs, cfg.network.hidden_sizes)
        c_sd = _pad_obs_input(state["critic_state_dict"], cur_obs, cfg.network.hidden_sizes)
        actor.load_state_dict(a_sd)
        critic.load_state_dict(c_sd)
        torchrl_logger.info(f"Warm-started from {init_ckpt} (obs padded to {cur_obs})")

    adv_module = GAE(
        gamma=cfg.loss.gamma, lmbda=cfg.loss.gae_lambda,
        value_network=critic, average_gae=False, device=device,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor, critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon, loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coeff=cfg.loss.entropy_coeff, critic_coeff=cfg.loss.critic_coeff,
        normalize_advantage=True,
    )

    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.optim.lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.optim.lr, eps=1e-5)
    optim = group_optimizers(actor_optim, critic_optim)
    del actor_optim, critic_optim

    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.collector.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.loss.mini_batch_size,
    )

    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        (cfg.collector.total_frames // cfg.collector.frames_per_batch)
        * cfg.loss.ppo_epochs * num_mini_batches
    )
    ppo_epochs = cfg.loss.ppo_epochs
    max_grad_norm = cfg.loss.max_grad_norm
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps
    anneal_lr = cfg.loss.anneal_lr
    anneal_clip_epsilon = cfg.loss.anneal_clip_epsilon
    cfg_lr = cfg.optim.lr
    cfg_clip_epsilon = cfg.loss.clip_epsilon
    steps_per_env = frames_per_batch // cfg.env.num_envs

    start_time = time.time()
    collected_frames = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    batch_outcomes = []
    losses = TensorDict(batch_size=[ppo_epochs, num_mini_batches])

    td = train_env.reset()

    # Track the best eval so late-training collapse never costs us the peak policy.
    best_score = float("-inf")

    while collected_frames < cfg.collector.total_frames:
        collect_start = time.time()
        rollout_tds = []
        for _ in range(steps_per_env):
            with torch.no_grad():
                td = actor(td)
            transition_td, td = train_env.step_and_maybe_reset(td)
            rollout_tds.append(transition_td.clone())
        data = torch.stack(rollout_tds, dim=1)
        collect_time = time.time() - collect_start

        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        metrics_to_log = {}
        metrics_to_log["train/collect_time"] = collect_time
        metrics_to_log["train/fps"] = frames_in_batch / collect_time

        episode_end = data["next", "done"]
        episode_rewards = data["next", "episode_reward"][episode_end]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = (
                episode_length.sum().item() / len(episode_length)
            )

        # Task-progress metrics at episode end — the honest "did it move toward the
        # finish?" signal, independent of reward shaping (guards against loitering).
        done_flat = data["next", "done"].squeeze(-1)
        if done_flat.any():
            if ("next", "checkpoint_idx") in data.keys(True):
                metrics_to_log["train/checkpoints_reached"] = (
                    data["next", "checkpoint_idx"][done_flat].mean().item()
                )
            if ("next", "dist_to_finish") in data.keys(True):
                metrics_to_log["train/final_dist_to_finish"] = (
                    data["next", "dist_to_finish"][done_flat].mean().item()
                )

        if ("next", "outcome") in data.keys(True):
            done_mask = data["next", "done"].squeeze(-1)
            if done_mask.any():
                batch_outcomes.extend(data["next", "outcome"][done_mask].tolist())

        if log_outcomes and batch_outcomes:
            stats = aggregate_outcomes(batch_outcomes)
            total_ep = sum(stats.values())
            for name, count in stats.items():
                if name == "ongoing":
                    continue
                metrics_to_log[f"train/{name}_rate"] = count / total_ep if total_ep else 0
            batch_outcomes = []

        training_start = time.time()
        for j in range(ppo_epochs):
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)
            data_buffer.extend(data_reshape)

            for k, batch in enumerate(data_buffer):
                alpha = 1.0
                if anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in optim.param_groups:
                        group["lr"] = cfg_lr * alpha
                if anneal_clip_epsilon:
                    loss_module.clip_epsilon.copy_(cfg_clip_epsilon * alpha)
                num_network_updates += 1

                optim.zero_grad(set_to_none=True)
                loss = loss_module(batch)
                total_loss = loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                losses[j, k] = loss.detach().select(
                    "loss_critic", "loss_entropy", "loss_objective"
                )
        training_time = time.time() - training_start

        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            metrics_to_log[f"train/{key}"] = value.item()
        metrics_to_log["train/training_time"] = training_time

        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_start = time.time()
                actor.eval()
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps, actor,
                    auto_cast_to_device=True, break_when_any_done=True,
                )
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward
                metrics_to_log["eval/episode_length"] = eval_rollout.batch_size[-1]
                metrics_to_log["eval/time"] = time.time() - eval_start

                # Did the single eval episode actually reach the finish, and how far?
                if ("next", "outcome") in eval_rollout.keys(True):
                    reached = (eval_rollout["next", "outcome"] == 1).any().item()
                    metrics_to_log["eval/reached_finish"] = float(reached)
                if ("next", "checkpoint_idx") in eval_rollout.keys(True):
                    metrics_to_log["eval/max_checkpoint"] = (
                        eval_rollout["next", "checkpoint_idx"].max().item()
                    )
                if ("next", "dist_to_finish") in eval_rollout.keys(True):
                    metrics_to_log["eval/final_dist_to_finish"] = (
                        eval_rollout["next", "dist_to_finish"][..., -1, :].mean().item()
                    )
                actor.train()

                # Keep the best policy by eval quality (finish > checkpoints reached >
                # closeness to finish), since PPO can oscillate / collapse late in training.
                score = (metrics_to_log.get("eval/reached_finish", 0.0) * 1000.0
                         + metrics_to_log.get("eval/max_checkpoint", 0.0)
                         - 0.01 * metrics_to_log.get("eval/final_dist_to_finish", 0.0))
                if score > best_score:
                    best_score = score
                    torch.save({
                        "actor_state_dict": actor.state_dict(),
                        "critic_state_dict": critic.state_dict(),
                        "collected_frames": collected_frames,
                        "config": dict(cfg),
                    }, os.path.join(checkpoint_dir, "ppo_best.pt"))
                    torchrl_logger.info(
                        f"New best policy (max_checkpoint="
                        f"{metrics_to_log.get('eval/max_checkpoint')}) -> ppo_best.pt")
                metrics_to_log["eval/best_score"] = best_score

            if cfg.logger.video and cfg.logger.backend:
                try:
                    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                        actor.eval()
                        # Cap generously and stop at the episode's natural end, so a
                        # full lap renders in full instead of being chopped at ~30 s.
                        video_steps = getattr(cfg.logger, "video_steps", 4000)
                        frames = rollout_video(
                            actor, cfg, max_steps=min(video_steps, eval_rollout_steps),
                            policy_device=str(device), stop_at_done=True,
                        )
                        actor.train()
                        vid = np.transpose(frames, (0, 3, 1, 2)).astype(np.uint8)
                        wandb.log({"eval/video": wandb.Video(vid, fps=30, format="mp4")})
                except Exception as exc:   # video is best-effort; never kill training
                    torchrl_logger.info(f"eval video skipped: {exc}")
                    actor.train()

            ckpt_path = os.path.join(checkpoint_dir, f"ppo_eval_{collected_frames}.pt")
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "collected_frames": collected_frames,
                "config": dict(cfg),
            }, ckpt_path)
            torchrl_logger.info(f"Saved checkpoint: {ckpt_path}")

        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)

    final_ckpt_path = os.path.join(checkpoint_dir, "ppo_final.pt")
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "collected_frames": collected_frames,
        "config": dict(cfg),
    }, final_ckpt_path)
    torchrl_logger.info(f"Saved final checkpoint: {final_ckpt_path}")
    torchrl_logger.info(f"Training took {time.time() - start_time:.2f}s")

    if logger is not None:
        wandb.finish()

    return os.path.join(checkpoint_dir, "ppo_best.pt")
