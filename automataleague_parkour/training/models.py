"""PPO actor/critic builders (task-agnostic; sized from env specs)."""
from __future__ import annotations

import torch
import torch.nn
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator


def get_activation(cfg):
    return {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh,
            "leaky_relu": torch.nn.LeakyReLU, "elu": torch.nn.ELU}[cfg.network.activation]


def check_obs_layout_compatible(src_layout, dst_layout):
    """Raise unless a policy with `src_layout` can be zero-padded into `dst_layout`.

    Zero-padding appends columns at the end, so it is only correct when the source
    layout is a **prefix** of the destination: every block the source had is still
    there, at the same width, in the same position. Widening alone is not enough.

    The case this exists for: a level-0 completion policy is
    `[proprio 49 | height_scan 12]` = 61 and the racing preset is
    `[proprio 49 | track_preview 17]` = 66. That is wider, so width-only padding
    accepted it and fed 12 columns of boundary-preview values into weights trained
    on terrain heights, with nothing raised and nothing logged.
    """
    def fmt(layout):
        return "[" + " | ".join(f"{n} {w}" for n, w in layout) + f"] = {sum(w for _, w in layout)}"

    for i, (name, width) in enumerate(src_layout):
        if i >= len(dst_layout):
            raise ValueError(
                f"Cannot warm-start: the checkpoint has a block the target does not.\n"
                f"  checkpoint: {fmt(src_layout)}\n  target:     {fmt(dst_layout)}\n"
                f"Dropping the trailing '{name}' block would shift every column after it."
            )
        dst_name, dst_width = dst_layout[i]
        if (name, width) != (dst_name, dst_width):
            raise ValueError(
                f"Cannot warm-start: observation block {i} differs.\n"
                f"  checkpoint: {fmt(src_layout)}\n  target:     {fmt(dst_layout)}\n"
                f"Block {i} is '{name}' ({width} wide) in the checkpoint but "
                f"'{dst_name}' ({dst_width} wide) in the target, so the trained weights "
                f"for '{name}' would receive '{dst_name}' values. Train from scratch, or "
                f"match the sensors (height_scan / track_perception) to the checkpoint."
            )


def _pad_obs_input(sd, cur_obs, hidden_sizes=()):
    """Zero-pad the input-layer weights of a warm-start state_dict to `cur_obs`
    columns, so a policy trained on a smaller observation (e.g. blind, 49-dim) loads
    into a sensor-augmented network (e.g. +12 height-scan). Only the first layer
    (in_features < cur_obs and not a hidden width) is padded; its new trailing
    columns start at zero, so the warm-started policy initially ignores the sensor
    and keeps its learned gait. A same-dim checkpoint is loaded unchanged.

    Correct only when the source layout is a prefix of the target: call
    `check_obs_layout_compatible` first (`ppo.run_ppo` does).
    """
    skip = {int(h) for h in hidden_sizes}
    out = {}
    for k, v in sd.items():
        if v.dim() == 2 and v.shape[1] < cur_obs and v.shape[1] not in skip:
            pad = torch.zeros(v.shape[0], cur_obs - v.shape[1], dtype=v.dtype, device=v.device)
            v = torch.cat([v, pad], dim=1)
        out[k] = v
    return out


def make_ppo_models(cfg, train_env, device):
    """Actor (MLP -> TanhNormal ProbabilisticActor) and critic (MLP -> ValueOperator)."""
    input_shape = train_env.observation_spec["observation"].shape
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]

    num_outputs = action_spec.shape[-1]
    activation_class = get_activation(cfg)
    hidden_sizes = cfg.network.hidden_sizes

    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=activation_class,
        out_features=num_outputs,
        num_cells=hidden_sizes,
        device=device,
    )
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(num_outputs, scale_lb=1e-8).to(device),
    )

    policy_module = ProbabilisticActor(
        TensorDictModule(module=policy_mlp, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=activation_class,
        out_features=1,
        num_cells=hidden_sizes,
        device=device,
    )
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    value_module = ValueOperator(value_mlp, in_keys=["observation"])
    return policy_module, value_module


def build_actor(cfg, robot, device):
    """Rebuild the actor from config/dims without a live GPU env (stub specs)."""
    from automataleague_parkour.envs.parkour.observation import obs_width
    # One source of truth for the width: the same obs_layout the envs are sized from,
    # so a rebuilt actor cannot drift from what build_observation concatenates.
    obs_dim = obs_width(getattr(cfg.env, "course", object()), robot)

    class _Stub:
        pass
    stub = _Stub()
    stub.batch_size = torch.Size([1])
    stub.observation_spec = Composite(
        observation=Unbounded(shape=(1, obs_dim), device=device), shape=(1,))
    action_spec = Composite(
        action=Bounded(low=-torch.ones(1, robot.action_dim, device=device),
                       high=torch.ones(1, robot.action_dim, device=device), device=device),
        shape=(1,))
    # make_ppo_models reads env.action_spec as the leaf Bounded.
    stub.action_spec = action_spec["action"]
    actor, _ = make_ppo_models(cfg, stub, device)
    return actor
