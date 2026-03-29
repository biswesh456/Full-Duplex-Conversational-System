from typing import Iterable

import torch


def get_parameter_groups(model, weight_decay: float):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_adamw(model, optimizer_cfg: dict):
    param_groups = get_parameter_groups(
        model=model,
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    return torch.optim.AdamW(
        param_groups,
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.95])),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
    )