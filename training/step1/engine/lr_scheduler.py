import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: dict,
    total_steps: int,
):
    name = scheduler_cfg.get("name", "cosine")
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))

    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        if name == "constant":
            return 1.0
        if name == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        raise ValueError(f"Unsupported scheduler name: {name}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)