from contextlib import nullcontext

import torch
from lightning.fabric import Fabric

from training.utils.checkpointing import save_checkpoint
from training.utils.logging import print_rank_zero


@torch.no_grad()
def run_validation(
    fabric: Fabric,
    model,
    val_loader,
    max_val_batches: int,
) -> float:
    model.eval()

    total_loss = torch.zeros((), device=fabric.device)
    total_batches = torch.zeros((), device=fabric.device)

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_val_batches:
            break

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        loss = outputs.loss.detach()

        total_loss += loss
        total_batches += 1

    total_loss = fabric.all_reduce(total_loss, reduce_op="sum")
    total_batches = fabric.all_reduce(total_batches, reduce_op="sum")

    model.train()

    if total_batches.item() == 0:
        return float("nan")

    return (total_loss / total_batches).item()


def train(
    fabric: Fabric,
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    grad_accum_steps: int,
    max_steps: int,
    grad_clip_norm: float,
    log_every_n_steps: int,
    val_every_n_steps: int,
    ckpt_every_n_steps: int,
    ckpt_dir: str,
    max_val_batches: int,
    start_step: int = 0,
    get_curriculum_stage_fn=None,
    rebuild_train_loader_fn=None,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_iter = iter(train_loader)
    step = start_step

    current_stage = None
    if get_curriculum_stage_fn is not None:
        current_stage = get_curriculum_stage_fn(start_step)

    while step < max_steps:
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            is_accumulating = micro_step < grad_accum_steps - 1
            sync_context = (
                fabric.no_backward_sync(model, enabled=is_accumulating)
                if grad_accum_steps > 1
                else nullcontext()
            )

            with sync_context:
                batch = next(train_iter)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = outputs.loss / grad_accum_steps
                fabric.backward(loss)

            loss_accum += loss.detach().float().item()

        grad_norm = fabric.clip_gradients(
            model,
            optimizer,
            max_norm=grad_clip_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1

        if (
            get_curriculum_stage_fn is not None
            and rebuild_train_loader_fn is not None
        ):
            new_stage = get_curriculum_stage_fn(step)
            if new_stage != current_stage:
                print_rank_zero(
                    fabric,
                    f"[step {step}] curriculum stage changed: {current_stage} -> {new_stage}. Rebuilding train loader.",
                    flush=True,
                )
                train_loader = rebuild_train_loader_fn(step)
                train_iter = iter(train_loader)
                current_stage = new_stage

        if step % log_every_n_steps == 0:
            lr = optimizer.param_groups[0]["lr"]
            metrics = {
                "train/loss": loss_accum,
                "train/lr": lr,
            }
            if isinstance(grad_norm, torch.Tensor):
                metrics["train/grad_norm"] = grad_norm.detach().float().item()
            else:
                metrics["train/grad_norm"] = float(grad_norm)

            fabric.log_dict(metrics, step=step)
            print_rank_zero(
                fabric,
                f"[step {step}] loss={loss_accum:.4f} lr={lr:.6e} grad_norm={metrics['train/grad_norm']:.4f}",
                flush=True,
            )

        if step % val_every_n_steps == 0:
            val_loss = run_validation(
                fabric=fabric,
                model=model,
                val_loader=val_loader,
                max_val_batches=max_val_batches,
            )
            fabric.log("val/loss", val_loss, step=step)
            print_rank_zero(fabric, f"[step {step}] val_loss={val_loss:.4f}", flush=True)

        if step % ckpt_every_n_steps == 0:
            save_checkpoint(
                fabric=fabric,
                ckpt_dir=ckpt_dir,
                state={
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "step": step,
                },
                step=step,
            )
            print_rank_zero(fabric, f"[step {step}] checkpoint saved", flush=True)

    save_checkpoint(
        fabric=fabric,
        ckpt_dir=ckpt_dir,
        state={
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "step": step,
        },
        step=step,
        tag="final",
    )