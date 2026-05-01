from contextlib import nullcontext
from collections import deque
import time
import math

import torch
from lightning.fabric import Fabric

from training.utils.checkpointing import save_checkpoint
from training.utils.logging import print_rank_zero

def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m {secs:02d}s"
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:02d}m {secs:02d}s"
    return f"{secs}s"


@torch.no_grad()
def run_validation(
    fabric: Fabric,
    model,
    val_loader,
    max_val_batches: int,
    val_dataset_names: list[str],
    print_every_n_val_batches: int = 5000,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    model.eval()

    total_loss = torch.zeros((), device=fabric.device)
    total_batches = torch.zeros((), device=fabric.device)

    per_dataset_loss_sum = {
        name: torch.zeros((), device=fabric.device) for name in val_dataset_names
    }
    per_dataset_batch_count = {
        name: torch.zeros((), device=fabric.device) for name in val_dataset_names
    }

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

        dataset_names = [m["eval_dataset"] for m in batch["meta"]]
        dataset_name = dataset_names[0]

        if any(name != dataset_name for name in dataset_names):
            raise ValueError(
                f"Validation batch contains mixed eval datasets: {dataset_names}"
            )

        if dataset_name not in per_dataset_loss_sum:
            raise ValueError(f"Unknown eval dataset name: {dataset_name}")

        per_dataset_loss_sum[dataset_name] += loss
        per_dataset_batch_count[dataset_name] += 1

        if (
            print_every_n_val_batches > 0
            and ((batch_idx + 1) % print_every_n_val_batches == 0 or (batch_idx + 1) == max_val_batches)
        ):
            running_mean_loss = (total_loss / total_batches).item()
            current_dataset_batches = int(per_dataset_batch_count[dataset_name].item())

            print_rank_zero(
                fabric,
                f"[val batch {batch_idx + 1}/{max_val_batches}] "
                f"dataset={dataset_name} "
                f"batch_loss={loss.item():.4f} "
                f"running_loss={running_mean_loss:.4f} "
                f"dataset_batches_seen={current_dataset_batches}",
                flush=True,
            )

    total_loss = fabric.all_reduce(total_loss, reduce_op="sum")
    total_batches = fabric.all_reduce(total_batches, reduce_op="sum")

    if total_batches.item() == 0:
        overall_loss = float("nan")
        overall_ppl = float("nan")
    else:
        overall_loss = (total_loss / total_batches).item()
        overall_ppl = torch.exp(total_loss / total_batches).item()

    per_dataset_losses: dict[str, float] = {}
    per_dataset_ppls: dict[str, float] = {}

    for dataset_name in val_dataset_names:
        ds_loss_sum = fabric.all_reduce(per_dataset_loss_sum[dataset_name], reduce_op="sum")
        ds_batch_count = fabric.all_reduce(per_dataset_batch_count[dataset_name], reduce_op="sum")

        if ds_batch_count.item() == 0:
            per_dataset_losses[dataset_name] = float("nan")
            per_dataset_ppls[dataset_name] = float("nan")
        else:
            mean_ds_loss = ds_loss_sum / ds_batch_count
            per_dataset_losses[dataset_name] = mean_ds_loss.item()
            per_dataset_ppls[dataset_name] = torch.exp(mean_ds_loss).item()

    model.train()
    return overall_loss, overall_ppl, per_dataset_losses, per_dataset_ppls


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
    val_dataset_names: list[str] | None = None
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_iter = iter(train_loader)
    step = start_step

    current_stage = None
    if get_curriculum_stage_fn is not None:
        current_stage = get_curriculum_stage_fn(start_step)

    train_start_time = time.perf_counter()
    recent_step_times = deque(maxlen=50)

    while step < max_steps:
        step_start_time = time.perf_counter()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            is_accumulating = micro_step < grad_accum_steps - 1
            sync_context = (
                fabric.no_backward_sync(model, enabled=is_accumulating)
                if grad_accum_steps > 1
                else nullcontext()
            ) # Do not sync the gradients across the gpus if it is still accumulating.

            with sync_context:
                batch = next(train_iter)

                batch_size = batch["input_ids"].shape[0]
                seq_len = batch["input_ids"].shape[1]
                padded_tokens = batch_size * seq_len

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False, # cache is useful during generation, disabling it saves memory
                )
                loss = outputs.loss / grad_accum_steps
                fabric.backward(loss) # will handle mixed precision, scaling distributed strategy specifics, other device-related details

            loss_accum += loss.detach().float().item()

        grad_norm = fabric.clip_gradients(
            model,
            optimizer,
            max_norm=grad_clip_norm,
        ) 

        optimizer.step() # Updates parameters
        scheduler.step() # Updates LR
        optimizer.zero_grad(set_to_none=True)

        step += 1

        step_time = time.perf_counter() - step_start_time
        recent_step_times.append(step_time)

        elapsed_time = time.perf_counter() - train_start_time
        avg_step_time_recent = sum(recent_step_times) / len(recent_step_times)
        steps_remaining = max_steps - step
        eta_seconds = steps_remaining * avg_step_time_recent

        # For curriculum learning
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
            train_pplx = math.exp(loss_accum)
            metrics = {
                "train/loss": loss_accum,
                "train/pplx": train_pplx,
                "train/lr": lr,
                "time/step_sec": step_time,
                "time/avg_step_sec_recent": avg_step_time_recent,
                "time/elapsed_sec": elapsed_time,
                "time/eta_sec": eta_seconds,
                "train/seq_len": seq_len,
                "train/padded_tokens": padded_tokens,
            }
            if isinstance(grad_norm, torch.Tensor):
                metrics["train/grad_norm"] = grad_norm.detach().float().item()
            else:
                metrics["train/grad_norm"] = float(grad_norm)

            fabric.log_dict(metrics, step=step)
            print_rank_zero(
                fabric,
                f"[step {step}/{max_steps}] "
                f"loss={loss_accum:.4f} "
                f"perplexity={train_pplx:.4f} "
                f"lr={lr:.6e} "
                f"grad_norm={metrics['train/grad_norm']:.4f} "
                f"step_time={step_time:.2f}s "
                f"avg_step={avg_step_time_recent:.2f}s "
                f"elapsed={format_seconds(elapsed_time)} "
                f"eta={format_seconds(eta_seconds)} ",
                f"train/seq_len={seq_len} ",
                f"train/padded_tokens={padded_tokens} ",
                flush=True,
            )

        if step % val_every_n_steps == 0:
            print_rank_zero(fabric, f"[step {step}] starting validation...", flush=True)
            val_loss, val_ppl, per_dataset_val_losses, per_dataset_val_ppls = run_validation(
                fabric=fabric,
                model=model,
                val_loader=val_loader,
                max_val_batches=max_val_batches,
                val_dataset_names=val_dataset_names or [],
            )
            print_rank_zero(fabric, f"[step {step}] finished validation.", flush=True)
            fabric.log("val/loss", val_loss, step=step)
            fabric.log("val/perplexity", val_ppl, step=step)
            print_rank_zero(
                    fabric,
                    f"[step {step}] val_loss={val_loss:.4f} val_ppl={val_ppl:.4f}",
                    flush=True,
                )

            for dataset_name in per_dataset_val_losses:
                dataset_loss = per_dataset_val_losses[dataset_name]
                dataset_ppl = per_dataset_val_ppls[dataset_name]

                fabric.log(f"val/{dataset_name}_loss", dataset_loss, step=step)
                fabric.log(f"val/{dataset_name}_perplexity", dataset_ppl, step=step)

                print_rank_zero(
                    fabric,
                    f"[step {step}] val/{dataset_name}_loss={dataset_loss:.4f} "
                    f"val/{dataset_name}_ppl={dataset_ppl:.4f}",
                    flush=True,
                )

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