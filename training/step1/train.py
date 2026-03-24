from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightning as L
import torch
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader

from training.step1.data.collator import PackedCollator
from training.step1.data.packed_webdataset import (
    EvalDataset,
    MixedTrainDataset,
    parse_eval_specs,
    parse_train_specs,
)
from training.step1.engine.lr_scheduler import build_scheduler
from training.step1.engine.train_loop import train
from training.step1.models.qwen_causal_lm import build_qwen_causal_lm
from training.step1.optimizers import build_adamw
from training.step1.strategies.fsdp import build_fsdp_strategy
from training.step1.utils.config import ensure_dir, load_yaml
from training.step1.utils.logging import print_rank_zero


def compute_text_vocab_size(pretrained_name_or_path: str, trust_remote_code: bool) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    return len(tokenizer)


def build_fabric(cfg: dict) -> L.Fabric:
    fabric_cfg = cfg["fabric"]
    log_cfg = cfg["logging"]

    logger = TensorBoardLogger(
        root_dir=log_cfg["log_dir"],
        name=cfg["run_name"],
    )

    plugins = []
    if "SLURM_JOB_ID" in os.environ:
        plugins.append(SLURMEnvironment())

    strategy_name = fabric_cfg.get("strategy", "fsdp")
    if strategy_name != "fsdp":
        raise ValueError("This starter code currently supports only strategy=fsdp")

    strategy = build_fsdp_strategy(fabric_cfg)

    fabric = L.Fabric(
        accelerator=fabric_cfg.get("accelerator", "cuda"),
        devices=fabric_cfg.get("devices", "auto"),
        num_nodes=int(fabric_cfg.get("num_nodes", 1)),
        precision=fabric_cfg.get("precision", "bf16-mixed"),
        strategy=strategy,
        loggers=logger,
        plugins=plugins,
    )
    return fabric


def build_dataloaders(cfg: dict, pad_token_id: int):
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    train_specs = parse_train_specs(data_cfg["train_mix"])
    val_specs = parse_eval_specs(data_cfg["val_sets"])

    train_dataset = MixedTrainDataset(
        specs=train_specs,
        seed=int(cfg.get("seed", 42)),
        sample_shuffle=1000,
    )
    val_dataset = EvalDataset(specs=val_specs)

    collator = PackedCollator(
        pad_token_id=pad_token_id,
        pad_to_multiple_of=cfg["model"].get("pad_to_multiple_of"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["per_device_train_batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["per_device_val_batch_size"]),
        num_workers=max(1, int(train_cfg.get("num_workers", 4)) // 2),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        collate_fn=collator,
    )

    return train_loader, val_loader


def maybe_resume(fabric: L.Fabric, resume_from: str | None, state: dict) -> int:
    if not resume_from:
        return 0

    fabric.load(resume_from, state)
    step = int(state["step"])
    print_rank_zero(fabric, f"Resumed from {resume_from} at step {step}", flush=True)
    return step


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)

    seed = int(cfg.get("seed", 42))
    torch.set_float32_matmul_precision("high")

    fabric = build_fabric(cfg)
    fabric.launch()
    fabric.seed_everything(seed)

    log_dir = ensure_dir(cfg["logging"]["log_dir"])
    ckpt_dir = ensure_dir(Path(cfg["logging"]["ckpt_dir"]) / cfg["run_name"])

    model_cfg = dict(cfg["model"])
    model_cfg["text_vocab_size"] = compute_text_vocab_size(
        pretrained_name_or_path=model_cfg["pretrained_name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    print_rank_zero(fabric, "Building model...", flush=True)
    model = build_qwen_causal_lm(model_cfg)

    pad_token_id = int(model.config.pad_token_id)
    train_loader, val_loader = build_dataloaders(cfg, pad_token_id=pad_token_id)

    print_rank_zero(fabric, "Setting up dataloaders...", flush=True)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    print_rank_zero(fabric, "Building optimizer/scheduler...", flush=True)
    optimizer = build_adamw(model, cfg["optimizer"])
    total_steps = int(cfg["training"]["max_steps"])
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_cfg=cfg["scheduler"],
        total_steps=total_steps,
    )

    print_rank_zero(fabric, "Wrapping model and optimizer with Fabric...", flush=True)
    model, optimizer = fabric.setup(model, optimizer)

    world_size = fabric.world_size
    per_device_bs = int(cfg["training"]["per_device_train_batch_size"])
    global_batch_size = int(cfg["training"]["global_batch_size"])

    denom = world_size * per_device_bs
    if global_batch_size % denom != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by "
            f"world_size({world_size}) * per_device_train_batch_size({per_device_bs}) = {denom}"
        )

    grad_accum_steps = global_batch_size // denom
    print_rank_zero(
        fabric,
        f"world_size={world_size} per_device_bs={per_device_bs} global_bs={global_batch_size} grad_accum_steps={grad_accum_steps}",
        flush=True,
    )

    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "step": 0,
    }

    start_step = maybe_resume(
        fabric=fabric,
        resume_from=cfg.get("resume_from"),
        state=state,
    )

    if fabric.is_global_zero:
        logger = fabric.loggers[0].experiment
        logger.add_text("config", str(cfg))

    train(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        grad_accum_steps=grad_accum_steps,
        max_steps=int(cfg["training"]["max_steps"]),
        grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
        log_every_n_steps=int(cfg["training"]["log_every_n_steps"]),
        val_every_n_steps=int(cfg["training"]["val_every_n_steps"]),
        ckpt_every_n_steps=int(cfg["training"]["ckpt_every_n_steps"]),
        ckpt_dir=str(ckpt_dir),
        max_val_batches=int(cfg["training"]["max_val_batches"]),
        start_step=start_step,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)