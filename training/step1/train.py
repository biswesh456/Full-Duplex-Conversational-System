from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from transformers import get_cosine_schedule_with_warmup

from config import (
    FullConfig,
    ModelConfig,
    TrainConfig,
    DatasetConfig,
)
from data.special_tokens import SpecialTokenLayout
from data.sequence_builder import SequenceBuilder
from data.webdataset_source import WdsSequenceDataset
from data.collator import CausalBatchCollator
from data.batch_mixer import WeightedBatchMixer
from model.qwen_speech import QwenSpeechCausalLM
from utils.distributed import (
    set_seed,
    init_distributed,
    cleanup_distributed,
    is_main_process,
)
from utils.checkpointing import save_checkpoint
from utils.logging_utils import log


def build_example_config() -> FullConfig:
    return FullConfig(
        model=ModelConfig(
            model_path="/path/to/local/Qwen3-8B",
            torch_dtype="bfloat16",
            num_codebooks=8,
            speech_codebook_size=2048,
            num_extra_special_tokens=8,
            pad_to_multiple_of=8,
        ),
        train=TrainConfig(
            output_dir="./outputs/qwen3_speech",
            seed=42,
            lr=2e-5,
            weight_decay=0.1,
            max_steps=50000,
            warmup_steps=1000,
            grad_accum_steps=4,
            log_every=10,
            save_every=1000,
            max_length=4096,
            use_fsdp=True,
            fsdp_use_orig_params=True,
            activation_checkpointing=False,
        ),
        commonvoice=DatasetConfig(
            name="commonvoice",
            urls="/path/to/commonvoice/train/*.tar",
            weight=2.0,
            batch_size=2,
            num_workers=4,
        ),
        covost=DatasetConfig(
            name="covost",
            urls="/path/to/covost/train/*.tar",
            weight=1.0,
            batch_size=2,
            num_workers=2,
        ),
        gigaspeech=DatasetConfig(
            name="gigaspeech",
            urls="/path/to/gigaspeech/train/*.tar",
            weight=2.0,
            batch_size=2,
            num_workers=6,
        ),
        spoken_squad=DatasetConfig(
            name="spoken_squad",
            urls="/path/to/spoken_squad/train/*.tar",
            weight=1.0,
            batch_size=2,
            num_workers=2,
        ),
        ultrachat=DatasetConfig(
            name="ultrachat",
            urls="/path/to/ultrachat/train/*.tar",
            weight=2.0,
            batch_size=2,
            num_workers=2,
        ),
    )


def build_dataloader(ds_cfg, sequence_builder, collator):
    ds = WdsSequenceDataset(
        urls=ds_cfg.urls,
        sequence_builder=sequence_builder,
        shuffle_buffer=ds_cfg.shuffle_buffer,
        resampled=ds_cfg.resampled,
    )

    return DataLoader(
        ds,
        batch_size=ds_cfg.batch_size,
        num_workers=ds_cfg.num_workers,
        pin_memory=ds_cfg.pin_memory,
        persistent_workers=ds_cfg.persistent_workers and ds_cfg.num_workers > 0,
        collate_fn=collator,
        drop_last=True,
    )


def maybe_wrap_fsdp(model: torch.nn.Module, train_cfg: TrainConfig):
    if not train_cfg.use_fsdp:
        return model

    transformer_block_cls = {model.model.model.layers[0].__class__}

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    wrapped = FSDP(
        model,
        auto_wrap_policy=lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
            module=module,
            recurse=recurse,
            nonwrapped_numel=nonwrapped_numel,
            transformer_layer_cls=transformer_block_cls,
        ),
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=train_cfg.fsdp_use_orig_params,
        forward_prefetch=False,
        limit_all_gathers=True,
    )

    if train_cfg.activation_checkpointing:
        non_reentrant_wrapper = lambda m: checkpoint_wrapper(
            m,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule.__class__ in transformer_block_cls,
        )

    return wrapped


def build_optimizer(model: torch.nn.Module, train_cfg: TrainConfig):
    decay_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        eps=train_cfg.eps,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if to
