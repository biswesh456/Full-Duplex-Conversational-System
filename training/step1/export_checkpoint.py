import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.environments import SLURMEnvironment
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

from training.models.qwen_causal_lm import build_qwen_causal_lm
from training.strategies.fsdp import build_fsdp_strategy
from training.utils.config import load_yaml
from tokenization.multimodal_tokenizer import MultimodalTokenizer


def load_tokenizer_config_from_path(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer config path does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_shared_tokenizer_cfg(cfg: dict) -> dict[str, Any]:
    tok_cfg = cfg.get("tokenizer", {})
    config_path = tok_cfg.get("config_path")

    if not config_path:
        raise ValueError(
            "Expected tokenizer.config_path in export config, pointing to a tokenizer_config.json file."
        )

    shared_tokenizer_cfg = load_tokenizer_config_from_path(config_path)

    required_keys = [
        "base_model",
        "text_vocab_size",
        "speech_offset",
        "speech_vocab_size_total",
        "full_vocab_size",
        "num_codebooks",
        "speech_codebook_size",
        "extra_special_ids",
    ]
    missing = [k for k in required_keys if k not in shared_tokenizer_cfg]
    if missing:
        raise ValueError(
            f"Tokenizer config at {config_path} is missing required keys: {missing}"
        )

    return shared_tokenizer_cfg


def build_fabric(cfg: dict, num_nodes: int, num_gpus_per_node: int) -> Fabric:
    fabric_cfg = cfg["fabric"]

    plugins = []
    if "SLURM_JOB_ID" in os.environ:
        plugins.append(SLURMEnvironment())

    strategy_name = fabric_cfg.get("strategy", "fsdp")
    if strategy_name != "fsdp":
        raise ValueError("This export script currently supports only strategy=fsdp")

    strategy = build_fsdp_strategy(fabric_cfg)

    return Fabric(
        accelerator=fabric_cfg.get("accelerator", "cuda"),
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        precision=fabric_cfg.get("precision", "bf16-mixed"),
        strategy=strategy,
        plugins=plugins,
    )


def build_model_and_tokenizer(cfg: dict):
    shared_tokenizer_cfg = get_shared_tokenizer_cfg(cfg)

    mm_tokenizer = MultimodalTokenizer.from_config(
        shared_tokenizer_cfg,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )

    model_cfg = dict(cfg["model"])

    # Build from the base model when reconstructing the architecture for checkpoint load.
    model_cfg["pretrained_name_or_path"] = shared_tokenizer_cfg["base_model"]
    model_cfg["text_vocab_size"] = int(shared_tokenizer_cfg["text_vocab_size"])
    model_cfg["num_codebooks"] = int(shared_tokenizer_cfg["num_codebooks"])
    model_cfg["speech_codebook_size"] = int(shared_tokenizer_cfg["speech_codebook_size"])
    model_cfg["full_vocab_size"] = int(shared_tokenizer_cfg["full_vocab_size"])

    model = build_qwen_causal_lm(model_cfg)
    return model, model_cfg, shared_tokenizer_cfg, mm_tokenizer


def export_full_state_dict(fabric: Fabric, model):
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        full_state_dict = model.state_dict()

    if fabric.global_rank == 0:
        return full_state_dict
    return None


def main(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    num_nodes: int,
    num_gpus_per_node: int,
) -> None:
    cfg = load_yaml(config_path)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_cudnn_sdp(False)

    fabric = build_fabric(cfg, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
    fabric.launch()

    model, model_cfg, shared_tokenizer_cfg, mm_tokenizer = build_model_and_tokenizer(cfg)
    model = fabric.setup(model)

    state = {"model": model}
    fabric.load(checkpoint_path, state)

    full_state_dict = export_full_state_dict(fabric=fabric, model=model)

    if fabric.global_rank == 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_model = model.module if hasattr(model, "module") else model
        raw_model.eval()

        if not hasattr(raw_model, "save_pretrained"):
            raise TypeError(
                "Underlying model does not implement save_pretrained(). "
                "build_qwen_causal_lm must return a Hugging Face PreTrainedModel-compatible object."
            )

        raw_model.save_pretrained(
            output_dir,
            state_dict=full_state_dict,
            safe_serialization=True,
        )

        mm_tokenizer.save_config(output_dir / "mm_tokenizer_config.json")

        torch.save(
            {
                "model_cfg": model_cfg,
                "tokenizer_cfg": shared_tokenizer_cfg,
                "source_checkpoint": checkpoint_path,
            },
            output_dir / "training_metadata.pt",
        )

        print(f"Saved HF export to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=1)
    args = parser.parse_args()

    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
    )