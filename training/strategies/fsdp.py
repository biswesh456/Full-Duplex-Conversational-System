from functools import partial

from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.plugins.precision import FSDPPrecision
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp import CPUOffload, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

def build_fsdp_strategy(fabric_cfg: dict):
    sharding_name = fabric_cfg.get("fsdp_sharding_strategy", "FULL_SHARD")
    state_dict_type = fabric_cfg.get("fsdp_state_dict_type", "sharded")
    limit_all_gathers = bool(fabric_cfg.get("fsdp_limit_all_gathers", True))
    cpu_offload = bool(fabric_cfg.get("fsdp_cpu_offload", False))
    use_orig_params = bool(fabric_cfg.get("fsdp_use_orig_params", True))
    min_num_params = int(fabric_cfg.get("fsdp_min_num_params", 10_000_000))
    precision = fabric_cfg.get("precision", "bf16-mixed")

    sharding_strategy = getattr(ShardingStrategy, sharding_name)

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3DecoderLayer},
    )

    return FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing_policy={Qwen3DecoderLayer},
        sharding_strategy=sharding_strategy,
        state_dict_type=state_dict_type,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        limit_all_gathers=limit_all_gathers,
        use_orig_params=use_orig_params,
        precision=FSDPPrecision(precision),
    )