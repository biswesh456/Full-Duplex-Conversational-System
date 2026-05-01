from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def initialize_new_embeddings(
    weight: torch.Tensor,
    old_vocab_size: int,
    scale_factor: float = 1,
) -> None:
    if old_vocab_size >= weight.shape[0]:
        return

    with torch.no_grad():
        old_mean = weight[:old_vocab_size].mean(dim=0, keepdim=True) # Does it for each dimension
        old_std = weight[:old_vocab_size].std(dim=0, keepdim=True)
        new_rows = weight[old_vocab_size:]

        # randn_like gives normal distribution with mean 0 and variance 1
        noise = torch.randn_like(new_rows)
        # To avoid multiplying by zero or something extremely tiny if one dimension has near-zero standard deviation, we clamp it. We can further provide a scaling factor to reduce the deviation further.
        new_rows.copy_(old_mean + noise * old_std.clamp_min(1e-6) * scale_factor) 


def resize_model_for_speech_tokens(
    model,
    full_vocab_size: int,
) -> None:
    full_vocab_size = ((full_vocab_size + 127) // 128) * 128  # Convert it to closest multiple of 128 to make it faster for gpus.
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    if old_vocab_size == full_vocab_size:
        return

    model.resize_token_embeddings(full_vocab_size)
    input_emb = model.get_input_embeddings().weight
    initialize_new_embeddings(input_emb, old_vocab_size=old_vocab_size)

    output_emb_module = model.get_output_embeddings()
    if output_emb_module is not None:
        output_emb = output_emb_module.weight
        if output_emb.shape[0] == full_vocab_size:
            initialize_new_embeddings(output_emb, old_vocab_size=old_vocab_size)
    model.config.vocab_size = full_vocab_size


def build_qwen_causal_lm(model_cfg: dict[str, Any]):
    pretrained_name_or_path = model_cfg["pretrained_name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    attn_implementation = model_cfg.get("attn_implementation", 'sdpa')
    torch_dtype_name = model_cfg.get("torch_dtype", "bfloat16")

    if torch_dtype_name == "bfloat16":
        torch_dtype = torch.bfloat16
    elif torch_dtype_name == "float16":
        torch_dtype = torch.float16
    elif torch_dtype_name == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype_name}")

    full_vocab_size = int(model_cfg["full_vocab_size"])

    config = AutoConfig.from_pretrained(
        pretrained_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_name_or_path,
        config=config,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )

    resize_model_for_speech_tokens(model, full_vocab_size=full_vocab_size)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.config.use_cache = False

    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model