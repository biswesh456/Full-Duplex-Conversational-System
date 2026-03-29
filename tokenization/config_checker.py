import json

def load_tokenizer_config_from_dataset_root(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    path = root / "tokenizer_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing tokenizer_config.json at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def assert_same_tokenizer_config(dataset_roots: list[str | Path]) -> dict[str, Any]:
    configs = [load_tokenizer_config_from_dataset_root(root) for root in dataset_roots]
    ref = configs[0]

    keys = [
        "base_model",
        "text_vocab_size",
        "speech_offset",
        "speech_vocab_size_total",
        "full_vocab_size",
        "num_codebooks",
        "speech_codebook_size",
        "extra_special_ids",
    ]

    for i, cfg in enumerate(configs[1:], start=1):
        for key in keys:
            if cfg.get(key) != ref.get(key):
                raise ValueError(
                    f"Tokenizer config mismatch between dataset 0 and dataset {i} for key '{key}': "
                    f"{ref.get(key)} != {cfg.get(key)}"
                )
    return ref