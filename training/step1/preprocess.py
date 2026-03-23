import argparse
import json
from pathlib import Path
import yaml
from tqdm import tqdm

from training.step1.preprocessing.builders import SequenceBuilder
from training.step1.preprocessing.raw_readers import iter_raw_webdataset
from training.step1.preprocessing.writer import PackedShardWriter
from tokenization.multimodal_tokenizer import MultimodalTokenizer

def resolve_webdataset_urls(urls: str | list[str]) -> list[str]:
    """
    Resolve WebDataset inputs into a flat list of .tar shard paths.
    """
    def _resolve_one(item: str) -> list[str]:
        p = Path(item)

        if p.exists():
            if p.is_dir():
                shards = sorted(p.rglob("*.tar"))
                if not shards:
                    raise FileNotFoundError(f"No .tar shards found under directory tree: {p}")
                return [str(s) for s in shards]

            if p.is_file():
                if p.suffix != ".tar":
                    raise ValueError(f"Expected a .tar shard file, got: {p}")
                return [str(p)]

        return [item]

    if isinstance(urls, str):
        urls = [urls]

    resolved: list[str] = []
    for item in urls:
        resolved.extend(_resolve_one(item))

    if not resolved:
        raise ValueError("No WebDataset shards were resolved")

    return resolved

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--maxcount", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=32000)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tokenizer_name = cfg["tokenizer_name"]
    num_codebooks = int(cfg["num_codebooks"])
    speech_codebook_size = int(cfg.get("speech_codebook_size", 2048))
    datasets = cfg["datasets"]

    for ds in datasets:
        urls = ds["urls"]
        if isinstance(urls, str):
            ds["urls"] = urls.format(num_codebooks=num_codebooks)
        elif isinstance(urls, list):
            ds["urls"] = [u.format(num_codebooks=num_codebooks) for u in urls]

        output_urls = ds["output_urls"]
        if not isinstance(output_urls, str):
            raise ValueError(f"'output_urls' must be a single directory path string for dataset {ds['name']}")

        ds["output_urls"] = output_urls.format(num_codebooks=num_codebooks)
            
    print('Preparing the multimodal tokenizer...', flush=True)
    mm_tokenizer = MultimodalTokenizer.from_pretrained(
        tokenizer_name,
        num_codebooks=num_codebooks,
        speech_codebook_size=speech_codebook_size,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )

    print('Preparing the builder...', flush=True)
    builder = SequenceBuilder(mm_tokenizer=mm_tokenizer, max_length=args.max_length)

    tokenizer_meta = {
        "base_model": tokenizer_name,
        "text_vocab_size": mm_tokenizer.ranges.text_vocab_size,
        "speech_offset": mm_tokenizer.ranges.speech_offset,
        "speech_vocab_size_total": mm_tokenizer.ranges.speech_vocab_size_total,
        "full_vocab_size": mm_tokenizer.ranges.full_vocab_size,
        "num_codebooks": num_codebooks,
        "speech_codebook_size": speech_codebook_size,
        "extra_special_ids": mm_tokenizer.ranges.extra_special_ids,
    }

    for ds in datasets:
        name = ds["name"]
        urls = resolve_webdataset_urls(ds["urls"])
        print("urls", urls)
        output_root = Path(ds["output_urls"])
        print('Creating output directory...', flush=True)
        output_root.mkdir(parents=True, exist_ok=True)

        with open(output_root / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_meta, f, indent=2, ensure_ascii=False)

        writer = PackedShardWriter(output_root, prefix="shard", maxcount=args.maxcount)
        count = 0
        for raw_sample in tqdm(iter_raw_webdataset(urls), desc=f"packing {name}"):
            packed = builder.build(raw_sample)
            writer.write(packed)
            count += 1
        writer.close()
        print(f"[DONE] {name}: {count} samples")


if __name__ == "__main__":
    main()