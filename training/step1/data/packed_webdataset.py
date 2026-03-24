import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset


def resolve_webdataset_urls(urls: str | list[str]) -> list[str]:
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


def _load_npz_bytes(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _decode_packed_sample(sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample.get("json")
    arrays = sample.get("npz")

    if isinstance(meta, (bytes, bytearray)):
        meta = json.loads(meta.decode("utf-8"))
    elif isinstance(meta, str):
        meta = json.loads(meta)

    if not isinstance(meta, dict):
        raise ValueError("Invalid or missing metadata in packed sample")

    if isinstance(arrays, (bytes, bytearray)):
        arrays = _load_npz_bytes(arrays)

    if not isinstance(arrays, dict):
        raise ValueError("Invalid or missing npz arrays in packed sample")

    return {
        "input_ids": torch.from_numpy(arrays["input_ids"]).long(),
        "labels": torch.from_numpy(arrays["labels"]).long(),
        "attention_mask": torch.from_numpy(arrays["attention_mask"]).long(),
        "meta": meta,
    }


@dataclass
class DatasetSpec:
    name: str
    urls: list[str]
    weight: float = 1.0
    max_samples: int | None = None


def build_wds_pipeline(
    urls: list[str],
    train: bool,
    shardshuffle: bool = True,
    sample_shuffle: int = 1000,
) -> Iterator[dict[str, Any]]:
    if train:
        pipeline = wds.DataPipeline(
            wds.ResampledShards(urls),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(sample_shuffle),
            wds.map(_decode_packed_sample),
        )
    else:
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(urls),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.map(_decode_packed_sample),
        )

    return iter(pipeline)


class MixedTrainDataset(IterableDataset):
    def __init__(
        self,
        specs: list[DatasetSpec],
        seed: int = 42,
        sample_shuffle: int = 1000,
    ) -> None:
        super().__init__()
        if len(specs) == 0:
            raise ValueError("MixedTrainDataset requires at least one dataset spec")
        self.specs = specs
        self.seed = seed
        self.sample_shuffle = sample_shuffle

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.seed + torch.initial_seed())

        iterators = [
            build_wds_pipeline(
                urls=spec.urls,
                train=True,
                shardshuffle=True,
                sample_shuffle=self.sample_shuffle,
            )
            for spec in self.specs
        ]
        weights = [spec.weight for spec in self.specs]

        while True:
            ds_idx = rng.choices(range(len(iterators)), weights=weights, k=1)[0]
            try:
                sample = next(iterators[ds_idx])
            except StopIteration:
                iterators[ds_idx] = build_wds_pipeline(
                    urls=self.specs[ds_idx].urls,
                    train=True,
                    shardshuffle=True,
                    sample_shuffle=self.sample_shuffle,
                )
                sample = next(iterators[ds_idx])

            sample["meta"] = dict(sample["meta"])
            sample["meta"]["mixture_dataset"] = self.specs[ds_idx].name
            yield sample


class EvalDataset(IterableDataset):
    def __init__(self, specs: list[DatasetSpec]) -> None:
        super().__init__()
        self.specs = specs

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for spec in self.specs:
            count = 0
            iterator = build_wds_pipeline(urls=spec.urls, train=False, shardshuffle=False)

            for sample in iterator:
                sample["meta"] = dict(sample["meta"])
                sample["meta"]["eval_dataset"] = spec.name
                yield sample

                count += 1
                if spec.max_samples is not None and count >= spec.max_samples:
                    break


def parse_train_specs(cfg_list: list[dict[str, Any]]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for ds in cfg_list:
        specs.append(
            DatasetSpec(
                name=ds["name"],
                urls=resolve_webdataset_urls(ds["urls"]),
                weight=float(ds["weight"]),
            )
        )
    return specs


def parse_eval_specs(cfg_list: list[dict[str, Any]]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for ds in cfg_list:
        specs.append(
            DatasetSpec(
                name=ds["name"],
                urls=resolve_webdataset_urls(ds["urls"]),
                max_samples=ds.get("max_samples"),
            )
        )
    return specs