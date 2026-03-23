import io
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import webdataset as wds

from training.step1.preprocessing.schema import RawSample


def _load_npz_bytes(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}

def iter_raw_webdataset(urls: str | list[str]) -> Iterator[RawSample]:
    dataset = wds.WebDataset(urls, shardshuffle=False).decode()

    for sample in dataset:
        meta = sample.get("json")
        arrays = sample.get("npz")
        if isinstance(meta, (bytes, bytearray)):
            meta = json.loads(meta.decode("utf-8"))
        elif isinstance(meta, str):
            meta = json.loads(meta)

        if not isinstance(meta, dict):
            raise ValueError("json metadata missing or invalid")

        if isinstance(arrays, (bytes, bytearray)):
            arrays = _load_npz_bytes(arrays)
        if not isinstance(arrays, dict):
            raise ValueError("npz arrays missing or invalid")

        task = meta["task"]
        dataset_name = meta.get("dataset", "unknown")
        split = meta.get("split", "unknown")
        instruction = meta.get("instruction", "")

        input_text = None
        target_text = None
        input_speech = None
        target_speech = None
        question_text = None
        input_text_token = None
        target_text_token = None

        if task in {"asr", "tts", "speech_to_text_translation", "spoken_extract_qa", "text_dialog_sft"}:
            if task == "asr":
                input_speech = arrays["input_tokens"]
                target_text = meta["text"] if "text" in meta else meta["text_tn"]
            elif task == "tts":
                input_text = meta["text"] if "text" in meta else meta["text_tn"]
                target_speech = arrays["target_tokens"]
            elif task == "speech_to_text_translation":
                input_speech = arrays["input_tokens"]
                target_text = meta["target_text"]
            elif task == "spoken_extract_qa":
                input_speech = arrays["input_tokens"]
                question_text = meta["question"]
                target_text = meta["answer_text"]
            elif task == "text_dialog_sft":
                input_text_token = arrays["input_tokens"]
                target_text_token =arrays["target_tokens"]
        else:
            raise ValueError(f"Unsupported task: {task}")

        yield RawSample(
            sample_id=meta.get("id", meta.get("sample_id", sample.get("__key__", "unknown"))),
            dataset=dataset_name,
            split=split,
            task=task,
            instruction=instruction,
            input_text=input_text,
            input_text_token=input_text_token,
            target_text=target_text,
            target_text_token=target_text_token,
            input_speech=input_speech,
            target_speech=target_speech,
            question_text=question_text,
            meta=meta,
        )