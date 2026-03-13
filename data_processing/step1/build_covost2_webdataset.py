# Converts Covost2 dataset into a webdataset. 
# It contains a .json containing the meta information and .npz containing the audio and text tokens.

import argparse
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
import webdataset as wds
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from tqdm import tqdm
from transformers import AutoTokenizer


def load_mimi(device: str, num_codebooks: int, mimi_ckpt: str):
    mimi = loaders.get_mimi(mimi_ckpt, device=device)
    mimi.set_num_codebooks(num_codebooks)
    mimi.eval()
    return mimi


def load_audio_24k_mono(audio_path: Path) -> tuple[torch.Tensor, float]:
    wav, sr = sf.read(str(audio_path), always_2d=True)  # [T, C]
    wav = torch.from_numpy(wav).float().transpose(0, 1)  # [C, T]

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 24000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=24000)
        sr = 24000

    duration_sec = wav.shape[-1] / sr
    return wav.unsqueeze(0), duration_sec  # [B=1, C=1, T]


@torch.no_grad()
def encode_mimi(mimi, wav: torch.Tensor, device: str) -> np.ndarray:
    wav = wav.to(device)
    codes = mimi.encode(wav)  # [B, K, T]
    codes = codes.squeeze(0).detach().cpu().numpy()  # [K, T]
    return codes.astype(np.uint16)

def read_covost_tsv(split_tsv: Path) -> pd.DataFrame:
    rows = []
    with open(split_tsv, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        expected = ["path", "sentence", "translation", "client_id"]
        if header != expected:
            raise ValueError(f"Unexpected header in {split_tsv}: {header}")

        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                print(f"[WARN] bad line {lineno} in {split_tsv}: expected 4 fields, got {len(parts)}")
                continue
            rows.append(parts)

    return pd.DataFrame(rows, columns=["path", "sentence", "translation", "client_id"])

def write_split(
    split_name: str,
    split_tsv: Path,
    audio_root: Path,
    out_dir: Path,
    tokenizer,
    mimi,
    device: str,
    num_codebooks: int,
    maxcount: int,
):
    df = read_covost_tsv(split_tsv)

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(out_dir / f"{split_name}-%06d.tar")
    sink = wds.ShardWriter(shard_pattern, maxcount=maxcount)

    instruction = "<text> Translate this speech from English to German. <speech>"

    num_written = 0
    num_missing = 0
    num_failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"building {split_name}"):
        audio_path = audio_root / Path(row["path"]).with_suffix(".wav")

        if not audio_path.exists():
            num_missing += 1
            print(f"[WARN] missing audio: {audio_path}")
            continue

        try:
            wav, duration_sec = load_audio_24k_mono(audio_path)
            speech_tokens = encode_mimi(mimi, wav, device=device)

            if not isinstance(row["sentence"], str):
                print(f"[WARN] missing sentence: {audio_path}")
                continue
            source_text = row["sentence"]

            if not isinstance(row["translation"], str):
                print(f"[WARN] missing translation: {audio_path}")
                continue
            target_text = row["translation"]

            client_id = row["client_id"] if isinstance(row.get("client_id"), str) else ""

            target_ids = tokenizer.encode(target_text, add_special_tokens=False)

            meta = {
                "id": f"covost2-en_de-{split_name}-{idx:09d}",
                "dataset": "covost2",
                "split": split_name,
                "task": "speech_to_text_translation",
                "instruction": instruction,
                "src_tgt_lang": "en_de",
                "audio_relpath": row["path"],
                "duration_sec": duration_sec,
                "source_text": source_text,
                "target_text": target_text,
                "num_codebooks": num_codebooks,
                "client_id": client_id
            }

            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                input_tokens=speech_tokens,  # [K, T], uint16
                target_tokens=np.asarray(target_ids, dtype=np.int32),
            )
            buf.seek(0)

            sample = {
                "__key__": meta["id"],
                "json": json.dumps(meta).encode("utf-8"),
                "npz": buf.getvalue(),
            }
            sink.write(sample)
            num_written += 1

        except Exception as e:
            num_failed += 1
            print(f"[WARN] failed on row {idx} file={audio_path}: {e}")

    sink.close()
    print(f"[{split_name}] written={num_written} missing_audio={num_missing} failed={num_failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--covost-dir", type=Path, required=True,
                        help="Directory containing covost_v2.en_de.train.tsv / dev / test")
    parser.add_argument("--audio-root", type=Path, required=True,
                        help="Root directory containing audio files or clips/new-clip")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="HF tokenizer name/path for your LLM")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--maxcount", type=int, default=10000,
                        help="Samples per tar shard")
    parser.add_argument("--mimi-ckpt", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    mimi = load_mimi(device=args.device, num_codebooks=args.num_codebooks, mimi_ckpt=args.mimi_ckpt)

    split_files = {
        "train": args.covost_dir / "covost_v2.en_de.train.tsv",
        "dev": args.covost_dir / "covost_v2.en_de.dev.tsv",
        "test": args.covost_dir / "covost_v2.en_de.test.tsv",
    }

    for split_name, split_tsv in split_files.items():
        if not split_tsv.exists():
            raise FileNotFoundError(f"Missing split TSV: {split_tsv}")

        write_split(
            split_name=split_name,
            split_tsv=split_tsv,
            audio_root=args.audio_root,
            out_dir=args.output_dir / split_name,
            tokenizer=tokenizer,
            mimi=mimi,
            device=args.device,
            num_codebooks=args.num_codebooks,
            maxcount=args.maxcount,
        )


if __name__ == "__main__":
    main()