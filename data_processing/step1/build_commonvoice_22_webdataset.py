import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import webdataset as wds
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
    wav = wav.to(device)  # [1, 1, T]
    codes = mimi.encode(wav)  # [1, K, T']
    codes = codes.squeeze(0).detach().cpu().numpy()  # [K, T']
    return codes.astype(np.uint16)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def choose_task_from_id(sample_id: str, asr_ratio: float = 0.5) -> str:
    h = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF
    return "asr" if value < asr_ratio else "tts"


def write_sample(sink, sample_id: str, meta: dict, arrays: dict):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)

    sample = {
        "__key__": sample_id,
        "json": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        "npz": buf.getvalue(),
    }
    sink.write(sample)


def iter_commonvoice_rows(tsv_path: Path) -> Iterator[dict]:
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def get_sentence_field(row: dict) -> str:
    for key in ["sentence", "text", "transcript"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_path_field(row: dict) -> str:
    for key in ["path", "audio", "audio_path", "filename"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def resolve_audio_path(audio_root: Path, relpath: str) -> Path:
    rel = Path(relpath)

    # most likely case: metadata says .mp3, local directory contains .wav
    wav_candidate = audio_root / rel.with_suffix(".wav").name
    if wav_candidate.exists():
        return wav_candidate

    # maybe already wav or exact filename exists
    direct_candidate = audio_root / rel.name
    if direct_candidate.exists():
        return direct_candidate

    # fallback if relpath has subdirs
    fallback = audio_root / rel
    if fallback.exists():
        return fallback

    return wav_candidate

def count_tsv_rows(tsv_path: Path) -> int:
    with open(tsv_path, "r", encoding="utf-8") as f:
        # subtract header
        return max(0, sum(1 for _ in f) - 1)

def process_commonvoice(
    tsv_path: Path,
    audio_root: Path,
    out_dir: Path,
    tokenizer,
    mimi,
    device: str,
    num_codebooks: int,
    maxcount: int,
    split_name: str,
    task: str,
    asr_ratio: float,
    min_duration: float,
    max_duration: float,
    dry_run_only_check: bool = False,
    start_index: int = 0,
    end_index: Optional[int] = None,
    num_workers: int = 1,
    worker_id: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    sink = None
    if not dry_run_only_check:
        shard_pattern = str(out_dir / f"{split_name}-{task}-w{worker_id:02d}-%06d.tar")
        sink = wds.ShardWriter(shard_pattern, maxcount=maxcount)

    num_rows_streamed = 0
    num_rows_selected_by_partition = 0
    num_rows_missing_audio = 0
    num_rows_empty_text = 0
    num_rows_bad_duration = 0
    num_rows_failed = 0

    num_samples_written = 0
    num_asr_written = 0
    num_tts_written = 0

    total_rows = count_tsv_rows(tsv_path)

    for row_idx, row in enumerate(tqdm(iter_commonvoice_rows(tsv_path), total=total_rows, desc=f"building {split_name}")):
        num_rows_streamed += 1

        if row_idx < start_index:
            continue
        if end_index is not None and row_idx >= end_index:
            break
        if row_idx % num_workers != worker_id:
            continue

        num_rows_selected_by_partition += 1

        relpath = row.get("path", "")
        relpath = relpath.strip() if isinstance(relpath, str) else ""
        if not relpath:
            num_rows_failed += 1
            print(f"[WARN] missing path at row {row_idx}")
            continue

        text = normalize_text(get_sentence_field(row))
        if not text:
            num_rows_empty_text += 1
            continue

        audio_path = resolve_audio_path(audio_root, relpath)
        if not audio_path.exists():
            num_rows_missing_audio += 1
            print(f"[WARN] missing audio: {audio_path} (from relpath={relpath})")
            continue

        try:
            wav, duration_sec = load_audio_24k_mono(audio_path)

            if duration_sec < min_duration or duration_sec > max_duration:
                num_rows_bad_duration += 1
                continue

            if dry_run_only_check:
                continue

            speech_tokens = encode_mimi(mimi, wav, device=device)
            text_ids = tokenizer.encode(text, add_special_tokens=False)

            base_id = f"commonvoice-en-{split_name}-{row_idx:09d}"

            assigned_task = task
            if task == "mixed":
                assigned_task = choose_task_from_id(base_id, asr_ratio=asr_ratio)

            common_meta = {
                "dataset": "commonvoice",
                "split": split_name,
                "task": assigned_task,
                "audio_path": str(audio_path),
                "text": text,
                "duration_sec": duration_sec,
                "num_codebooks": num_codebooks,
                "segment": row.get("segment", ""),
            }

            if assigned_task in ("asr", "both"):
                sample_id = f"{base_id}-asr"
                meta = {
                    **common_meta,
                    "id": sample_id,
                    "task": "asr",
                    "instruction": "<text> Transcribe the speech into text. <speech>",
                }
                arrays = {
                    "input_tokens": speech_tokens,
                    "target_tokens": np.asarray(text_ids, dtype=np.int32),
                }
                write_sample(sink, sample_id, meta, arrays)
                num_samples_written += 1
                num_asr_written += 1

            if assigned_task in ("tts", "both"):
                sample_id = f"{base_id}-tts"
                meta = {
                    **common_meta,
                    "id": sample_id,
                    "task": "tts",
                    "instruction": "<text> Synthesize speech for the given text - ",
                }
                arrays = {
                    "input_tokens": np.asarray(text_ids, dtype=np.int32),
                    "target_tokens": speech_tokens,
                }
                write_sample(sink, sample_id, meta, arrays)
                num_samples_written += 1
                num_tts_written += 1

        except Exception as e:
            num_rows_failed += 1
            print(f"[WARN] failed on row {row_idx} file={audio_path}: {e}")

    if sink is not None:
        sink.close()

    print("\n===== SUMMARY =====")
    print(f"TSV: {tsv_path}")
    print(f"Split: {split_name}")
    print(f"Task mode: {task}")
    if task == "mixed":
        print(f"ASR ratio requested: {asr_ratio}")
    print(
        f"Partition: start={start_index}, end={end_index}, "
        f"num_workers={num_workers}, worker_id={worker_id}"
    )
    print(f"Rows streamed before partition filtering: {num_rows_streamed}")
    print(f"Rows selected by partition: {num_rows_selected_by_partition}")
    print(f"Rows missing audio: {num_rows_missing_audio}")
    print(f"Rows skipped for empty text: {num_rows_empty_text}")
    print(f"Rows skipped for duration: {num_rows_bad_duration}")
    print(f"Rows failed during processing: {num_rows_failed}")
    print(f"Total samples written: {num_samples_written}")
    print(f"ASR samples written: {num_asr_written}")
    print(f"TTS samples written: {num_tts_written}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commonvoice-dir",
        type=Path,
        required=True,
        help="Directory containing train.tsv/dev.tsv/test.tsv/validated.tsv and clips_wav",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--mimi-ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--maxcount", type=int, default=10000)

    parser.add_argument(
        "--task",
        type=str,
        choices=["asr", "tts", "both", "mixed"],
        default="mixed",
        help="asr: speech->text, tts: text->speech, both: both per sample, mixed: assign one direction per sample",
    )
    parser.add_argument(
        "--asr-ratio",
        type=float,
        default=0.5,
        help="Used only when --task mixed. Fraction assigned to ASR.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test", "validated"],
        default="train",
    )
    parser.add_argument("--min-duration", type=float, default=0.2)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument(
        "--dry-run-only-check",
        action="store_true",
        help="Only check metadata/audio availability and counts; do not write shards.",
    )

    # Parallel / partitioning arguments
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)

    args = parser.parse_args()

    if not (0.0 <= args.asr_ratio <= 1.0):
        raise ValueError("--asr-ratio must be between 0.0 and 1.0")

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")

    if args.end_index is not None and args.end_index <= args.start_index:
        raise ValueError("--end-index must be > --start-index")

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"--worker-id must be in [0, {args.num_workers - 1}]")

    commonvoice_dir = args.commonvoice_dir
    tsv_path = commonvoice_dir / f"{args.split}.tsv"
    audio_root = commonvoice_dir / "clips_wav"

    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing TSV file: {tsv_path}")

    if not audio_root.exists():
        raise FileNotFoundError(f"Missing audio root: {audio_root}")

    tokenizer = None
    mimi = None
    if not args.dry_run_only_check:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        mimi = load_mimi(
            device=args.device,
            num_codebooks=args.num_codebooks,
            mimi_ckpt=args.mimi_ckpt,
        )

    process_commonvoice(
        tsv_path=tsv_path,
        audio_root=audio_root,
        out_dir=args.output_dir / args.split,
        tokenizer=tokenizer,
        mimi=mimi,
        device=args.device,
        num_codebooks=args.num_codebooks,
        maxcount=args.maxcount,
        split_name=args.split,
        task=args.task,
        asr_ratio=args.asr_ratio,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        dry_run_only_check=args.dry_run_only_check,
        start_index=args.start_index,
        end_index=args.end_index,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
    )


if __name__ == "__main__":
    main()
