import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Iterator, Optional

import ijson
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

def load_audio_segment(audio_path: Path, begin_time: float, end_time: float) -> tuple[torch.Tensor, int]:
    """
    Load only the requested segment from disk.
    """
    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        start_frame = max(0, int(round(begin_time * sr)))
        end_frame = int(round(end_time * sr))

        if end_frame <= start_frame:
            raise ValueError(
                f"Invalid segment boundaries: begin={begin_time}, end={end_time}"
            )

        num_frames = end_frame - start_frame
        f.seek(start_frame)
        wav = f.read(num_frames, dtype="float32", always_2d=True)  # [T, C]

    wav = torch.from_numpy(wav).transpose(0, 1)  # [C, T]

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True) # [1, T]

    return wav, sr


def resample_to_24k_mono(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 24000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=24000)

    return wav  # [1, T]

@torch.no_grad()
def encode_mimi(mimi, wav_24k: torch.Tensor, device: str) -> np.ndarray:
    wav_24k = wav_24k.unsqueeze(0).to(device)  # [B=1, C=1, T]
    codes = mimi.encode(wav_24k)               # [B, K, T]
    codes = codes.squeeze(0).detach().cpu().numpy()  # [K, T]
    return codes.astype(np.uint16)


SPECIAL_MAP = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!",
    "<COLON>": ":",
    "<SEMICOLON>": ";",
    "<APOSTROPHE>": "'",
    "<QUOTE>": '"',
    "<LEFTPAREN>": "(",
    "<RIGHTPAREN>": ")",
    "<HYPHEN>": "-",
}

GARBAGE_UTTERANCE_TAGS = {
    "<SIL>",
    "<MUSIC>",
    "<NOISE>",
    "<OTHER>",
}

def is_gigaspeech_garbage_utterance(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return text.strip() in GARBAGE_UTTERANCE_TAGS

def normalize_gigaspeech_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    out = text.strip()
    for k, v in SPECIAL_MAP.items():
        out = out.replace(k, v)

    # remove any leftover <...> tags not explicitly mapped
    out = re.sub(r"\s*<[^<>]+>\s*", " ", out)
    # remove spaces before punctuation
    out = re.sub(r"\s+([,.:;?!])", r"\1", out)
    # collapse repeated whitespace
    out = re.sub(r"\s+", " ", out).strip()

    return out.lower()


def iter_gigaspeech_audios_stream(json_path: Path) -> Iterator[dict]:
    with open(json_path, "rb") as f:
        yield from ijson.items(f, "audios.item")


def build_audio_index(audio_wav_root: Path) -> dict[str, Path]:
    index = {}
    for wav_path in tqdm(audio_wav_root.glob("*/*/*.wav"), desc="indexing wav files"):
        key = wav_path.stem
        if key in index:
            raise ValueError(f"Duplicate audio key found: {key}\n"
                             f"Existing: {index[key]}\nNew: {wav_path}")
        index[key] = wav_path
    return index


def subset_tag(size: str) -> str:
    return f"{{{size}}}"


def segment_matches_size(segment_subsets: list[str], size: str) -> bool:
    return subset_tag(size) in set(segment_subsets or [])


def choose_task_from_sid(sid: str, asr_ratio: float = 0.5) -> str:
    h = hashlib.md5(sid.encode("utf-8")).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF
    return "asr" if value < asr_ratio else "tts"


def write_sample(sink, sample_id: str, meta: dict, arrays: dict):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)

    sample = {
        "__key__": sample_id,
        "json": json.dumps(meta).encode("utf-8"),
        "npz": buf.getvalue(),
    }
    sink.write(sample)


def process_gigaspeech(
    metadata_json: Path,
    audio_wav_root: Path,
    out_dir: Path,
    tokenizer,
    mimi,
    device: str,
    num_codebooks: int,
    maxcount: int,
    task: str,
    size: str,
    asr_ratio: float,
    min_duration: float,
    max_duration: float,
    dry_run_only_check: bool = False,
    start_audio_index: int = 0,
    end_audio_index: Optional[int] = None,
    num_workers: int = 1,
    worker_id: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_index = build_audio_index(audio_wav_root)

    sink = None
    if not dry_run_only_check:
        shard_pattern = str(
            out_dir / f"{size.lower()}-{task}-w{worker_id:02d}-%06d.tar"
        )
        sink = wds.ShardWriter(shard_pattern, maxcount=maxcount)

    num_audio_streamed = 0
    num_audio_selected_by_partition = 0
    num_audio_found = 0
    num_audio_missing = 0

    num_segments_total_streamed = 0
    num_segments_matching_size = 0
    num_segments_missing_audio = 0
    num_segments_empty_text = 0
    num_segments_bad_duration = 0
    num_segments_failed = 0
    num_segments_garbage = 0

    num_samples_written = 0
    num_asr_written = 0
    num_tts_written = 0

    for audio_idx, audio_entry in enumerate(
        tqdm(iter_gigaspeech_audios_stream(metadata_json))
    ):
        num_audio_streamed += 1

        if audio_idx < start_audio_index:
            continue
        if end_audio_index is not None and audio_idx >= end_audio_index:
            break
        if audio_idx % num_workers != worker_id:
            continue

        num_audio_selected_by_partition += 1

        aid = audio_entry["aid"]
        source = audio_entry.get("source", "")
        audio_subsets = audio_entry.get("subsets", [])
        segments = audio_entry.get("segments", [])

        selected_segments = []
        for seg in segments:
            num_segments_total_streamed += 1
            if segment_matches_size(seg.get("subsets", []), size):
                num_segments_matching_size += 1
                selected_segments.append(seg)

        if not selected_segments:
            continue

        wav_path = audio_index.get(aid)
        if wav_path is None:
            num_audio_missing += 1
            num_segments_missing_audio += len(selected_segments)
            continue

        num_audio_found += 1

        if dry_run_only_check:
            continue

        for seg in selected_segments:
            try:
                sid = seg["sid"]
                begin_time = float(seg["begin_time"])
                end_time = float(seg["end_time"])
                seg_subsets = seg.get("subsets", [])
                raw_text = seg.get("text_tn", "")

                if is_gigaspeech_garbage_utterance(raw_text):
                    num_segments_garbage += 1
                    continue

                text_tn = normalize_gigaspeech_text(raw_text)

                if not text_tn:
                    num_segments_empty_text += 1
                    continue

                duration_sec = end_time - begin_time
                if duration_sec < min_duration or duration_sec > max_duration:
                    num_segments_bad_duration += 1
                    continue

                seg_wav, sr = load_audio_segment(wav_path, begin_time, end_time)
                seg_wav_24k = resample_to_24k_mono(seg_wav, sr)

                speech_tokens = encode_mimi(mimi, seg_wav_24k, device=device)
                text_ids = tokenizer.encode(text_tn, add_special_tokens=False)

                assigned_task = task
                if task == "mixed":
                    assigned_task = choose_task_from_sid(sid, asr_ratio=asr_ratio)

                common_meta = {
                    "dataset": "gigaspeech",
                    "aid": aid,
                    "source": source,
                    "segment_subsets": seg_subsets,
                    "task": assigned_task,
                    "asr_ratio_requested": asr_ratio if task == "mixed" else None,
                    "begin_time": begin_time,
                    "end_time": end_time,
                    "duration_sec": duration_sec,
                    "text_tn_raw": raw_text,
                    "text_tn": text_tn,
                    "wav_path": str(wav_path),
                    "num_codebooks": num_codebooks,
                    "audio_index_in_metadata": audio_idx
                }

                if assigned_task in ("asr", "both"):
                    sample_id = f"gigaspeech-{size.lower()}-asr-{sid}"
                    meta = {
                        **common_meta,
                        "sample_id": sample_id,
                        "instruction": "<text> Transcribe the speech into text. <speech>",
                    }
                    if assigned_task == "both":
                        meta['task'] = "asr"
                    arrays = {
                        "input_tokens": speech_tokens,
                        "target_tokens": np.asarray(text_ids, dtype=np.int32),
                    }
                    write_sample(sink, sample_id, meta, arrays)
                    num_samples_written += 1
                    num_asr_written += 1

                if assigned_task in ("tts", "both"):
                    sample_id = f"gigaspeech-{size.lower()}-tts-{sid}"
                    meta = {
                        **common_meta,
                        "sample_id": sample_id,
                        "instruction": "<text> Synthesize speech for the given text - ",
                    }
                    if assigned_task == "both":
                        meta['task'] = "tts"
                    arrays = {
                        "input_tokens": np.asarray(text_ids, dtype=np.int32),
                        "target_tokens": speech_tokens,
                    }
                    write_sample(sink, sample_id, meta, arrays)
                    num_samples_written += 1
                    num_tts_written += 1

            except Exception as e:
                num_segments_failed += 1
                print(f"[WARN] failed on sid={seg.get('sid', 'NA')} aid={aid}: {e}")

    if sink is not None:
        sink.close()

    print("\n===== SUMMARY =====")
    print(f"Requested size: {size}")
    print(f"Task mode: {task}")
    if task == "mixed":
        print(f"ASR ratio requested: {asr_ratio}")
    print(f"Partition: start={start_audio_index}, end={end_audio_index}, "
          f"num_workers={num_workers}, worker_id={worker_id}")
    print(f"Total audio entries streamed before partition filtering: {num_audio_streamed}")
    print(f"Audio entries selected by partition: {num_audio_selected_by_partition}")
    print(f"Audio files indexed locally: {len(audio_index)}")
    print(f"Audio entries found for selected samples: {num_audio_found}")
    print(f"Audio entries missing for selected samples: {num_audio_missing}")
    print(f"Segments seen inside selected partition: {num_segments_total_streamed}")
    print(f"Segments matching requested size: {num_segments_matching_size}")
    print(f"Segments missing because parent wav missing/unreadable: {num_segments_missing_audio}")
    print(f"Segments skipped for empty text: {num_segments_empty_text}")
    print(f"Segments skipped for duration: {num_segments_bad_duration}")
    print(f"Segments failed during processing: {num_segments_failed}")
    print(f"Segments skipped as garbage utterances: {num_segments_garbage}")
    print(f"Total samples written: {num_samples_written}")
    print(f"ASR samples written: {num_asr_written}")
    print(f"TTS samples written: {num_tts_written}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gigaspeech-dir", type=Path, required=True)
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
        help="asr: speech->text, tts: text->speech, both: both per segment, mixed: assign one direction per segment",
    )
    parser.add_argument("--asr-ratio", type=float, default=0.5, help="Used only when --task mixed. Fraction assigned to ASR.")
    parser.add_argument(
        "--size",
        type=str,
        choices=["XS", "S", "M", "L", "XL", "DEV", "TEST"],
        default="XL",
        help="Which GigaSpeech subset tag to keep.",
    )
    parser.add_argument("--min-duration", type=float, default=0.2)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument(
        "--dry-run-only-check",
        action="store_true",
        help="Only check metadata/audio availability and counts; do not write shards.",
    )

    # Parallel / partitioning arguments
    parser.add_argument("--start-audio-index", type=int, default=0)
    parser.add_argument("--end-audio-index", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)

    args = parser.parse_args()

    if not (0.0 <= args.asr_ratio <= 1.0):
        raise ValueError("--asr-ratio must be between 0.0 and 1.0")

    if args.start_audio_index < 0:
        raise ValueError("--start-audio-index must be >= 0")

    if args.end_audio_index is not None and args.end_audio_index <= args.start_audio_index:
        raise ValueError("--end-audio-index must be > --start-audio-index")

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"--worker-id must be in [0, {args.num_workers - 1}]")

    metadata_json = args.gigaspeech_dir / "GigaSpeech.json"
    audio_wav_root = args.gigaspeech_dir / "audio_wav"

    if not metadata_json.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_json}")
    if not audio_wav_root.exists():
        raise FileNotFoundError(f"Missing audio_wav dir: {audio_wav_root}")

    tokenizer = None
    mimi = None
    if not args.dry_run_only_check:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        mimi = load_mimi(
            device=args.device,
            num_codebooks=args.num_codebooks,
            mimi_ckpt=args.mimi_ckpt,
        )

    process_gigaspeech(
        metadata_json=metadata_json,
        audio_wav_root=audio_wav_root,
        out_dir=args.output_dir,
        tokenizer=tokenizer,
        mimi=mimi,
        device=args.device,
        num_codebooks=args.num_codebooks,
        maxcount=args.maxcount,
        task=args.task,
        size=args.size,
        asr_ratio=args.asr_ratio,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        dry_run_only_check=args.dry_run_only_check,
        start_audio_index=args.start_audio_index,
        end_audio_index=args.end_audio_index,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
    )


if __name__ == "__main__":
    main()