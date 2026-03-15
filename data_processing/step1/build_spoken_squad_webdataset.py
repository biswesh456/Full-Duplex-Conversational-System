import argparse
import io
import json
from pathlib import Path
from typing import Optional

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

    # If two channels i.e. stereo instead of mono then just take a mean out of it.
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


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_sentence_audio_files(audio_dir: Path, topic_idx: int, para_idx: int) -> list[Path]:
    """
    Returns sorted files matching:
      {topic_idx}_{para_idx}_{sentence_idx}.wav
    """
    pattern = f"{topic_idx}_{para_idx}_*.wav"
    files = list(audio_dir.glob(pattern))

    def sentence_index(p: Path) -> int:
        # filename like 12_3_5.wav
        parts = p.stem.split("_")
        return int(parts[2])

    files = sorted(files, key=sentence_index)
    return files


def concat_wavs(audio_paths: list[Path]) -> tuple[torch.Tensor, float]:
    """
    Concatenate sentence-level wavs into one paragraph-level waveform.
    Returns [B=1, C=1, T], duration_sec
    """
    wavs = []
    total_duration = 0.0

    for p in audio_paths:
        wav, duration = load_audio_24k_mono(p)
        wavs.append(wav)  # [1, 1, T]
        total_duration += duration

    if not wavs:
        raise ValueError("No audio files to concatenate.")

    full_wav = torch.cat(wavs, dim=-1)  # [1, 1, T_total]
    return full_wav, total_duration


def get_first_answer_text(qa: dict) -> Optional[str]:
    answers = qa.get("answers", [])
    if not answers:
        return None

    first = answers[0]
    text = first.get("text")
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None
    return text


def iter_spoken_squad_examples(dataset_json: dict):
    """
    Yields one example per QA pair.
    """
    data = dataset_json.get("data", [])

    for topic_idx, article in enumerate(data):
        title = article.get("title", "")
        paragraphs = article.get("paragraphs", [])

        for para_idx, paragraph in enumerate(paragraphs):
            context = paragraph.get("context", "")
            qas = paragraph.get("qas", [])

            for qa in qas:
                qid = qa.get("id", "")
                question = qa.get("question", "")
                answer_text = get_first_answer_text(qa)

                if not isinstance(question, str) or len(question.strip()) == 0:
                    continue
                if answer_text is None:
                    continue

                yield {
                    "topic_idx": topic_idx,
                    "para_idx": para_idx,
                    "title": title,
                    "context": context,
                    "qa_id": qid,
                    "question": question,
                    "answer_text": answer_text,
                    "answers": qa.get("answers", []),
                }


def write_split(
    split_name: str,
    json_path: Path,
    audio_dir: Path,
    out_dir: Path,
    tokenizer,
    mimi,
    device: str,
    num_codebooks: int,
    maxcount: int,
):
    dataset_json = load_json(json_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(out_dir / f"{split_name}-%06d.tar")
    sink = wds.ShardWriter(shard_pattern, maxcount=maxcount)

    instruction = "<text> Answer the question using the spoken passage. <speech>"

    num_written = 0
    num_missing_audio = 0
    num_failed = 0

    examples = list(iter_spoken_squad_examples(dataset_json))

    for ex_idx, ex in tqdm(enumerate(examples), total=len(examples), desc=f"building {split_name}"):
        if ex_idx == 0:
            print(ex, flush=True)
        topic_idx = ex["topic_idx"]
        para_idx = ex["para_idx"]

        try:
            sentence_audio_files = find_sentence_audio_files(audio_dir, topic_idx, para_idx)

            if len(sentence_audio_files) == 0:
                num_missing_audio += 1
                print(
                    f"[WARN] no audio found for topic={topic_idx} para={para_idx} "
                    f"in {audio_dir}"
                )
                continue

            paragraph_wav, duration_sec = concat_wavs(sentence_audio_files)
            speech_tokens = encode_mimi(mimi, paragraph_wav, device=device)

            question_ids = tokenizer.encode(ex["question"], add_special_tokens=False)
            target_ids = tokenizer.encode(ex["answer_text"], add_special_tokens=False)

            meta = {
                "id": f"spoken_squad-{split_name}-{ex_idx:09d}",
                "dataset": "spoken_squad",
                "split": split_name,
                "task": "spoken_extract_qa",
                "instruction": instruction,
                "title": ex["title"],
                "qa_id": ex["qa_id"],
                "topic_idx": topic_idx,
                "para_idx": para_idx,
                "duration_sec": duration_sec,
                "question": ex["question"],
                "answer_text": ex["answer_text"],
                "context_text": ex["context"],
                "num_codebooks": num_codebooks,
                "audio_files": [p.name for p in sentence_audio_files],
            }

            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                input_tokens=speech_tokens,  # [K, T]
                question_tokens=np.asarray(question_ids, dtype=np.int32),
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
            print(
                f"[WARN] failed on example {ex_idx} "
                f"(topic={topic_idx}, para={para_idx}, qa_id={ex['qa_id']}): {e}"
            )

    sink.close()
    print(
        f"[{split_name}] written={num_written} "
        f"missing_audio={num_missing_audio} failed={num_failed}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spoken-squad-dir",
        type=Path,
        required=True,
        help="Directory containing spoken_train-v1.1.json / spoken_test-v1.1.json",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Root directory containing Spoken-SQuAD_audio/train and Spoken-SQuAD_audio/dev",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HF tokenizer name/path for your LLM",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--maxcount", type=int, default=5000)
    parser.add_argument("--mimi-ckpt", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    mimi = load_mimi(
        device=args.device,
        num_codebooks=args.num_codebooks,
        mimi_ckpt=args.mimi_ckpt,
    )

    split_specs = {
        "train": {
            "json": args.spoken_squad_dir / "spoken_train-v1.1.json",
            "audio": args.audio_root / "train_wav",
        },
        "dev": {
            "json": args.spoken_squad_dir / "spoken_test-v1.1.json",
            "audio": args.audio_root / "dev_wav",
        },
    }

    for split_name, spec in split_specs.items():
        if not spec["json"].exists():
            raise FileNotFoundError(f"Missing json file: {spec['json']}")
        if not spec["audio"].exists():
            raise FileNotFoundError(f"Missing audio dir: {spec['audio']}")

        write_split(
            split_name=split_name,
            json_path=spec["json"],
            audio_dir=spec["audio"],
            out_dir=args.output_dir / split_name,
            tokenizer=tokenizer,
            mimi=mimi,
            device=args.device,
            num_codebooks=args.num_codebooks,
            maxcount=args.maxcount,
        )


if __name__ == "__main__":
    main()