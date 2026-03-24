import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from moshi.models import loaders

from tokenization.multimodal_tokenizer import MultimodalTokenizer


IGNORE_INDEX = -100


def load_mimi(device: str, num_codebooks: int, mimi_ckpt: str):
    mimi = loaders.get_mimi(mimi_ckpt, device=device)
    mimi.set_num_codebooks(num_codebooks)
    mimi.eval()
    return mimi


def load_sample_from_tar(
    tar_path: Path,
    sample_index: int = 0,
) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, bytes]] = {}

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]

        for member in members:
            name = Path(member.name).name
            stem = Path(name).stem
            suffix = Path(name).suffix

            if suffix not in {".json", ".npz"}:
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            blob = extracted.read()
            grouped.setdefault(stem, {})[suffix] = blob

    keys = sorted(k for k, v in grouped.items() if ".json" in v and ".npz" in v)
    if not keys:
        raise ValueError(f"No complete samples found in shard: {tar_path}")

    if sample_index < 0 or sample_index >= len(keys):
        raise IndexError(
            f"sample_index={sample_index} out of range; shard has {len(keys)} samples"
        )

    key = keys[sample_index]
    record = grouped[key]

    meta = json.loads(record[".json"].decode("utf-8"))
    with np.load(io.BytesIO(record[".npz"]), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    return key, meta, arrays


@torch.no_grad()
def decode_codes_to_wav(
    mimi,
    codes: np.ndarray,
    device: str,
) -> tuple[np.ndarray, int]:
    code_tensor = torch.from_numpy(codes).long().unsqueeze(0).to(device)  # [1, K, T]
    wav = mimi.decode(code_tensor)  # [1, 1, T] typically
    wav = wav.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return wav, 24000


def save_segment_wavs(
    segments: list[dict[str, Any]],
    prefix: str,
    mm_tokenizer: MultimodalTokenizer,
    mimi,
    device: str,
    out_dir: Path,
):
    for i, seg in enumerate(segments):
        print(
            f"  segment {i}: type={seg['type']} start={seg['start']} "
            f"end={seg['end']} len={len(seg['ids'])}"
        )

        if seg["type"] == "text":
            preview = mm_tokenizer.decode_text_ids(seg["ids"])
            preview = preview.replace("\n", "\\n")
            print(f"text preview: {preview}")
        else:
            codes = mm_tokenizer.speech_ids_to_codes(seg["ids"])
            wav, sr = decode_codes_to_wav(mimi, codes, device=device)
            wav_path = out_dir / f"{prefix}_segment_{i}.wav"
            sf.write(wav_path, wav, sr)
            print(f"saved wav: {wav_path}")
            print(f"recovered codes shape: {codes.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--mimi-ckpt", type=str, required=True)
    parser.add_argument("--num-codebooks", type=int, required=True)
    parser.add_argument("--speech-codebook-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("./inspect_out"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mm_tokenizer = MultimodalTokenizer.from_pretrained(
        pretrained_name_or_path=args.tokenizer,
        num_codebooks=args.num_codebooks,
        speech_codebook_size=args.speech_codebook_size,
        trust_remote_code=True,
    )

    mimi = load_mimi(
        device=args.device,
        num_codebooks=args.num_codebooks,
        mimi_ckpt=args.mimi_ckpt,
    )

    key, meta, arrays = load_sample_from_tar(args.tar, args.sample_index)

    input_ids = arrays["input_ids"].tolist()
    labels = arrays["labels"].tolist()
    attention_mask = arrays["attention_mask"].tolist()

    target_ids = [x for x in labels if x != IGNORE_INDEX]

    print("=" * 100)
    print(f"sample key: {key}")
    print(f"dataset: {meta.get('dataset')}")
    print(f"split: {meta.get('split')}")
    print(f"task: {meta.get('task')}")
    print(f"num input ids: {len(input_ids)}")
    print(f"num target ids: {len(target_ids)}")
    print(f"attention mask length: {len(attention_mask)}")
    print("=" * 100)

    with open(args.out_dir / f"{key}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        args.out_dir / f"{key}.packed_arrays.npz",
        input_ids=np.asarray(input_ids, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int32),
        attention_mask=np.asarray(attention_mask, dtype=np.int8),
    )

    input_text_only = [x for x in input_ids if not mm_tokenizer.is_speech_token(x)]
    input_text_decoded = mm_tokenizer.decode_text_ids(input_text_only)

    with open(args.out_dir / f"{key}.input_text.txt", "w", encoding="utf-8") as f:
        f.write(input_text_decoded)

    print("\n[full text-only projection of input_ids]\n")
    print(input_text_decoded[:4000])

    target_text_only = [x for x in target_ids if not mm_tokenizer.is_speech_token(x)]
    target_text_decoded = mm_tokenizer.decode_text_ids(target_text_only)

    with open(args.out_dir / f"{key}.target_text.txt", "w", encoding="utf-8") as f:
        f.write(target_text_decoded)

    print("\n[target text-only projection]\n")
    print(target_text_decoded[:2000])

    prompt_ids = [inp for inp, lab in zip(input_ids, labels) if lab == IGNORE_INDEX]
    target_ids = [lab for lab in labels if lab != IGNORE_INDEX]

    prompt_segments = mm_tokenizer.split_modalities(prompt_ids)
    target_segments = mm_tokenizer.split_modalities(target_ids)

    print("\n[prompt modality segments]")
    save_segment_wavs(
        segments=prompt_segments,
        prefix=f"{key}.input",
        mm_tokenizer=mm_tokenizer,
        mimi=mimi,
        device=args.device,
        out_dir=args.out_dir,
    )

    print("\n[target modality segments]")
    target_segments = mm_tokenizer.split_modalities(target_ids)
    save_segment_wavs(
        segments=target_segments,
        prefix=f"{key}.target",
        mm_tokenizer=mm_tokenizer,
        mimi=mimi,
        device=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()