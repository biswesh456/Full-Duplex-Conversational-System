import argparse
import io
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import webdataset as wds
from tqdm import tqdm
from transformers import AutoTokenizer


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()


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


def normalize_role(role: Any) -> str:
    if not isinstance(role, str):
        return ""
    role = role.strip().lower()

    if role in ["human", "user"]:
        return "user"
    if role in ["assistant", "gpt", "bot"]:
        return "assistant"
    if role == "system":
        return "system"
    return role


def extract_message_role(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    if 'role' in msg:
        return normalize_role(msg['role'])
    return ""


def extract_message_content(msg: Any) -> str:
    if isinstance(msg, str):
        return normalize_text(msg)

    if not isinstance(msg, dict):
        return ""
    
    value = msg.get('content')
    if isinstance(value, str) and value.strip():
        return normalize_text(value)

    return ""


def extract_messages(row: dict) -> tuple[Optional[list[dict]], int, int]:
    key = "messages"
    value = row.get(key)

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, list) or len(value) == 0:
        return None, 0, 0

    messages = []
    no_content = 0
    no_role = 0
    for item in value:
        role = extract_message_role(item)
        content = extract_message_content(item)

        if not content:
            no_content += 1 
            continue

        if role not in {"system", "user", "assistant"}:
            no_role += 1
            continue

        messages.append({
            "role": role,
            "content": f"<text> {content}",
        })

    if messages:
        return messages, no_content, no_role

    return None, no_content, no_role


def render_prompt_with_chat_template(tokenizer, prompt_messages: list[dict]) -> str:
    """
    Uses the tokenizer's native chat template.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "This tokenizer does not support apply_chat_template(). "
            "Use an instruct/chat tokenizer or switch to manual formatting."
        )

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("apply_chat_template returned empty prompt text")

    return prompt_text.rstrip() + " <text> "

def extract_sft_pair(messages: list[dict]) -> Optional[tuple[list[dict], str]]:
    """
    Returns:
      prompt_messages = all turns before final assistant turn
      target_text     = final assistant message content
    """
    if not messages:
        return None

    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant" and messages[i]["content"]:
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return None
    if last_assistant_idx == 0:
        return None

    prompt_messages = messages[:last_assistant_idx]
    target_text = normalize_text(messages[last_assistant_idx]["content"])
    if target_text.startswith("<text> "):
        target_text = target_text[len("<text> "):]

    if not prompt_messages or not target_text:
        return None

    return prompt_messages, target_text


def process_split(
    split_name: str,
    parquet_files: list[Path],
    out_dir: Path,
    tokenizer,
    maxcount: int,
    min_prompt_chars: int,
    min_target_chars: int,
    max_prompt_tokens: Optional[int],
    max_target_tokens: Optional[int],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(out_dir / f"{split_name}-%06d.tar")
    sink = wds.ShardWriter(shard_pattern, maxcount=maxcount)

    num_rows_total = 0
    num_missing_messages = 0
    num_bad_pair = 0
    num_too_short = 0
    num_too_long = 0
    num_failed = 0
    num_written = 0
    num_content_fail = 0
    num_role_fail = 0

    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)

        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"building {split_name} from {parquet_path.name}"):
            num_rows_total += 1

            try:
                row_dict = row.to_dict()
                messages, no_content, no_role = extract_messages(row_dict)

                num_content_fail += no_content
                num_role_fail += no_role

                if messages is None:
                    num_missing_messages += 1
                    continue

                pair = extract_sft_pair(messages)
                if pair is None:
                    num_bad_pair += 1
                    continue

                prompt_messages, target_text = pair
                prompt_text = render_prompt_with_chat_template(tokenizer, prompt_messages)

                if len(prompt_text) < min_prompt_chars or len(target_text) < min_target_chars:
                    num_too_short += 1
                    continue

                input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                target_ids = tokenizer.encode(target_text, add_special_tokens=False)

                if max_prompt_tokens is not None and len(input_ids) > max_prompt_tokens:
                    num_too_long += 1
                    continue

                if max_target_tokens is not None and len(target_ids) > max_target_tokens:
                    num_too_long += 1
                    continue

                sample_id = f"ultrachat_200k-{split_name}-{num_rows_total - 1:09d}"

                meta = {
                    "id": sample_id,
                    "dataset": "ultrachat_200k",
                    "split": split_name,
                    "task": "text_dialog_sft",
                    "instruction": "<text> Continue the conversation and produce the assistant reply.",
                    "source_file": parquet_path.name,
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "messages": messages,
                    "num_input_tokens": len(input_ids),
                    "num_target_tokens": len(target_ids),
                }

                arrays = {
                    "input_tokens": np.asarray(input_ids, dtype=np.int32),
                    "target_tokens": np.asarray(target_ids, dtype=np.int32),
                }

                write_sample(sink, sample_id, meta, arrays)
                num_written += 1

            except Exception as e:
                num_failed += 1
                print(f"[WARN] failed on row {row_idx} in {parquet_path}: {e}")

    sink.close()

    print(f"\n===== SUMMARY: {split_name} =====")
    print(f"Parquet files: {[p.name for p in parquet_files]}")
    print(f"Rows total: {num_rows_total}")
    print(f"Rows missing messages: {num_missing_messages}")
    print(f"Rows with invalid SFT pair: {num_bad_pair}")
    print(f"Rows skipped for short text: {num_too_short}")
    print(f"Rows skipped for token length: {num_too_long}")
    print(f"Messages without content: {num_content_fail}")
    print(f"Messages without role: {num_role_fail}")
    print(f"Rows failed: {num_failed}")
    print(f"Samples written: {num_written}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ultrachat-dir",
        type=Path,
        required=True,
        help="Directory containing UltraChat parquet files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--maxcount", type=int, default=10000)

    parser.add_argument("--min-prompt-chars", type=int, default=1)
    parser.add_argument("--min-target-chars", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)

    args = parser.parse_args()

    ultrachat_dir = args.ultrachat_dir
    if not ultrachat_dir.exists():
        raise FileNotFoundError(f"Missing UltraChat directory: {ultrachat_dir}")

    train_sft_files = sorted(ultrachat_dir.glob("train_sft-*.parquet"))
    test_sft_files = sorted(ultrachat_dir.glob("test_sft-*.parquet"))

    if not train_sft_files:
        raise FileNotFoundError(f"No train_sft parquet files found in {ultrachat_dir}")
    if not test_sft_files:
        raise FileNotFoundError(f"No test_sft parquet files found in {ultrachat_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    process_split(
        split_name="train",
        parquet_files=train_sft_files,
        out_dir=args.output_dir / "train",
        tokenizer=tokenizer,
        maxcount=args.maxcount,
        min_prompt_chars=args.min_prompt_chars,
        min_target_chars=args.min_target_chars,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
    )

    process_split(
        split_name="test",
        parquet_files=test_sft_files,
        out_dir=args.output_dir / "test",
        tokenizer=tokenizer,
        maxcount=args.maxcount,
        min_prompt_chars=args.min_prompt_chars,
        min_target_chars=args.min_target_chars,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
    )


if __name__ == "__main__":
    main()
