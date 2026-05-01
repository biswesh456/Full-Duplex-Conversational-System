import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader
from moshi.models import loaders
from collections import defaultdict
from jiwer import wer as jiwer_wer
import sacrebleu

from training.step1.data.collator import PackedCollator, IGNORE_INDEX
from training.step1.data.packed_webdataset import EvalDataset, parse_eval_specs
from training.models.qwen_causal_lm import build_qwen_causal_lm
from training.strategies.fsdp import build_fsdp_strategy
from training.utils.config import load_yaml, ensure_dir
from training.utils.logging import print_rank_zero
from tokenization.multimodal_tokenizer import MultimodalTokenizer
from tokenization.config_checker import assert_same_tokenizer_config


def infer_text_metric(task_name: str, dataset_name: str) -> str | None:
    task = (task_name or "").lower()
    ds = (dataset_name or "").lower()

    if any(x in task for x in ["asr", "transcribe"]):
        return "wer"

    if "commonvoice" in ds or "gigaspeech" in ds:
        if "tts" not in ds:
            return "wer"

    if any(x in task for x in ["translate", "translation", "speech_to_text"]):
        return "bleu"

    if "covost" in ds:
        return "bleu"

    if "spoken_squad" in ds or "ultrachat" in ds:
        return "exact_match"

    return "exact_match"


def safe_normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def compute_text_metric(metric_name: str, pred_text: str, target_text: str) -> float | None:
    pred = safe_normalize_text(pred_text)
    ref = safe_normalize_text(target_text)

    if len(ref) == 0:
        return None

    if metric_name == "wer":
        return float(jiwer_wer(ref, pred))

    if metric_name == "bleu":
        return float(sacrebleu.corpus_bleu([pred], [[ref]]).score)

    if metric_name == "exact_match":
        return float(pred == ref)

    return None


def collect_dataset_roots_for_tokenizer_check(cfg: dict) -> list[str]:
    roots: list[str] = []

    for ds in cfg["data"].get("test_sets", []):
        urls = ds["urls"]
        if isinstance(urls, str):
            roots.append(urls)
        else:
            roots.extend(urls)

    if len(roots) == 0:
        for ds in cfg["data"].get("val_sets", []):
            urls = ds["urls"]
            if isinstance(urls, str):
                roots.append(urls)
            else:
                roots.extend(urls)

    deduped = list(dict.fromkeys(roots))
    return deduped


def build_fabric(cfg: dict, num_nodes: int, num_gpus_per_node: int) -> Fabric:
    fabric_cfg = cfg["fabric"]

    plugins = []
    if "SLURM_JOB_ID" in os.environ:
        plugins.append(SLURMEnvironment())

    strategy_name = fabric_cfg.get("strategy", "fsdp")
    if strategy_name != "fsdp":
        raise ValueError("This eval script currently supports only strategy=fsdp")

    strategy = build_fsdp_strategy(fabric_cfg)

    fabric = Fabric(
        accelerator=fabric_cfg.get("accelerator", "cuda"),
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        precision=fabric_cfg.get("precision", "bf16-mixed"),
        strategy=strategy,
        plugins=plugins,
    )
    return fabric

def log_line(fabric: Fabric | None, *args, **kwargs) -> None:
    if fabric is None:
        print(*args, **kwargs)
    else:
        print_rank_zero(fabric, *args, **kwargs)

def get_runtime_device(cfg: dict, load_mode: str) -> str:
    if load_mode == "fabric":
        return "cuda"
    return cfg["eval"].get("device", "cuda" if torch.cuda.is_available() else "cpu")


def build_model_cfg(cfg: dict, shared_tokenizer_cfg: dict[str, Any], load_mode: str) -> dict[str, Any]:
    model_cfg = dict(cfg["model"])

    if load_mode == "direct":
        # exported HF directory
        model_cfg["pretrained_name_or_path"] = cfg["eval"]["checkpoint_path"]
    else:
        # original base model; weights loaded later through fabric.load()
        model_cfg["pretrained_name_or_path"] = shared_tokenizer_cfg["base_model"]

    model_cfg["text_vocab_size"] = int(shared_tokenizer_cfg["text_vocab_size"])
    model_cfg["num_codebooks"] = int(shared_tokenizer_cfg["num_codebooks"])
    model_cfg["speech_codebook_size"] = int(shared_tokenizer_cfg["speech_codebook_size"])
    model_cfg["full_vocab_size"] = int(shared_tokenizer_cfg["full_vocab_size"])
    return model_cfg

def setup_model(
    cfg: dict,
    model_cfg: dict[str, Any],
    load_mode: str,
    fabric: Fabric | None,
) -> tuple[Any, str]:
    if load_mode == "fabric":
        if fabric is None:
            raise ValueError("fabric must not be None in fabric load mode")

        log_line(fabric, "Building model...", flush=True)
        model = build_qwen_causal_lm(model_cfg)

        log_line(fabric, "Wrapping model with Fabric...", flush=True)
        model = fabric.setup(model)

        ckpt_path = cfg["eval"]["checkpoint_path"]
        state = {"model": model}
        fabric.load(ckpt_path, state)
        log_line(fabric, f"Loaded checkpoint from {ckpt_path}", flush=True)

        model.eval()
        return model, str(fabric.device)

    # direct mode: load straight from exported HF directory
    device = get_runtime_device(cfg, load_mode)
    log_line(None, "Building model directly from exported HF directory...", flush=True)
    model = build_qwen_causal_lm(model_cfg)
    model.to(device)
    model.eval()
    log_line(None, f"Loaded model directly from {cfg['eval']['checkpoint_path']} onto {device}", flush=True)
    return model, device

def build_eval_dataloader(cfg: dict, pad_token_id: int):
    eval_cfg = cfg["eval"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    max_length = int(train_cfg["max_length"])

    test_cfg_list = data_cfg.get("test_sets", None)
    if test_cfg_list is None or len(test_cfg_list) == 0:
        raise ValueError("Expected data.test_sets in eval config")

    test_specs = parse_eval_specs(test_cfg_list)
    test_dataset = EvalDataset(specs=test_specs, max_length=max_length)

    collator = PackedCollator(
        pad_token_id=pad_token_id,
        pad_to_multiple_of=cfg["model"].get("pad_to_multiple_of"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(eval_cfg.get("per_device_eval_batch_size", 1)),
        num_workers=int(eval_cfg.get("num_workers", 1)),
        pin_memory=bool(eval_cfg.get("pin_memory", True)),
        prefetch_factor=int(eval_cfg.get("prefetch_factor", 2)),
        collate_fn=collator,
    )

    return test_loader


def load_mimi(device: str, num_codebooks: int, mimi_ckpt: str):
    mimi = loaders.get_mimi(mimi_ckpt, device=device)
    mimi.set_num_codebooks(num_codebooks)
    mimi.eval()
    return mimi


@torch.no_grad()
def decode_codes_to_wav(
    mimi,
    codes: np.ndarray,
    device: str,
) -> tuple[np.ndarray, int]:
    code_tensor = torch.from_numpy(codes).long().unsqueeze(0).to(device)
    wav = mimi.decode(code_tensor)
    wav = wav.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return wav, 24000


def extract_prompt_and_target(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[list[int], list[int], list[int]]:
    valid_len = int(attention_mask.long().sum().item())

    input_ids = input_ids[:valid_len].tolist()
    labels = labels[:valid_len].tolist()

    prompt_ids = [inp for inp, lab in zip(input_ids, labels) if lab == IGNORE_INDEX]
    target_ids = [lab for lab in labels if lab != IGNORE_INDEX]

    return input_ids, prompt_ids, target_ids


def split_text_and_speech_ids(mm_tokenizer: MultimodalTokenizer, ids: list[int]) -> tuple[list[int], list[int]]:
    text_ids = [x for x in ids if not mm_tokenizer.is_speech_token(x)]
    speech_ids = [x for x in ids if mm_tokenizer.is_speech_token(x)]
    return text_ids, speech_ids


def maybe_strip_after_eos(ids: list[int], eos_token_id: int) -> list[int]:
    if eos_token_id in ids:
        idx = ids.index(eos_token_id)
        return ids[:idx + 1]
    return ids


def save_speech_segments(
    ids: list[int],
    base_path_stem: Path,
    mm_tokenizer: MultimodalTokenizer,
    mimi,
    device: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Saves one wav per contiguous speech segment.

    Returns:
        wav_paths: list[str]
        speech_debug: list[dict] with info about each speech segment
    """
    wav_paths: list[str] = []
    speech_debug: list[dict[str, Any]] = []

    segments = mm_tokenizer.split_modalities(ids)

    speech_seg_idx = 0
    for seg in segments:
        if seg["type"] != "speech":
            continue

        seg_ids = list(seg["ids"])
        raw_len = len(seg_ids)

        if raw_len == 0:
            continue

        usable_len = (raw_len // mm_tokenizer.num_codebooks) * mm_tokenizer.num_codebooks
        trimmed = raw_len != usable_len

        debug_info = {
            "segment_index": speech_seg_idx,
            "raw_num_tokens": raw_len,
            "usable_num_tokens": usable_len,
            "num_codebooks": mm_tokenizer.num_codebooks,
            "trimmed_incomplete_tail": trimmed,
            "saved_wav": False,
            "wav_path": None,
        }

        if usable_len == 0:
            speech_debug.append(debug_info)
            speech_seg_idx += 1
            continue

        seg_ids = seg_ids[:usable_len]

        try:
            codes = mm_tokenizer.speech_ids_to_codes(seg_ids)
            wav, sr = decode_codes_to_wav(mimi, codes, device=device)

            wav_path = base_path_stem.parent / f"{base_path_stem.name}_speechseg{speech_seg_idx}.wav"
            sf.write(wav_path, wav, sr)

            wav_paths.append(str(wav_path))
            debug_info["saved_wav"] = True
            debug_info["wav_path"] = str(wav_path)

        except Exception as e:
            debug_info["decode_error"] = str(e)

        speech_debug.append(debug_info)
        speech_seg_idx += 1

    return wav_paths, speech_debug


@torch.no_grad()
def generate_one(
    model,
    mm_tokenizer: MultimodalTokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
) -> list[int]:
    gen_model = model.module if hasattr(model, "module") else model

    input_ids = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=next(gen_model.parameters()).device,
    ).unsqueeze(0)

    attention_mask = torch.ones_like(input_ids)

    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        eos_token_id=mm_tokenizer.eos_token_id,
        pad_token_id=mm_tokenizer.pad_token_id,
    )

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
        generate_kwargs["top_k"] = top_k

    generated = gen_model.generate(**generate_kwargs)

    pred_ids = generated[0, input_ids.shape[1]:].tolist()
    pred_ids = maybe_strip_after_eos(pred_ids, mm_tokenizer.eos_token_id)
    return pred_ids


def build_dataset_summary(dataset_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "num_samples": len(records),
        "tasks": sorted({str(r.get("task", "unknown")) for r in records}),
    }

    text_records = [r for r in records if not r["target_has_speech"]]
    speech_target_records = [r for r in records if r["target_has_speech"]]

    summary["num_text_target_samples"] = len(text_records)
    summary["num_speech_target_samples"] = len(speech_target_records)

    if len(text_records) > 0:
        metric_groups: dict[str, list[float]] = defaultdict(list)

        for r in text_records:
            metric_name = r.get("metric_name")
            metric_value = r.get("metric_value")
            if metric_name is not None and metric_value is not None:
                metric_groups[metric_name].append(float(metric_value))

        for metric_name, values in metric_groups.items():
            if len(values) == 0:
                continue
            summary[f"{metric_name}_mean"] = float(sum(values) / len(values))
            summary[f"{metric_name}_num_samples"] = len(values)

        exact_values = [
            float(r["text_exact_match"])
            for r in text_records
            if r.get("text_exact_match") is not None
        ]
        if len(exact_values) > 0:
            summary["text_exact_match_mean"] = float(sum(exact_values) / len(exact_values))
            summary["text_exact_match_num_samples"] = len(exact_values)

        nonempty_pred_text = sum(
            1 for r in text_records if len(safe_normalize_text(r.get("pred_text", ""))) > 0
        )
        summary["nonempty_pred_text_count"] = nonempty_pred_text
        summary["nonempty_pred_text_rate"] = float(nonempty_pred_text) / float(len(text_records))

    if len(speech_target_records) > 0:
        pred_has_speech_count = sum(1 for r in speech_target_records if r["pred_has_speech"])
        ref_has_speech_count = sum(1 for r in speech_target_records if len(r["ref_speech_paths"]) > 0)
        pred_wav_count = sum(1 for r in speech_target_records if len(r["pred_speech_paths"]) > 0)

        summary["speech_pred_present_count"] = pred_has_speech_count
        summary["speech_pred_present_rate"] = float(pred_has_speech_count) / float(len(speech_target_records))
        summary["speech_ref_wav_saved_count"] = ref_has_speech_count
        summary["speech_pred_wav_saved_count"] = pred_wav_count

        nonempty_pred_text_on_speech_targets = sum(
            1 for r in speech_target_records if len(safe_normalize_text(r.get("pred_text", ""))) > 0
        )
        summary["nonempty_pred_text_on_speech_targets_count"] = nonempty_pred_text_on_speech_targets

    return summary


def main(config_path: str, num_nodes: int, num_gpus_per_node: int) -> None:
    cfg = load_yaml(config_path)

    seed = int(cfg.get("seed", 42))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_cudnn_sdp(False)

    load_mode = cfg["eval"].get("load_mode", "fabric").lower()
    if load_mode not in {"fabric", "direct"}:
        raise ValueError("eval.load_mode must be either 'fabric' or 'direct'")

    fabric = None
    if load_mode == "fabric":
        fabric = build_fabric(cfg, num_nodes, num_gpus_per_node)
        fabric.launch()
        fabric.seed_everything(seed)

        log_line(
            fabric,
            f"torch={torch.__version__} "
            f"cuda={torch.version.cuda} "
            f"cudnn={torch.backends.cudnn.version()} "
            f"flash_sdp={torch.backends.cuda.flash_sdp_enabled()} "
            f"mem_efficient_sdp={torch.backends.cuda.mem_efficient_sdp_enabled()} "
            f"math_sdp={torch.backends.cuda.math_sdp_enabled()} "
            f"cudnn_sdp={torch.backends.cuda.cudnn_sdp_enabled()}",
            flush=True,
        )
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        log_line(
            None,
            f"torch={torch.__version__} "
            f"cuda={torch.version.cuda} "
            f"cudnn={torch.backends.cudnn.version()}",
            flush=True,
        )

    dataset_roots = collect_dataset_roots_for_tokenizer_check(cfg)
    shared_tokenizer_cfg = assert_same_tokenizer_config(dataset_roots)

    mm_tokenizer = MultimodalTokenizer.from_config(
        shared_tokenizer_cfg,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )

    model_cfg = build_model_cfg(cfg, shared_tokenizer_cfg, load_mode=load_mode)
    model, runtime_device = setup_model(cfg, model_cfg, load_mode=load_mode, fabric=fabric)

    test_loader = build_eval_dataloader(cfg, pad_token_id=mm_tokenizer.pad_token_id)
    if fabric is not None:
        test_loader = fabric.setup_dataloaders(test_loader)

    out_root = ensure_dir(Path(cfg["eval"]["output_dir"]) / cfg["run_name"])
    rank_suffix = f"rank_{fabric.global_rank}" if fabric is not None else "rank_0"
    rank_out_dir = ensure_dir(out_root / rank_suffix)

    save_speech = bool(cfg["eval"].get("save_speech_outputs", True))
    mimi = None
    if save_speech:
        mimi_ckpt = cfg["eval"].get("mimi_ckpt")
        if not mimi_ckpt:
            raise ValueError("eval.mimi_ckpt must be set when save_speech_outputs=True")
        mimi = load_mimi(
                device=runtime_device,
                num_codebooks=int(shared_tokenizer_cfg["num_codebooks"]),
                mimi_ckpt=mimi_ckpt,
            )

    manifest_path = rank_out_dir / "predictions.jsonl"
    max_samples_total = cfg["eval"].get("max_samples_total", None)
    max_new_tokens_cfg = cfg["eval"].get("max_new_tokens", None)
    generation_buffer = int(cfg["eval"].get("generation_buffer", 16))
    print_examples = int(cfg["eval"].get("print_examples", 20))

    do_sample = bool(cfg["eval"].get("do_sample", True))
    temperature = float(cfg["eval"].get("temperature", 1.0))
    top_p = float(cfg["eval"].get("top_p", 1.0))
    top_k = int(cfg["eval"].get("top_k", 50))

    num_seen = 0
    per_dataset_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for batch_idx, batch in enumerate(test_loader):
            batch_size = batch["input_ids"].shape[0]

            for i in range(batch_size):
                if max_samples_total is not None and num_seen >= int(max_samples_total):
                    break

                meta = batch["meta"][i]
                dataset_name = meta.get("eval_dataset", "unknown")
                task_name = meta.get("task", "unknown")

                _, prompt_ids, target_ids = extract_prompt_and_target(
                    input_ids=batch["input_ids"][i],
                    labels=batch["labels"][i],
                    attention_mask=batch["attention_mask"][i],
                )

                if len(prompt_ids) == 0:
                    log_line(fabric, f"Skipping sample with empty prompt in {dataset_name}", flush=True)
                    continue

                if max_new_tokens_cfg is None:
                    max_new_tokens = max(8, len(target_ids) + generation_buffer)
                else:
                    max_new_tokens = int(max_new_tokens_cfg)

                pred_ids = generate_one(
                    model=model,
                    mm_tokenizer=mm_tokenizer,
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

                prompt_text_ids, prompt_speech_ids = split_text_and_speech_ids(mm_tokenizer, prompt_ids)
                target_text_ids, target_speech_ids = split_text_and_speech_ids(mm_tokenizer, target_ids)
                pred_text_ids, pred_speech_ids = split_text_and_speech_ids(mm_tokenizer, pred_ids)

                prompt_text = mm_tokenizer.decode_text_ids(prompt_text_ids)
                target_text = mm_tokenizer.decode_text_ids(target_text_ids)
                pred_text = mm_tokenizer.decode_text_ids(pred_text_ids)

                sample_dir = ensure_dir(rank_out_dir / dataset_name)
                base_stem = sample_dir / f"sample_{num_seen:06d}"

                pred_speech_paths: list[str] = []
                ref_speech_paths: list[str] = []
                pred_speech_debug: list[dict[str, Any]] = []
                ref_speech_debug: list[dict[str, Any]] = []

                if save_speech and mimi is not None:
                    if len(pred_speech_ids) > 0:
                        pred_speech_paths, pred_speech_debug = save_speech_segments(
                            ids=pred_ids,
                            base_path_stem=base_stem.with_name(base_stem.name + "_pred"),
                            mm_tokenizer=mm_tokenizer,
                            mimi=mimi,
                            device=runtime_device,
                        )
                    if len(target_speech_ids) > 0:
                        ref_speech_paths, ref_speech_debug = save_speech_segments(
                            ids=target_ids,
                            base_path_stem=base_stem.with_name(base_stem.name + "_ref"),
                            mm_tokenizer=mm_tokenizer,
                            mimi=mimi,
                            device=runtime_device,
                        )

                has_target_speech = len(target_speech_ids) > 0
                has_pred_speech = len(pred_speech_ids) > 0

                metric_name = None
                metric_value = None
                text_exact_match = None

                if not has_target_speech:
                    metric_name = infer_text_metric(task_name=task_name, dataset_name=dataset_name)
                    metric_value = compute_text_metric(metric_name, pred_text, target_text)
                    text_exact_match = float(
                        safe_normalize_text(pred_text) == safe_normalize_text(target_text)
                    )

                record = {
                    "sample_index": num_seen,
                    "dataset": dataset_name,
                    "task": task_name,
                    "meta": meta,
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "pred_text": pred_text,
                    "prompt_num_tokens": len(prompt_ids),
                    "target_num_tokens": len(target_ids),
                    "pred_num_tokens": len(pred_ids),
                    "target_has_speech": has_target_speech,
                    "pred_has_speech": has_pred_speech,
                    "pred_speech_paths": pred_speech_paths,
                    "ref_speech_paths": ref_speech_paths,
                    "pred_speech_debug": pred_speech_debug,
                    "ref_speech_debug": ref_speech_debug,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "text_exact_match": text_exact_match,
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                per_dataset_records[dataset_name].append(record)

                is_rank_zero = (fabric is None) or fabric.is_global_zero
                if num_seen < print_examples and is_rank_zero:
                    print("=" * 100)
                    print(f"[sample {num_seen}] dataset={dataset_name} task={task_name}")
                    print("- prompt_text -")
                    print(prompt_text[:3000])
                    print("- target_text -")
                    print(target_text[:2000] if target_text else "[no text target]")
                    print("- pred_text -")
                    print(pred_text[:2000] if pred_text else "[no text prediction]")
                    if len(ref_speech_paths) > 0:
                        print(f"- ref_speech_paths - {ref_speech_paths}")
                    if len(pred_speech_paths) > 0:
                        print(f"- pred_speech_paths - {pred_speech_paths}")
                    print(f"- metric_name - {metric_name}")
                    print(f"- metric_value - {metric_value}")
                    print(f"- text_exact_match - {text_exact_match}")

                num_seen += 1

            if max_samples_total is not None and num_seen >= int(max_samples_total):
                break

    dataset_summaries: dict[str, Any] = {}
    for dataset_name, records in per_dataset_records.items():
        ds_summary = build_dataset_summary(dataset_name, records)
        dataset_summaries[dataset_name] = ds_summary

        with open(rank_out_dir / f"summary_{dataset_name}.json", "w", encoding="utf-8") as f:
            json.dump(ds_summary, f, indent=2, ensure_ascii=False)

    all_records = [r for records in per_dataset_records.values() for r in records]
    global_summary = build_dataset_summary("ALL_DATASETS", all_records)
    global_summary["manifest_path"] = str(manifest_path)
    global_summary["num_datasets"] = len(dataset_summaries)
    global_summary["dataset_names"] = sorted(dataset_summaries.keys())

    with open(rank_out_dir / "dataset_summaries.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summaries, f, indent=2, ensure_ascii=False)

    with open(rank_out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    global_summary["load_mode"] = load_mode
    log_line(fabric, json.dumps(global_summary, indent=2, ensure_ascii=False), flush=True)
    log_line(fabric, json.dumps(dataset_summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=1)
    args = parser.parse_args()
    main(args.config, args.num_nodes, args.num_gpus_per_node)