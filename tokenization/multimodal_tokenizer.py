from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from pathlib import Path
import json
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tokenization.special_tokens import TEXT_SPECIAL_TOKENS

@dataclass
class TokenRanges:
    text_vocab_size: int
    speech_offset: int
    speech_vocab_size_total: int
    extra_special_ids: dict[str, int]

    @property
    def full_vocab_size(self) -> int:
        return self.speech_offset + self.speech_vocab_size_total

class MultimodalTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_codebooks: int,
        speech_codebook_size: int,
        base_model: str | None = None,
    ) -> None:

        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.speech_codebook_size = speech_codebook_size
        self.base_model = base_model

        extra = list(TEXT_SPECIAL_TOKENS.values())
        if len(extra) > 0:
            added = tokenizer.add_special_tokens({"additional_special_tokens": extra})

        # Taking the length after adding the special tokens
        self.text_vocab_size = len(tokenizer)
        self.speech_offset = self.text_vocab_size
        self.speech_vocab_size_total = num_codebooks * speech_codebook_size

        self.extra_special_ids = {
            key: tokenizer.convert_tokens_to_ids(tok)
            for key, tok in TEXT_SPECIAL_TOKENS.items()
        }

        self.ranges = TokenRanges(
            text_vocab_size=self.text_vocab_size,
            speech_offset=self.speech_offset,
            speech_vocab_size_total=self.speech_vocab_size_total,
            extra_special_ids=self.extra_special_ids,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        num_codebooks: int,
        speech_codebook_size: int,
        use_fast: bool = True,
        trust_remote_code: bool = False,
    ) -> MultimodalTokenizer:
        
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name_or_path,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )
        return cls(
            tokenizer=tokenizer,
            num_codebooks=num_codebooks,
            speech_codebook_size=speech_codebook_size,
            base_model=pretrained_name_or_path,
        )
    
    @classmethod
    def from_config(
        cls,
        config_or_path: dict[str, Any] | str | Path,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        validate: bool = True,
    ) -> MultimodalTokenizer:
        if isinstance(config_or_path, (str, Path)):
            with open(config_or_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = dict(config_or_path)

        mm_tokenizer = cls.from_pretrained(
            pretrained_name_or_path=cfg["base_model"],
            num_codebooks=int(cfg["num_codebooks"]),
            speech_codebook_size=int(cfg["speech_codebook_size"]),
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        if validate:
            mm_tokenizer.validate_against_config(cfg)

        return mm_tokenizer

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "text_vocab_size": self.ranges.text_vocab_size,
            "speech_offset": self.ranges.speech_offset,
            "speech_vocab_size_total": self.ranges.speech_vocab_size_total,
            "full_vocab_size": self.ranges.full_vocab_size,
            "num_codebooks": self.num_codebooks,
            "speech_codebook_size": self.speech_codebook_size,
            "extra_special_ids": self.ranges.extra_special_ids,
        }

    def validate_against_config(self, cfg: dict[str, Any]) -> None:
        expected = {
            "text_vocab_size": int(cfg["text_vocab_size"]),
            "speech_offset": int(cfg["speech_offset"]),
            "speech_vocab_size_total": int(cfg["speech_vocab_size_total"]),
            "full_vocab_size": int(cfg["full_vocab_size"]),
            "num_codebooks": int(cfg["num_codebooks"]),
            "speech_codebook_size": int(cfg["speech_codebook_size"]),
        }

        actual = {
            "text_vocab_size": int(self.ranges.text_vocab_size),
            "speech_offset": int(self.ranges.speech_offset),
            "speech_vocab_size_total": int(self.ranges.speech_vocab_size_total),
            "full_vocab_size": int(self.ranges.full_vocab_size),
            "num_codebooks": int(self.num_codebooks),
            "speech_codebook_size": int(self.speech_codebook_size),
        }

        for key, expected_value in expected.items():
            actual_value = actual[key]
            if actual_value != expected_value:
                raise ValueError(
                    f"Tokenizer config mismatch for {key}: "
                    f"expected {expected_value}, got {actual_value}"
                )

        expected_special_ids = cfg.get("extra_special_ids", {})

        if set(expected_special_ids.keys()) != set(self.ranges.extra_special_ids.keys()):
            raise ValueError(
                "Tokenizer config mismatch for special token names: "
                f"expected {sorted(expected_special_ids.keys())}, "
                f"got {sorted(self.ranges.extra_special_ids.keys())}"
            )

        for token_name, expected_id in expected_special_ids.items():
            actual_id = self.ranges.extra_special_ids.get(token_name)
            if actual_id != expected_id:
                raise ValueError(
                    f"Tokenizer config mismatch for special token {token_name}: "
                    f"expected {expected_id}, got {actual_id}"
                )
    
    def save_config(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_config_dict(), f, indent=2, ensure_ascii=False)

    def text_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_text_ids(self, ids: Sequence[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def speech_ids(self, codes: np.ndarray | Sequence[Sequence[int]]) -> list[int]:
        arr = np.asarray(codes)
        if arr.ndim != 2:
            raise ValueError(f"speech codes must be [K, T], got shape={arr.shape}")
        k, t = arr.shape
        if k != self.num_codebooks:
            raise ValueError(f"expected {self.num_codebooks} codebooks, got {k}")

        out: list[int] = []
        for frame_idx in range(t):
            for codebook_idx in range(k):
                code = int(arr[codebook_idx, frame_idx])
                if code < 0 or code >= self.speech_codebook_size:
                    raise ValueError(
                        f"speech code out of range at codebook={codebook_idx}, frame={frame_idx}, code={code}"
                    )
                out.append(
                    self.speech_offset + codebook_idx * self.speech_codebook_size + code
                )
        return out

    def speech_ids_to_codes(self, speech_ids: Sequence[int]) -> np.ndarray:
        """
        Inverse of speech_ids().

        Input:
          flattened speech token ids of length T * K

        Output:
          np.ndarray of shape [K, T]
        """
        speech_ids = list(speech_ids)

        if len(speech_ids) % self.num_codebooks != 0:
            raise ValueError(
                f"Speech span length {len(speech_ids)} is not divisible by "
                f"num_codebooks={self.num_codebooks}"
            )

        num_frames = len(speech_ids) // self.num_codebooks
        codes = np.zeros((self.num_codebooks, num_frames), dtype=np.int64)

        idx = 0
        for frame_idx in range(num_frames):
            for codebook_idx in range(self.num_codebooks):
                token_id = int(speech_ids[idx])
                rel = token_id - self.speech_offset

                if rel < 0 or rel >= self.speech_vocab_size_total:
                    raise ValueError(
                        f"Speech token id {token_id} out of range for speech vocabulary"
                    )

                recovered_codebook_idx = rel // self.speech_codebook_size
                code = rel % self.speech_codebook_size

                if recovered_codebook_idx != codebook_idx:
                    raise ValueError(
                        f"Unexpected speech ordering: expected codebook {codebook_idx}, "
                        f"got {recovered_codebook_idx}"
                    )

                codes[codebook_idx, frame_idx] = code
                idx += 1

        return codes

    def is_speech_token(self, token_id: int) -> bool:
        return self.speech_offset <= token_id < (
            self.speech_offset + self.speech_vocab_size_total
        )

    def split_modalities(self, ids: Sequence[int]) -> list[dict[str, Any]]:
        """
        Split a mixed multimodal sequence into contiguous text/speech segments.
        """
        ids = list(ids)
        if not ids:
            return []

        def kind(x: int) -> str:
            return "speech" if self.is_speech_token(x) else "text"

        segments: list[dict[str, Any]] = []
        start = 0
        current_kind = kind(ids[0])

        for i in range(1, len(ids)):
            k = kind(ids[i])
            if k != current_kind:
                segments.append(
                    {
                        "type": current_kind,
                        "start": start,
                        "end": i,
                        "ids": ids[start:i],
                    }
                )
                start = i
                current_kind = k

        segments.append(
            {
                "type": current_kind,
                "start": start,
                "end": len(ids),
                "ids": ids[start:],
            }
        )
        return segments

    def special_id(self, name: str) -> int:
        return self.extra_special_ids[name]

    @property
    def pad_token_id(self) -> int:
        if self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            return int(self.tokenizer.eos_token_id)
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    @property
    def eos_token_id(self) -> int:
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id")
        return int(self.tokenizer.eos_token_id)

    @property
    def bos_token_id(self) -> int:
        if self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer has no bos_token_id")
        return int(self.tokenizer.bos_token_id)

    @property
    def hf_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer