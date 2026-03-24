from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
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
    ) -> None:

        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.speech_codebook_size = speech_codebook_size

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
        )

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