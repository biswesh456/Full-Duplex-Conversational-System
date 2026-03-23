from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RawSample:
    sample_id: str
    dataset: str
    split: str
    task: str
    instruction: str
    input_text: Optional[str] = None
    target_text: Optional[str] = None
    input_text_token: Optional[np.ndarray] = None
    target_text_token: Optional[np.ndarray] = None
    input_speech: Optional[np.ndarray] = None
    target_speech: Optional[np.ndarray] = None
    question_text: Optional[str] = None
    meta: Optional[dict] = None


@dataclass
class PackedSample:
    sample_id: str
    dataset: str
    split: str
    task: str
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray
    meta: dict