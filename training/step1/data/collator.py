from typing import Any

import torch


IGNORE_INDEX = -100


class PackedCollator:
    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def _round_up(self, n: int) -> int:
        if self.pad_to_multiple_of is None:
            return n
        m = self.pad_to_multiple_of
        return ((n + m - 1) // m) * m

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(x["input_ids"]) for x in batch)
        max_len = self._round_up(max_len)

        input_ids = []
        labels = []
        attention_mask = []
        meta = []

        for sample in batch:
            seq_len = len(sample["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(
                torch.cat(
                    [
                        sample["input_ids"],
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            labels.append(
                torch.cat(
                    [
                        sample["labels"],
                        torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        sample["attention_mask"],
                        torch.zeros((pad_len,), dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            meta.append(sample["meta"])

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "labels": torch.stack(labels, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "meta": meta,
        }