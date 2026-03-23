import io
import json
from pathlib import Path

import numpy as np
import webdataset as wds

from training.step1.preprocessing.schema import PackedSample


class PackedShardWriter:
    def __init__(self, out_dir: str | Path, prefix: str, maxcount: int) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(self.out_dir / f"{prefix}-%06d.tar")
        self.sink = wds.ShardWriter(pattern, maxcount=maxcount)

    def write(self, sample: PackedSample) -> None:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            input_ids=sample.input_ids,
            labels=sample.labels,
            attention_mask=sample.attention_mask,
        )
        buf.seek(0)

        self.sink.write(
            {
                "__key__": sample.sample_id,
                "json": json.dumps(sample.meta, ensure_ascii=False).encode("utf-8"),
                "npz": buf.getvalue(),
            }
        )

    def close(self) -> None:
        self.sink.close()