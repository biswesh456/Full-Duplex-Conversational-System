from pathlib import Path

from lightning.fabric import Fabric


def save_checkpoint(
    fabric: Fabric,
    ckpt_dir: str | Path,
    state: dict,
    step: int,
    tag: str | None = None,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    name = f"step-{step:08d}"
    if tag is not None:
        name = f"{name}-{tag}"

    path = ckpt_dir / name
    fabric.save(str(path), state)

    latest_path = ckpt_dir / "latest"
    fabric.save(str(latest_path), state)