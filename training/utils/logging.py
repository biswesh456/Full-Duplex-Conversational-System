from lightning.fabric import Fabric


def print_rank_zero(fabric: Fabric, *args, **kwargs) -> None:
    if fabric.is_global_zero:
        print(*args, **kwargs)