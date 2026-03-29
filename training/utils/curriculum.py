from training.step1.data.packed_webdataset import DatasetSpec, resolve_webdataset_urls


def get_curriculum_stage(cfg: dict, global_step: int) -> int:
    curriculum_cfg = cfg.get("curriculum", {})
    if not curriculum_cfg.get("enabled", False):
        return 0

    update_every_steps = int(curriculum_cfg["update_every_steps"])
    if update_every_steps <= 0:
        raise ValueError("curriculum.update_every_steps must be > 0")

    return min(global_step // update_every_steps, 3)

def get_stage_progress(stage: int) -> float:
    if stage <= 0:
        return 0.0
    if stage == 1:
        return 1.0 / 3.0
    if stage == 2:
        return 2.0 / 3.0
    return 1.0

def build_train_specs_for_step(cfg: dict, global_step: int) -> list[DatasetSpec]:
    stage = get_curriculum_stage(cfg, global_step)
    t = get_stage_progress(stage)

    specs = []
    total_weight = 0.0

    for ds in cfg["data"]["train_mix"]:
        start_weight = float(ds["start_weight"])
        end_weight = float(ds["end_weight"])
        weight = start_weight + t * (end_weight - start_weight)
        urls = resolve_webdataset_urls(ds["urls"])

        specs.append(
            DatasetSpec(
                name=ds["name"],
                urls=urls,
                weight=weight,
            )
        )
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Curriculum weights sum to zero")

    for spec in specs:
        spec.weight /= total_weight

    return specs