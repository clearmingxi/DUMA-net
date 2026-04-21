import csv
import json
import os
import re
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


SUBJECTS_DEFAULT = [f"sub-{i:02d}" for i in range(1, 11)]
SUMMARY_METRICS = [
    "test_accuracy",
    "top5_acc",
    "v2_acc",
    "v4_acc",
    "v10_acc",
    "v50_acc",
    "v100_acc",
    "v50_top5_acc",
    "v100_top5_acc",
    "best_epoch",
]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    inherits = config.pop("inherits", None)
    if inherits is None:
        return config

    parent_path = Path(inherits)
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()

    parent_config = load_config(str(parent_path))
    return deep_merge(parent_config, config)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def parse_override(text: str) -> tuple[list[str], Any]:
    if "=" not in text:
        raise ValueError(f"Override must be key=value, got: {text}")
    key, raw_value = text.split("=", 1)
    value = yaml.safe_load(raw_value)
    return key.split("."), value


def set_nested(config: dict[str, Any], path_parts: list[str], value: Any) -> None:
    cursor = config
    for part in path_parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[path_parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = deepcopy(config)
    for item in overrides:
        path_parts, value = parse_override(item)
        set_nested(merged, path_parts, value)
    return merged


def sanitize_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return cleaned or "default"


def get_clip_variant(config: dict[str, Any]) -> str:
    return str(config.get("clip", {}).get("variant", "ViT-H-14"))


def get_target_space(config: dict[str, Any]) -> str:
    ubp_cfg = config.get("ubp", {})
    target_space = ubp_cfg.get("target_space", ubp_cfg.get("mode", "adaptive"))
    normalized = str(target_space).replace("-", "_")
    if normalized not in {"orig_only", "curriculum", "adaptive", "hybrid"}:
        raise ValueError(f"Unsupported ubp.target_space={target_space!r}")
    return normalized


def get_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("experiment", {}).get("mode", "within_subject"))
    if mode not in {"within_subject", "cross_subject"}:
        raise ValueError(f"Unsupported experiment.mode={mode!r}")
    return mode


def get_avg_flag(config: dict[str, Any]) -> bool:
    return str2bool(config.get("data", {}).get("avg", True))


def get_subject_conditioning_setting(config: dict[str, Any]) -> str:
    setting = config.get("model", {}).get("subject_conditioning", "auto")
    if setting in {None, "auto"}:
        return "auto"
    return "on" if str2bool(setting) else "off"


def resolve_subject_conditioning(config: dict[str, Any]) -> bool:
    setting = get_subject_conditioning_setting(config)
    if setting == "auto":
        return get_mode(config) == "within_subject"
    return setting == "on"


def get_insubject_flag(config: dict[str, Any]) -> bool:
    return get_mode(config) == "within_subject"


def get_subjects(config: dict[str, Any]) -> list[str]:
    subjects = config.get("data", {}).get("subjects", SUBJECTS_DEFAULT)
    return list(subjects)


def build_run_id(config: dict[str, Any], timestamp: str | None = None) -> str:
    experiment = config.get("experiment", {})
    train = config.get("train", {})
    model = config.get("model", {})
    ubp = config.get("ubp", {})
    data = config.get("data", {})
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

    name = sanitize_slug(experiment.get("name", "experiment"))
    clip_variant = sanitize_slug(get_clip_variant(config))
    encoder = sanitize_slug(model.get("encoder_type", "NCT_C"))
    ubp_mode = sanitize_slug(get_target_space(config))
    avg_tag = "avg" if get_avg_flag(config) else "single"
    seed_tag = f"seed{train.get('seed', 3047)}"
    return f"{timestamp}_{name}_{clip_variant}_{encoder}_{ubp_mode}_{avg_tag}_{seed_tag}"


def build_run_dir(config: dict[str, Any], run_id: str | None = None) -> Path:
    experiment = config.get("experiment", {})
    model = config.get("model", {})
    ubp = config.get("ubp", {})
    train = config.get("train", {})
    output_root = Path(experiment.get("output_root", "outputs"))
    mode = get_mode(config)
    run_id = run_id or build_run_id(config)
    mode_tag = "within" if mode == "within_subject" else "cross"

    parts = [
        output_root,
        mode_tag,
        f"backbone_{sanitize_slug(get_clip_variant(config))}",
        f"encoder_{sanitize_slug(model.get('encoder_type', 'NCT_C'))}",
        f"target_{sanitize_slug(get_target_space(config))}",
        f"insubject_{'on' if get_insubject_flag(config) else 'off'}",
        f"subject_conditioning_{sanitize_slug(get_subject_conditioning_setting(config))}",
        f"avg_{'true' if get_avg_flag(config) else 'false'}",
        f"seed_{train.get('seed', 3047)}",
        run_id,
    ]
    return Path(*parts)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_summary_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    metrics: dict[str, Any] = {"num_subjects": len(rows)}
    for key in SUMMARY_METRICS:
        values: list[float] = []
        for row in rows:
            raw = row.get(key, "")
            if raw == "":
                continue
            values.append(float(raw))
        if values:
            metrics[f"mean_{key}"] = sum(values) / len(values)
    return metrics


def find_single_summary_csv(raw_output_dir: Path) -> Path | None:
    matches = sorted(raw_output_dir.rglob("best_summary_*.csv"))
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime)


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True
