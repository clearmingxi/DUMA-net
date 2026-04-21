import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from experiment_utils import (
    SUBJECTS_DEFAULT,
    apply_overrides,
    build_run_dir,
    build_run_id,
    compute_summary_metrics,
    copy_if_exists,
    dump_json,
    dump_yaml,
    find_single_summary_csv,
    get_avg_flag,
    get_clip_variant,
    get_insubject_flag,
    get_mode,
    get_subjects,
    get_subject_conditioning_setting,
    get_target_space,
    load_config,
    resolve_subject_conditioning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified experiment runner that wraps EEG_retrieval.py without changing its core logic."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dotted key=value syntax. Example: --set train.seed=1234",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command, environment, and output directory without running anything.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config after applying inheritance and overrides.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch EEG_retrieval.py.",
    )
    return parser.parse_args()


def resolve_clip_env(config: dict[str, Any]) -> dict[str, str]:
    data_config_path = Path(config.get("clip", {}).get("registry_config", "data_config.json")).resolve()
    with data_config_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle).get("clip_models", {})

    variant = get_clip_variant(config)
    if variant not in registry:
        raise ValueError(f"Unknown clip.variant={variant!r}. Available: {sorted(registry)}")

    model_cfg = registry[variant]
    env = {
        "CLIP_VARIANT": variant,
        "CLIP_MODEL_NAME": model_cfg["model_name"],
        "CLIP_PRETRAINED_PATH": model_cfg["pretrained_path"],
    }
    return env


def _extend_bool_flag(cmd: list[str], name: str, enabled: bool) -> None:
    cmd.extend([f"--{name}", "true" if enabled else "false"])


def _extend_optional_list(cmd: list[str], name: str, values: list[Any] | None) -> None:
    if not values:
        return
    cmd.append(f"--{name}")
    cmd.extend(str(v) for v in values)


def build_legacy_command(config: dict[str, Any], raw_output_dir: Path) -> list[str]:
    experiment = config.get("experiment", {})
    model = config.get("model", {})
    train = config.get("train", {})
    data = config.get("data", {})
    ubp = config.get("ubp", {})

    cmd = [
        "EEG_retrieval.py",
        "--data_path",
        str(data.get("data_path")),
        "--output_dir",
        str(raw_output_dir),
        "--name",
        str(experiment.get("name", "experiment")),
        "--lr",
        str(train.get("lr", 2e-4)),
        "--epochs",
        str(train.get("epochs", 100)),
        "--batch_size",
        str(train.get("batch_size", 1024)),
        "--gpu",
        str(train.get("gpu", "cuda:0")),
        "--device",
        str(train.get("device", "gpu")),
        "--alpha",
        str(train.get("alpha", 0.99)),
        "--encoder_type",
        str(model.get("encoder_type", "NCT_C")),
        "--seed",
        str(train.get("seed", 3047)),
        "--emb_size",
        str(train.get("emb_size", 256)),
        "--num_heads",
        str(train.get("num_heads", 4)),
        "--dropout",
        str(train.get("dropout", 0.7)),
        "--warmup_epochs",
        str(train.get("warmup_epochs", 5)),
        "--weight_decay",
        str(train.get("weight_decay", 0.05)),
        "--patience",
        str(train.get("patience", 15)),
        "--ubp_mode",
        str(get_target_space(config)),
        "--lambda_mu",
        str(ubp.get("lambda_mu", 0.5)),
        "--lambda_tau",
        str(ubp.get("lambda_tau", 0.1)),
        "--curriculum_start_level",
        str(ubp.get("curriculum_start_level", 4.0)),
        "--curriculum_end_level",
        str(ubp.get("curriculum_end_level", 0.0)),
        "--curriculum_begin_epoch",
        str(ubp.get("curriculum_begin_epoch", 0)),
        "--curriculum_end_epoch",
        str(ubp.get("curriculum_end_epoch", -1)),
        "--hybrid_switch_epoch",
        str(ubp.get("hybrid_switch_epoch", -1)),
        "--orig_mix_ratio",
        str(ubp.get("orig_mix_ratio", 0.0)),
    ]

    if train.get("deterministic", False):
        cmd.append("--deterministic")
    if get_insubject_flag(config):
        cmd.append("--insubject")
    cmd.extend(["--subject_conditioning", str(resolve_subject_conditioning(config)).lower()])

    _extend_optional_list(cmd, "subjects", get_subjects(config))

    target_subject = data.get("target_subject")
    if target_subject:
        cmd.extend(["--target_subject", str(target_subject)])

    pretrained_path = train.get("pretrained_path")
    if pretrained_path:
        cmd.extend(["--pretrained_path", str(pretrained_path)])

    _extend_bool_flag(cmd, "avg", get_avg_flag(config))

    selected_ch = data.get("selected_ch")
    if selected_ch:
        if isinstance(selected_ch, str):
            selected_ch = [selected_ch]
        _extend_optional_list(cmd, "selected_ch", selected_ch)

    time_window = data.get("time_window", [None, None])
    if not isinstance(time_window, list) or len(time_window) != 2:
        raise ValueError("data.time_window must be a two-item list like [start, end]")
    if time_window[0] is not None:
        cmd.extend(["--time_window_start", str(time_window[0])])
    if time_window[1] is not None:
        cmd.extend(["--time_window_end", str(time_window[1])])

    return cmd


def write_command_snapshot(run_dir: Path, command: list[str], env_overrides: dict[str, str]) -> None:
    payload = {
        "command": " ".join(shlex.quote(part) for part in command),
        "env_overrides": env_overrides,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    dump_json(run_dir / "command_snapshot.json", payload)


def summarize_run(config: dict[str, Any], run_dir: Path, raw_output_dir: Path, command: list[str], env_overrides: dict[str, str]) -> dict[str, Any]:
    summary_src = find_single_summary_csv(raw_output_dir)
    summary_dst = run_dir / "summary.csv"
    summary_found = copy_if_exists(summary_src, summary_dst)

    summary_rows = []
    if summary_found:
        with summary_dst.open("r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))

    metrics = {
        "status": "completed" if summary_found else "missing_summary",
        "run_dir": str(run_dir),
        "output_dir": str(run_dir),
        "raw_output_dir": str(raw_output_dir),
        "mode": "within" if get_mode(config) == "within_subject" else "cross",
        "backbone": get_clip_variant(config),
        "clip_variant": get_clip_variant(config),
        "encoder_type": config.get("model", {}).get("encoder_type", "NCT_C"),
        "target_space": get_target_space(config),
        "ubp_mode": get_target_space(config),
        "insubject": get_insubject_flag(config),
        "subject_conditioning": get_subject_conditioning_setting(config),
        "subject_conditioning_enabled": resolve_subject_conditioning(config),
        "avg": get_avg_flag(config),
        "seed": config.get("train", {}).get("seed", 3047),
        "subjects": get_subjects(config),
        "summary_csv": str(summary_dst) if summary_found else None,
        "command": " ".join(shlex.quote(part) for part in command),
        "env_overrides": env_overrides,
        "aggregates": compute_summary_metrics(summary_rows),
    }
    dump_json(run_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)

    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=True))

    run_id = build_run_id(config)
    run_dir = build_run_dir(config, run_id=run_id)
    raw_output_dir = run_dir / "raw_outputs"
    log_dir = run_dir / "logs"
    log_path = log_dir / "run.log"
    env_overrides = resolve_clip_env(config)
    legacy_command = build_legacy_command(config, raw_output_dir)
    full_command = [args.python, *legacy_command]

    print(f"[run_dir] {run_dir}")
    print(f"[mode] {get_mode(config)}")
    print(f"[target_space] {get_target_space(config)}")
    print(f"[subject_conditioning] setting={get_subject_conditioning_setting(config)} resolved={resolve_subject_conditioning(config)}")
    print(f"[command] {' '.join(shlex.quote(part) for part in full_command)}")
    print(f"[env] {json.dumps(env_overrides, ensure_ascii=True)}")

    if args.dry_run:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = json.loads(json.dumps(config))
    config_snapshot.setdefault("runtime", {})
    config_snapshot["runtime"].update(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "raw_output_dir": str(raw_output_dir),
            "resolved_python": args.python,
        }
    )
    dump_yaml(run_dir / "config_snapshot.yaml", config_snapshot)
    write_command_snapshot(run_dir, full_command, env_overrides)

    env = os.environ.copy()
    env.update(env_overrides)

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"[run_dir] {run_dir}\n")
        log_handle.write(f"[command] {' '.join(shlex.quote(part) for part in full_command)}\n")
        log_handle.write(f"[env] {json.dumps(env_overrides, ensure_ascii=True)}\n")
        log_handle.flush()
        completed = subprocess.run(
            full_command,
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    summarize_run(config, run_dir, raw_output_dir, full_command, env_overrides)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
