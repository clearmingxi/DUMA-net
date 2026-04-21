import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_utils import compute_summary_metrics

P2_ENCODER = "NCT_S"
P2_SUBJECT_CONDITIONING = False
P2_AVG = True
DEFAULT_METRIC = "mean_test_accuracy"
TARGET_ORDER = ["orig_only", "curriculum", "adaptive", "hybrid"]
MODE_ORDER = ["within", "cross"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P2 summary tables from normalized run outputs.")
    parser.add_argument("--input-root", default="outputs", help="Root directory to scan for run folders.")
    parser.add_argument("--output-dir", default="outputs/collected/p2", help="Directory for P2 tables.")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Primary ranking metric.")
    return parser.parse_args()


def load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    summary_path = run_dir / "summary.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {
            "status": "summary_only",
            "run_dir": str(run_dir),
            "output_dir": str(run_dir),
            "summary_csv": str(summary_path),
            "aggregates": compute_summary_metrics(rows),
        }
    return {}


def normalize_target(value: Any) -> str:
    text = str(value or "").strip().replace("-", "_")
    return text


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    aggregates = dict(record.get("aggregates", {}))
    normalized = {
        "run_dir": record.get("run_dir"),
        "output_dir": record.get("output_dir", record.get("run_dir")),
        "status": record.get("status"),
        "mode": record.get("mode"),
        "backbone": record.get("backbone", record.get("clip_variant")),
        "encoder_type": record.get("encoder_type"),
        "subject_conditioning_enabled": bool(record.get("subject_conditioning_enabled")),
        "subject_conditioning": record.get("subject_conditioning"),
        "avg": bool(record.get("avg")),
        "seed": int(record.get("seed")) if record.get("seed") is not None else None,
        "target_space": normalize_target(record.get("target_space", record.get("ubp_mode"))),
        "summary_csv": record.get("summary_csv"),
        "num_subjects": aggregates.get("num_subjects"),
    }
    normalized.update(aggregates)
    return normalized


def read_all_records(root: Path) -> list[dict[str, Any]]:
    run_dirs = sorted(path.parent for path in root.rglob("config_snapshot.yaml"))
    records = []
    for run_dir in run_dirs:
        loaded = load_metrics(run_dir)
        if loaded:
            records.append(normalize_record(loaded))
    return records


def timestamp_key(run_dir: str | None) -> str:
    if not run_dir:
        return ""
    return Path(run_dir).name.split("_", 1)[0]


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record.get("mode"),
            record.get("backbone"),
            record.get("encoder_type"),
            record.get("subject_conditioning_enabled"),
            record.get("avg"),
            record.get("target_space"),
            record.get("seed"),
        )
        grouped[key].append(record)

    deduped = []
    for group in grouped.values():
        group.sort(
            key=lambda item: (
                1 if item.get("status") == "completed" else 0,
                timestamp_key(item.get("run_dir")),
            )
        )
        deduped.append(group[-1])
    return deduped


def filter_p2_main(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        if record.get("status") != "completed":
            continue
        if record.get("encoder_type") != P2_ENCODER:
            continue
        if record.get("subject_conditioning_enabled") != P2_SUBJECT_CONDITIONING:
            continue
        if record.get("avg") != P2_AVG:
            continue
        if record.get("mode") not in MODE_ORDER:
            continue
        filtered.append(record)
    return filtered


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def aggregate_group(records: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {"num_runs": len(records)}
    seeds = sorted({record["seed"] for record in records if record.get("seed") is not None})
    row["seeds"] = " ".join(str(seed) for seed in seeds)
    row["num_seeds"] = len(seeds)
    for metric in metric_names:
        values = [safe_float(record.get(metric)) for record in records]
        values = [value for value in values if value is not None]
        if values:
            row[metric] = mean(values)
            row[f"std_{metric}"] = std(values)
    num_subjects = [safe_float(record.get("num_subjects")) for record in records]
    num_subjects = [value for value in num_subjects if value is not None]
    if num_subjects:
        row["num_subjects"] = mean(num_subjects)
    latest = max(records, key=lambda item: timestamp_key(item.get("run_dir")))
    row["latest_run_dir"] = latest.get("run_dir")
    row["latest_summary_csv"] = latest.get("summary_csv")
    return row


def build_backbone_table(records: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    stage_a = [record for record in records if record.get("target_space") == "curriculum"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in stage_a:
        grouped[(record["mode"], record["backbone"])].append(record)

    rows = []
    metric_names = [metric, "mean_top5_acc", "mean_v50_acc", "mean_v100_acc"]
    for mode in MODE_ORDER:
        backbones = sorted({key[1] for key in grouped if key[0] == mode})
        for backbone in backbones:
            agg = aggregate_group(grouped[(mode, backbone)], metric_names)
            agg.update(
                {
                    "mode": mode,
                    "backbone": backbone,
                    "target_space": "curriculum",
                    "encoder_type": P2_ENCODER,
                    "subject_conditioning": "off",
                    "avg": True,
                }
            )
            rows.append(agg)
    return rows


def pick_best_backbones(backbone_rows: list[dict[str, Any]], metric: str) -> dict[str, str]:
    best: dict[str, str] = {}
    for mode in MODE_ORDER:
        mode_rows = [row for row in backbone_rows if row.get("mode") == mode and row.get(metric) is not None]
        if not mode_rows:
            continue
        mode_rows.sort(
            key=lambda row: (
                row.get(metric, float("-inf")),
                row.get("mean_top5_acc", float("-inf")),
                row.get("backbone", ""),
            )
        )
        best[mode] = mode_rows[-1]["backbone"]
    return best


def build_targetspace_table(records: list[dict[str, Any]], best_backbones: dict[str, str], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        mode = record["mode"]
        backbone = record["backbone"]
        target = record["target_space"]
        if best_backbones.get(mode) != backbone:
            continue
        grouped[(mode, backbone, target)].append(record)

    rows = []
    metric_names = [metric, "mean_top5_acc", "mean_v50_acc", "mean_v100_acc"]
    for mode in MODE_ORDER:
        backbone = best_backbones.get(mode)
        if not backbone:
            continue
        for target in TARGET_ORDER:
            bucket = grouped.get((mode, backbone, target), [])
            if not bucket:
                continue
            agg = aggregate_group(bucket, metric_names)
            agg.update(
                {
                    "mode": mode,
                    "backbone": backbone,
                    "target_space": target,
                    "encoder_type": P2_ENCODER,
                    "subject_conditioning": "off",
                    "avg": True,
                }
            )
            rows.append(agg)
    return rows


def pick_best_settings(records: list[dict[str, Any]], metric: str) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["mode"], record["backbone"], record["target_space"])].append(record)

    best: dict[str, dict[str, Any]] = {}
    metric_names = [metric, "mean_top5_acc", "mean_v50_acc", "mean_v100_acc"]
    for mode in MODE_ORDER:
        candidates = []
        for (group_mode, backbone, target), bucket in grouped.items():
            if group_mode != mode:
                continue
            agg = aggregate_group(bucket, metric_names)
            agg.update(
                {
                    "mode": mode,
                    "backbone": backbone,
                    "target_space": target,
                    "encoder_type": P2_ENCODER,
                    "subject_conditioning": "off",
                    "avg": True,
                }
            )
            candidates.append(agg)
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                row.get(metric, float("-inf")),
                row.get("mean_top5_acc", float("-inf")),
                row.get("backbone", ""),
                row.get("target_space", ""),
            )
        )
        best[mode] = candidates[-1]
    return best


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["empty"])
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    records = dedupe_records(read_all_records(root))
    p2_records = filter_p2_main(records)

    backbone_rows = build_backbone_table(p2_records, args.metric)
    best_backbones = pick_best_backbones(backbone_rows, args.metric)
    target_rows = build_targetspace_table(p2_records, best_backbones, args.metric)
    best_settings = pick_best_settings(p2_records, args.metric)
    best_setting_rows = [best_settings[mode] for mode in MODE_ORDER if mode in best_settings]

    write_csv(output_dir / "backbone_main_table.csv", backbone_rows)
    write_csv(output_dir / "targetspace_main_table.csv", target_rows)
    write_csv(output_dir / "best_setting_summary.csv", best_setting_rows)

    selection_payload = {
        "metric": args.metric,
        "best_backbones": best_backbones,
        "best_settings": best_settings,
    }
    with (output_dir / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection_payload, handle, indent=2, ensure_ascii=True)

    summary = {
        "num_all_records": len(records),
        "num_p2_records": len(p2_records),
        "best_backbones": best_backbones,
        "best_settings": {
            mode: {
                "backbone": row.get("backbone"),
                "target_space": row.get("target_space"),
                "num_seeds": row.get("num_seeds"),
                args.metric: row.get(args.metric),
                f"std_{args.metric}": row.get(f"std_{args.metric}"),
            }
            for mode, row in best_settings.items()
        },
        "outputs": {
            "backbone_main_table": str(output_dir / "backbone_main_table.csv"),
            "targetspace_main_table": str(output_dir / "targetspace_main_table.csv"),
            "best_setting_summary": str(output_dir / "best_setting_summary.csv"),
            "selection_json": str(output_dir / "selection.json"),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
