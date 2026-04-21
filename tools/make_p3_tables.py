import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_utils import compute_summary_metrics

P3_SEED = 3047
P3_TARGET = "curriculum"
P3_SUBJECT_CONDITIONING = False
P3_AVG = True
MODE_ORDER = ["within", "cross"]
BACKBONE_ORDER = ["RN50", "ViT-B-32", "ViT-H-14"]
ENCODER_ORDER = ["NCT_C", "NCT_S"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P3 encoder-backbone interaction tables.")
    parser.add_argument("--input-root", default="outputs", help="Root directory to scan for run folders.")
    parser.add_argument("--output-dir", default="outputs/collected/p3", help="Directory for P3 tables.")
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
    return str(value or "").strip().replace("-", "_")


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    aggregates = dict(record.get("aggregates", {}))
    normalized = {
        "run_dir": record.get("run_dir"),
        "output_dir": record.get("output_dir", record.get("run_dir")),
        "status": record.get("status"),
        "mode": record.get("mode"),
        "backbone": record.get("backbone", record.get("clip_variant")),
        "encoder": record.get("encoder_type"),
        "encoder_type": record.get("encoder_type"),
        "target_space": normalize_target(record.get("target_space", record.get("ubp_mode"))),
        "subject_conditioning": record.get("subject_conditioning"),
        "subject_conditioning_enabled": normalize_bool(record.get("subject_conditioning_enabled")),
        "avg": normalize_bool(record.get("avg")),
        "seed": int(record.get("seed")) if record.get("seed") is not None else None,
        "summary_csv": record.get("summary_csv"),
        "num_subjects": aggregates.get("num_subjects"),
        "top1": aggregates.get("mean_test_accuracy"),
        "top5": aggregates.get("mean_top5_acc"),
        "v50": aggregates.get("mean_v50_acc"),
        "v100": aggregates.get("mean_v100_acc"),
        "best_epoch": aggregates.get("mean_best_epoch"),
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
            record.get("encoder"),
            record.get("target_space"),
            record.get("subject_conditioning_enabled"),
            record.get("avg"),
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


def filter_p3(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        run_dir_name = Path(record.get("run_dir") or "").name
        if "p3_encoder_backbone" not in run_dir_name:
            continue
        if record.get("status") != "completed":
            continue
        if record.get("mode") not in MODE_ORDER:
            continue
        if record.get("backbone") not in BACKBONE_ORDER:
            continue
        if record.get("encoder") not in ENCODER_ORDER:
            continue
        if record.get("target_space") != P3_TARGET:
            continue
        if record.get("subject_conditioning_enabled") != P3_SUBJECT_CONDITIONING:
            continue
        if record.get("avg") != P3_AVG:
            continue
        if record.get("seed") != P3_SEED:
            continue
        filtered.append(record)
    return filtered


def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        MODE_ORDER.index(row["mode"]),
        BACKBONE_ORDER.index(row["backbone"]),
        ENCODER_ORDER.index(row["encoder"]),
    )


def main_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in sorted(records, key=sort_key):
        rows.append(
            {
                "mode": record.get("mode"),
                "backbone": record.get("backbone"),
                "encoder": record.get("encoder"),
                "target_space": record.get("target_space"),
                "subject_conditioning": "off",
                "avg": record.get("avg"),
                "seed": record.get("seed"),
                "top1": record.get("top1"),
                "top5": record.get("top5"),
                "v50": record.get("v50"),
                "v100": record.get("v100"),
                "best_epoch": record.get("best_epoch"),
                "num_subjects": record.get("num_subjects"),
                "output_dir": record.get("output_dir"),
                "summary_csv": record.get("summary_csv"),
            }
        )
    return rows


def contrast_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_combo = {
        (record["mode"], record["backbone"], record["encoder"]): record
        for record in records
    }
    rows = []
    for mode in MODE_ORDER:
        for backbone in BACKBONE_ORDER:
            nct_c = by_combo.get((mode, backbone, "NCT_C"))
            nct_s = by_combo.get((mode, backbone, "NCT_S"))
            row: dict[str, Any] = {
                "mode": mode,
                "backbone": backbone,
                "target_space": P3_TARGET,
                "subject_conditioning": "off",
                "avg": P3_AVG,
                "seed": P3_SEED,
            }
            if nct_c:
                row["nct_c_top1"] = nct_c.get("top1")
                row["nct_c_top5"] = nct_c.get("top5")
            if nct_s:
                row["nct_s_top1"] = nct_s.get("top1")
                row["nct_s_top5"] = nct_s.get("top5")
            if nct_c and nct_s:
                if nct_c.get("top1") is not None and nct_s.get("top1") is not None:
                    row["delta_top1_nct_s_minus_nct_c"] = nct_s["top1"] - nct_c["top1"]
                if nct_c.get("top5") is not None and nct_s.get("top5") is not None:
                    row["delta_top5_nct_s_minus_nct_c"] = nct_s["top5"] - nct_c["top5"]
            rows.append(row)
    return rows


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
    p3_records = filter_p3(records)

    main_rows = main_table(p3_records)
    contrast_rows = contrast_table(p3_records)

    write_csv(output_dir / "encoder_backbone_main_table.csv", main_rows)
    write_csv(output_dir / "encoder_contrast_table.csv", contrast_rows)

    summary = {
        "num_all_records": len(records),
        "num_p3_records": len(p3_records),
        "expected_p3_records": len(MODE_ORDER) * len(BACKBONE_ORDER) * len(ENCODER_ORDER),
        "outputs": {
            "main_table": str(output_dir / "encoder_backbone_main_table.csv"),
            "contrast_table": str(output_dir / "encoder_contrast_table.csv"),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
