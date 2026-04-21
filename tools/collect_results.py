import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_utils import compute_summary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect normalized experiment outputs produced by run_experiment.py."
    )
    parser.add_argument(
        "--input-root",
        default="outputs",
        help="Root directory to scan for run folders.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path for a consolidated CSV file.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for a consolidated JSON file.",
    )
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
            "summary_csv": str(summary_path),
            "aggregates": compute_summary_metrics(rows),
        }
    return {}


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "run_dir": record.get("run_dir"),
        "output_dir": record.get("output_dir", record.get("run_dir")),
        "status": record.get("status"),
        "mode": record.get("mode"),
        "backbone": record.get("backbone", record.get("clip_variant")),
        "clip_variant": record.get("clip_variant"),
        "encoder_type": record.get("encoder_type"),
        "target_space": record.get("target_space", record.get("ubp_mode")),
        "ubp_mode": record.get("ubp_mode"),
        "insubject": record.get("insubject"),
        "subject_conditioning": record.get("subject_conditioning"),
        "subject_conditioning_enabled": record.get("subject_conditioning_enabled"),
        "avg": record.get("avg"),
        "seed": record.get("seed"),
        "summary_csv": record.get("summary_csv"),
    }
    aggregates = record.get("aggregates", {})
    for key, value in aggregates.items():
        flat[key] = value
    return flat


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    run_dirs = sorted(path.parent for path in root.rglob("config_snapshot.yaml"))

    records = []
    for run_dir in run_dirs:
        loaded = load_metrics(run_dir)
        if loaded:
            records.append(loaded)

    flat_records = [flatten_record(record) for record in records]

    if args.output_csv and flat_records:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in flat_records for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_records)

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=True)

    print(json.dumps(flat_records, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
