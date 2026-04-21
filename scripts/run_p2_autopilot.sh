#!/usr/bin/env bash
set -u

GPU="cuda:0"
EPOCHS="100"
SEED_MAIN="3047"
SEED_RERUN="3407"
DRY_RUN="false"

COLLECT_DIR="outputs/collected"
P2_DIR="${COLLECT_DIR}/p2"

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_p2_autopilot.sh [--gpu cuda:0] [--epochs 100] [--seed-main 3047] [--seed-rerun 3407] [--dry-run]

Description:
  Autopilot runner for the full P2 pipeline:
    1. Stage A: backbone matrix
    2. Stage B: target-space matrix
    3. Stage C: best-setting rerun
    4. Final refresh of collected results and P2 tables

Examples:
  bash scripts/run_p2_autopilot.sh --dry-run
  bash scripts/run_p2_autopilot.sh --gpu cuda:0 --epochs 100 --seed-main 3047 --seed-rerun 3407
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --seed-main)
      SEED_MAIN="$2"
      shift 2
      ;;
    --seed-rerun)
      SEED_RERUN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

mkdir -p "${COLLECT_DIR}" "${P2_DIR}"

run_cmd() {
  local stage_name="$1"
  shift

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] $*"
    return 0
  fi

  "$@" || {
    echo "${stage_name} failed" >&2
    exit 1
  }
}

echo "[Stage A]"
run_cmd "[Stage A]" \
  bash scripts/run_p2_backbone_matrix.sh \
  --mode both \
  --gpu "${GPU}" \
  --epochs "${EPOCHS}" \
  --seeds "${SEED_MAIN}" \
  $( [[ "$DRY_RUN" == "true" ]] && printf '%s' "--dry-run" )
echo "completed"

echo "[Stage B]"
run_cmd "[Stage B]" \
  bash scripts/run_p2_targetspace_matrix.sh \
  --mode both \
  --gpu "${GPU}" \
  --epochs "${EPOCHS}" \
  --seeds "${SEED_MAIN}" \
  $( [[ "$DRY_RUN" == "true" ]] && printf '%s' "--dry-run" )
echo "completed"

echo "[Stage C]"
run_cmd "[Stage C]" \
  bash scripts/run_p2_best_rerun.sh \
  --mode both \
  --gpu "${GPU}" \
  --epochs "${EPOCHS}" \
  --seeds "${SEED_RERUN}" \
  $( [[ "$DRY_RUN" == "true" ]] && printf '%s' "--dry-run" )
echo "completed"

echo "[Final Refresh]"
run_cmd "[Final Refresh]" \
  python tools/collect_results.py \
  --input-root outputs \
  --output-csv outputs/collected/p2_autopilot_results.csv \
  --output-json outputs/collected/p2_autopilot_results.json

run_cmd "[Final Refresh]" \
  python tools/make_p2_tables.py \
  --input-root outputs \
  --output-dir outputs/collected/p2

echo "completed"
echo "results_dir=outputs/collected/p2"
echo "key_outputs:"
echo "  outputs/collected/p2_backbone_results.csv"
echo "  outputs/collected/p2_backbone_results.json"
echo "  outputs/collected/p2_targetspace_results.csv"
echo "  outputs/collected/p2_targetspace_results.json"
echo "  outputs/collected/p2_best_rerun_results.csv"
echo "  outputs/collected/p2_best_rerun_results.json"
echo "  outputs/collected/p2_autopilot_results.csv"
echo "  outputs/collected/p2_autopilot_results.json"
echo "  outputs/collected/p2/backbone_main_table.csv"
echo "  outputs/collected/p2/targetspace_main_table.csv"
echo "  outputs/collected/p2/best_setting_summary.csv"
echo "  outputs/collected/p2/selection.json"
