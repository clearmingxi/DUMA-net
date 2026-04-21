#!/usr/bin/env bash
set -u

GPU="cuda:0"
EPOCHS="100"
SEED="3047"
DRY_RUN="false"

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/rerun_p1_cross_avg_false.sh [--gpu cuda:0] [--epochs 100] [--seed 3047] [--dry-run]

Description:
  Re-runs only the two unfinished P1 experiments:
    1. cross + RN50 + NCT_C + subject_conditioning=false + avg=false + curriculum
    2. cross + RN50 + NCT_S + subject_conditioning=false + avg=false + curriculum

Behavior:
  - Checks existing outputs/ for a completed run of the same combo and seed.
  - Skips combos that already have status=completed in metrics.json.
  - Re-runs only missing/incomplete combos.
  - Calls collect_results.py at the end.

Examples:
  bash scripts/rerun_p1_cross_avg_false.sh
  bash scripts/rerun_p1_cross_avg_false.sh --gpu cuda:1 --seed 3047
  bash scripts/rerun_p1_cross_avg_false.sh --epochs 1 --dry-run
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
    --seed)
      SEED="$2"
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

has_completed_run() {
  local encoder="$1"
  local metrics_files
  metrics_files=$(find outputs/cross/backbone_RN50/"encoder_${encoder}"/target_curriculum/insubject_off/subject_conditioning_off/avg_false/"seed_${SEED}" \
    -name metrics.json 2>/dev/null | sort -r || true)

  if [[ -z "$metrics_files" ]]; then
    return 1
  fi

  while IFS= read -r metrics; do
    [[ -z "$metrics" ]] && continue
    if python -c 'import json,sys; data=json.load(open(sys.argv[1])); sys.exit(0 if data.get("status")=="completed" else 1)' "$metrics"; then
      return 0
    fi
  done <<< "$metrics_files"

  return 1
}

run_one() {
  local index="$1"
  local total="$2"
  local label="$3"
  local encoder="$4"

  echo "[$index/$total] $label"

  if has_completed_run "$encoder"; then
    echo "[skip] $label already has a completed run for seed=${SEED}"
    return 0
  fi

  local cmd=(
    python run_experiment.py
    --config configs/cross_subject_baseline.yaml
    --set train.gpu="${GPU}"
    --set train.epochs="${EPOCHS}"
    --set train.seed="${SEED}"
    --set clip.variant=RN50
    --set ubp.target_space=curriculum
    --set ubp.mode=curriculum
    --set model.encoder_type="${encoder}"
    --set model.subject_conditioning=false
    --set data.avg=false
  )

  if [[ "$DRY_RUN" == "true" ]]; then
    "${cmd[@]}" --dry-run || {
      echo "[fail] $label" >&2
      return 1
    }
  else
    "${cmd[@]}" || {
      echo "[fail] $label" >&2
      return 1
    }
  fi
}

FAILED=0
run_one 1 2 "cross_nctc_avg_false_seed${SEED}" "NCT_C" || FAILED=1
run_one 2 2 "cross_ncts_avg_false_seed${SEED}" "NCT_S" || FAILED=1

echo "[collect] outputs/collected/p1_minimal_results.csv"
python tools/collect_results.py \
  --input-root outputs \
  --output-csv outputs/collected/p1_minimal_results.csv \
  --output-json outputs/collected/p1_minimal_results.json

if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
