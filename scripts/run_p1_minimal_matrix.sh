#!/usr/bin/env bash
set -u

GPU="cuda:0"
EPOCHS="100"
DRY_RUN="false"
SEEDS=(3047)

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_p1_minimal_matrix.sh [--gpu cuda:0] [--epochs 100] [--seeds "3047 3407 42"] [--dry-run]

Description:
  Runs the fixed 8-experiment P1 minimal matrix through run_experiment.py.

Matrix:
  A. Subject conditioning ablation
    1. within + conditioning=true
    2. within + conditioning=false
    3. cross + conditioning=true
    4. cross + conditioning=false
  B. Cross-subject encoder + avg ablation
    5. cross + NCT_C + avg=true
    6. cross + NCT_C + avg=false
    7. cross + NCT_S + avg=true
    8. cross + NCT_S + avg=false

Fixed settings:
  - backbone=RN50
  - target_space=curriculum
  - within config: configs/within_subject_baseline.yaml
  - cross config: configs/cross_subject_baseline.yaml

Examples:
  bash scripts/run_p1_minimal_matrix.sh --dry-run
  bash scripts/run_p1_minimal_matrix.sh --gpu cuda:0 --epochs 100 --seeds "3047"
  bash scripts/run_p1_minimal_matrix.sh --gpu cuda:1 --epochs 50 --seeds "3047 3407 42"
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
    --seeds)
      read -r -a SEEDS <<< "$2"
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

run_one() {
  local label="$1"
  shift

  echo "[$CURRENT_INDEX/$TOTAL_RUNS] $label"

  local cmd=(
    python run_experiment.py
    "$@"
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

TOTAL_RUNS=$(( ${#SEEDS[@]} * 8 ))
CURRENT_INDEX=0
FAILED=0

for seed in "${SEEDS[@]}"; do
  CURRENT_INDEX=$((CURRENT_INDEX + 1))
  run_one "within_cond_on_seed${seed}" \
    --config configs/within_subject_baseline.yaml \
    --set train.gpu="${GPU}" \
    --set train.epochs="${EPOCHS}" \
    --set train.seed="${seed}" \
    --set clip.variant=RN50 \
    --set ubp.target_space=curriculum \
    --set ubp.mode=curriculum \
    --set model.encoder_type=NCT_C \
    --set model.subject_conditioning=true \
    --set data.avg=true || FAILED=1

  CURRENT_INDEX=$((CURRENT_INDEX + 1))
  run_one "within_cond_off_seed${seed}" \
    --config configs/within_subject_baseline.yaml \
    --set train.gpu="${GPU}" \
    --set train.epochs="${EPOCHS}" \
    --set train.seed="${seed}" \
    --set clip.variant=RN50 \
    --set ubp.target_space=curriculum \
    --set ubp.mode=curriculum \
    --set model.encoder_type=NCT_C \
    --set model.subject_conditioning=false \
    --set data.avg=true || FAILED=1

  CURRENT_INDEX=$((CURRENT_INDEX + 1))
  run_one "cross_cond_on_seed${seed}" \
    --config configs/cross_subject_baseline.yaml \
    --set train.gpu="${GPU}" \
    --set train.epochs="${EPOCHS}" \
    --set train.seed="${seed}" \
    --set clip.variant=RN50 \
    --set ubp.target_space=curriculum \
    --set ubp.mode=curriculum \
    --set model.encoder_type=NCT_C \
    --set model.subject_conditioning=true \
    --set data.avg=true || FAILED=1

  CURRENT_INDEX=$((CURRENT_INDEX + 1))
  run_one "cross_cond_off_seed${seed}" \
    --config configs/cross_subject_baseline.yaml \
    --set train.gpu="${GPU}" \
    --set train.epochs="${EPOCHS}" \
    --set train.seed="${seed}" \
    --set clip.variant=RN50 \
    --set ubp.target_space=curriculum \
    --set ubp.mode=curriculum \
    --set model.encoder_type=NCT_C \
    --set model.subject_conditioning=false \
    --set data.avg=true || FAILED=1

  CURRENT_INDEX=$((CURRENT_INDEX + 1))
  run_one "cross_nctc_avg_true_seed${seed}" \
    --config configs/cross_subject_baseline.yaml \
    --set train.gpu="${GPU}" \
    --set train.epochs="${EPOCHS}" \
    --set train.seed="${seed}" \
    --set clip.variant=RN50 \
    --set ubp.target_space=curriculum \
    --set ubp.mode=curriculum \
    --set model.encoder_type=NCT_C \
    --set model.subject_conditioning=false \
    --set data.avg=true || FAILED=1

  CURRENT_INDEX=$((CURRENT_INDEX + 1))
  run_one "cross_nctc_avg_false_seed${seed}" \
    --config configs/cross_subject_baseline.yaml \
    --set train.gpu="${GPU}" \
    --set train.epochs="${EPOCHS}" \
    --set train.seed="${seed}" \
    --set clip.variant=RN50 \
    --set ubp.target_space=curriculum \
    --set ubp.mode=curriculum \
    --set model.encoder_type=NCT_C \
    --set model.subject_conditioning=false \
    --set data.avg=false || FAILED=1

  CURRENT_INDEX=$((CURRENT_INDEX + 1))
  run_one "cross_ncts_avg_true_seed${seed}" \
    --config configs/cross_subject_baseline.yaml \
    --set train.gpu="${GPU}" \
    --set train.epochs="${EPOCHS}" \
    --set train.seed="${seed}" \
    --set clip.variant=RN50 \
    --set ubp.target_space=curriculum \
    --set ubp.mode=curriculum \
    --set model.encoder_type=NCT_S \
    --set model.subject_conditioning=false \
    --set data.avg=true || FAILED=1

  CURRENT_INDEX=$((CURRENT_INDEX + 1))
  run_one "cross_ncts_avg_false_seed${seed}" \
    --config configs/cross_subject_baseline.yaml \
    --set train.gpu="${GPU}" \
    --set train.epochs="${EPOCHS}" \
    --set train.seed="${seed}" \
    --set clip.variant=RN50 \
    --set ubp.target_space=curriculum \
    --set ubp.mode=curriculum \
    --set model.encoder_type=NCT_S \
    --set model.subject_conditioning=false \
    --set data.avg=false || FAILED=1
done

echo "[collect] outputs/collected/p1_minimal_results.csv"
if [[ "$DRY_RUN" == "true" ]]; then
  python tools/collect_results.py \
    --input-root outputs \
    --output-csv outputs/collected/p1_minimal_results.csv \
    --output-json outputs/collected/p1_minimal_results.json
else
  python tools/collect_results.py \
    --input-root outputs \
    --output-csv outputs/collected/p1_minimal_results.csv \
    --output-json outputs/collected/p1_minimal_results.json
fi

if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
