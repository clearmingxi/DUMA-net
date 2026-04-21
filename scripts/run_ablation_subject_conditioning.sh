#!/usr/bin/env bash
set -u

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_ablation_subject_conditioning.sh [--mode within|cross|both] [--gpu cuda:0] [--epochs 100] [--dry-run]

Examples:
  bash scripts/run_ablation_subject_conditioning.sh --mode within --gpu cuda:0
  bash scripts/run_ablation_subject_conditioning.sh --mode both --epochs 1 --dry-run

Description:
  Runs subject-conditioning ablations through run_experiment.py.
  Sweeps:
    - mode: within_subject / cross_subject
    - subject_conditioning: auto / true / false
  Fixed defaults:
    - encoder_type: NCT_C
    - avg: true
    - target_space: curriculum
EOF
}

MODE="both"
GPU="cuda:0"
EPOCHS="100"
DRY_RUN="false"
LOG_DIR="outputs/script_logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
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

mkdir -p "$LOG_DIR"

run_one() {
  local mode="$1"
  local conditioning="$2"
  local config="configs/within_subject_baseline.yaml"
  local log_file="$LOG_DIR/subject_conditioning_${mode}_${conditioning}.log"

  if [[ "$mode" == "cross" ]]; then
    config="configs/cross_subject_baseline.yaml"
  fi

  local cmd=(
    python run_experiment.py
    --config "$config"
    --set "train.gpu=${GPU}"
    --set "train.epochs=${EPOCHS}"
    --set "model.subject_conditioning=${conditioning}"
  )

  if [[ "$DRY_RUN" == "true" ]]; then
    cmd+=(--dry-run)
  fi

  printf '[run] mode=%s subject_conditioning=%s\n' "$mode" "$conditioning" | tee "$log_file"
  "${cmd[@]}" | tee -a "$log_file"
}

if [[ "$MODE" == "within" || "$MODE" == "both" ]]; then
  for conditioning in auto true false; do
    run_one "within" "$conditioning"
  done
fi

if [[ "$MODE" == "cross" || "$MODE" == "both" ]]; then
  for conditioning in auto true false; do
    run_one "cross" "$conditioning"
  done
fi
