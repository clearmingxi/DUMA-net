#!/usr/bin/env bash
set -u

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_ablation_encoder_avg.sh [--mode within|cross|both] [--gpu cuda:0] [--epochs 100] [--dry-run]

Examples:
  bash scripts/run_ablation_encoder_avg.sh --mode cross --gpu cuda:0
  bash scripts/run_ablation_encoder_avg.sh --mode both --epochs 1 --dry-run

Description:
  Runs encoder and averaging ablations through run_experiment.py.
  Sweeps:
    - mode: within_subject / cross_subject
    - encoder_type: NCT_C / NCT_S
    - avg: true / false
  Fixed defaults:
    - subject_conditioning: auto
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
  local encoder="$2"
  local avg="$3"
  local config="configs/within_subject_baseline.yaml"
  local log_file="$LOG_DIR/encoder_avg_${mode}_${encoder}_avg-${avg}.log"

  if [[ "$mode" == "cross" ]]; then
    config="configs/cross_subject_baseline.yaml"
  fi

  local cmd=(
    python run_experiment.py
    --config "$config"
    --set "train.gpu=${GPU}"
    --set "train.epochs=${EPOCHS}"
    --set "model.encoder_type=${encoder}"
    --set "data.avg=${avg}"
  )

  if [[ "$DRY_RUN" == "true" ]]; then
    cmd+=(--dry-run)
  fi

  printf '[run] mode=%s encoder=%s avg=%s\n' "$mode" "$encoder" "$avg" | tee "$log_file"
  "${cmd[@]}" | tee -a "$log_file"
}

for encoder in NCT_C NCT_S; do
  for avg in true false; do
    if [[ "$MODE" == "within" || "$MODE" == "both" ]]; then
      run_one "within" "$encoder" "$avg"
    fi
    if [[ "$MODE" == "cross" || "$MODE" == "both" ]]; then
      run_one "cross" "$encoder" "$avg"
    fi
  done
done
