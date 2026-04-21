#!/usr/bin/env bash
set -u

GPU="cuda:0"
EPOCHS="100"
DRY_RUN="false"
SEED="3047"
MODES=(within cross)
ENCODERS=(NCT_C NCT_S)
BACKBONES=(RN50 ViT-B-32 ViT-H-14)
COLLECT_CSV="outputs/collected/p3_encoder_backbone_results.csv"
COLLECT_JSON="outputs/collected/p3_encoder_backbone_results.json"
TABLE_DIR="outputs/collected/p3"

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_p3_encoder_backbone_matrix.sh [--gpu cuda:0] [--epochs 100] [--dry-run]

Description:
  Runs the fixed P3 encoder-backbone matrix:
    - modes: within / cross
    - encoders: NCT_C / NCT_S
    - backbones: RN50 / ViT-B-32 / ViT-H-14
    - target_space=curriculum
    - subject_conditioning=false
    - avg=true
    - seed=3047

This is exactly 12 runs. It does not include hybrid/adaptive/orig_only,
subject_conditioning=on, avg=false, extra backbones, or extra seeds.

Examples:
  bash scripts/run_p3_encoder_backbone_matrix.sh --dry-run
  bash scripts/run_p3_encoder_backbone_matrix.sh --gpu cuda:0 --epochs 100
  nohup bash scripts/run_p3_encoder_backbone_matrix.sh --gpu cuda:0 --epochs 100 > outputs/p3_matrix.log 2>&1 &
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

config_for_mode() {
  local mode="$1"
  if [[ "$mode" == "within" ]]; then
    echo "configs/p2_backbone_within.yaml"
  elif [[ "$mode" == "cross" ]]; then
    echo "configs/p2_backbone_cross.yaml"
  else
    echo "Invalid mode: $mode" >&2
    return 1
  fi
}

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

TOTAL_RUNS=$((${#MODES[@]} * ${#ENCODERS[@]} * ${#BACKBONES[@]}))
CURRENT_INDEX=0
FAILED=0

for mode in "${MODES[@]}"; do
  config="$(config_for_mode "$mode")" || exit 1
  for backbone in "${BACKBONES[@]}"; do
    for encoder in "${ENCODERS[@]}"; do
      CURRENT_INDEX=$((CURRENT_INDEX + 1))
      run_one "p3_${mode}_${backbone}_${encoder}_curriculum_subject_conditioning_off_avg_true_seed${SEED}" \
        --config "${config}" \
        --set experiment.name="p3_encoder_backbone_${mode}" \
        --set train.gpu="${GPU}" \
        --set train.epochs="${EPOCHS}" \
        --set train.seed="${SEED}" \
        --set clip.variant="${backbone}" \
        --set model.encoder_type="${encoder}" \
        --set model.subject_conditioning=false \
        --set data.avg=true \
        --set ubp.target_space=curriculum \
        --set ubp.mode=curriculum || FAILED=1
    done
  done
done

if [[ "$DRY_RUN" == "true" ]]; then
  if [[ "$FAILED" -ne 0 ]]; then
    exit 1
  fi
  exit 0
fi

echo "[collect] ${COLLECT_CSV}"
python tools/collect_results.py \
  --input-root outputs \
  --output-csv "${COLLECT_CSV}" \
  --output-json "${COLLECT_JSON}"

echo "[tables] ${TABLE_DIR}"
python tools/make_p3_tables.py \
  --input-root outputs \
  --output-dir "${TABLE_DIR}"

if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
