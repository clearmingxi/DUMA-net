#!/usr/bin/env bash
set -u

MODE="both"
GPU="cuda:0"
EPOCHS="100"
DRY_RUN="false"
SEEDS=(3047)
BACKBONES=(RN50 ViT-B-32 ViT-H-14)
COLLECT_CSV="outputs/collected/p2_backbone_results.csv"
COLLECT_JSON="outputs/collected/p2_backbone_results.json"
TABLE_DIR="outputs/collected/p2"

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_p2_backbone_matrix.sh [--mode within|cross|both] [--gpu cuda:0] [--epochs 100] [--seeds "3047 3407"] [--backbones "RN50 ViT-B-32 ViT-H-14"] [--dry-run]

Description:
  Stage A of P2. Runs the backbone comparison matrix with fixed:
    - encoder_type=NCT_S
    - subject_conditioning=false
    - avg=true
    - target_space=curriculum

Sweeps:
  - mode: within / cross / both
  - backbone: RN50 / ViT-B-32 / ViT-H-14
  - seed: one or more values

Examples:
  bash scripts/run_p2_backbone_matrix.sh --dry-run
  bash scripts/run_p2_backbone_matrix.sh --mode both --gpu cuda:0 --epochs 100 --seeds "3047"
  bash scripts/run_p2_backbone_matrix.sh --mode cross --gpu cuda:1 --epochs 100 --seeds "3047 3407"
EOF
}

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
    --seeds)
      read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --backbones)
      read -r -a BACKBONES <<< "$2"
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

case "$MODE" in
  within|cross|both)
    ;;
  *)
    echo "Invalid --mode: $MODE" >&2
    exit 1
    ;;
esac

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

TOTAL_RUNS=0
for seed in "${SEEDS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    if [[ "$MODE" == "within" || "$MODE" == "both" ]]; then
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
    fi
    if [[ "$MODE" == "cross" || "$MODE" == "both" ]]; then
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
    fi
  done
done

CURRENT_INDEX=0
FAILED=0

for seed in "${SEEDS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    if [[ "$MODE" == "within" || "$MODE" == "both" ]]; then
      CURRENT_INDEX=$((CURRENT_INDEX + 1))
      run_one "p2A_within_${backbone}_seed${seed}" \
        --config configs/p2_backbone_within.yaml \
        --set train.gpu="${GPU}" \
        --set train.epochs="${EPOCHS}" \
        --set train.seed="${seed}" \
        --set clip.variant="${backbone}" \
        --set model.encoder_type=NCT_S \
        --set model.subject_conditioning=false \
        --set data.avg=true \
        --set ubp.target_space=curriculum \
        --set ubp.mode=curriculum || FAILED=1
    fi

    if [[ "$MODE" == "cross" || "$MODE" == "both" ]]; then
      CURRENT_INDEX=$((CURRENT_INDEX + 1))
      run_one "p2A_cross_${backbone}_seed${seed}" \
        --config configs/p2_backbone_cross.yaml \
        --set train.gpu="${GPU}" \
        --set train.epochs="${EPOCHS}" \
        --set train.seed="${seed}" \
        --set clip.variant="${backbone}" \
        --set model.encoder_type=NCT_S \
        --set model.subject_conditioning=false \
        --set data.avg=true \
        --set ubp.target_space=curriculum \
        --set ubp.mode=curriculum || FAILED=1
    fi
  done
done

echo "[collect] ${COLLECT_CSV}"
python tools/collect_results.py \
  --input-root outputs \
  --output-csv "${COLLECT_CSV}" \
  --output-json "${COLLECT_JSON}"

echo "[tables] ${TABLE_DIR}"
python tools/make_p2_tables.py \
  --input-root outputs \
  --output-dir "${TABLE_DIR}"

if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
