#!/usr/bin/env bash
set -u

MODE="both"
GPU="cuda:0"
EPOCHS="100"
DRY_RUN="false"
SEEDS=(3047)
TARGETS=(orig_only curriculum adaptive hybrid)
WITHIN_BACKBONE=""
CROSS_BACKBONE=""
COLLECT_CSV="outputs/collected/p2_targetspace_results.csv"
COLLECT_JSON="outputs/collected/p2_targetspace_results.json"
TABLE_DIR="outputs/collected/p2"
SELECTION_JSON="${TABLE_DIR}/selection.json"

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_p2_targetspace_matrix.sh [--mode within|cross|both] [--gpu cuda:0] [--epochs 100] [--seeds "3047 3407"] [--within-backbone RN50] [--cross-backbone ViT-H-14] [--targets "orig_only curriculum adaptive hybrid"] [--dry-run]

Description:
  Stage B of P2. Runs target-space sweeps with fixed:
    - encoder_type=NCT_S
    - subject_conditioning=false
    - avg=true

Behavior:
  - If --within-backbone / --cross-backbone are not provided, the script auto-selects
    the best backbone from Stage A using tools/make_p2_tables.py.
  - within mode uses the within-selected backbone only.
  - cross mode uses the cross-selected backbone only.

Examples:
  bash scripts/run_p2_targetspace_matrix.sh --dry-run
  bash scripts/run_p2_targetspace_matrix.sh --mode both --gpu cuda:0 --epochs 100 --seeds "3047"
  bash scripts/run_p2_targetspace_matrix.sh --mode cross --cross-backbone ViT-H-14 --seeds "3047 3407"
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
    --within-backbone)
      WITHIN_BACKBONE="$2"
      shift 2
      ;;
    --cross-backbone)
      CROSS_BACKBONE="$2"
      shift 2
      ;;
    --targets)
      read -r -a TARGETS <<< "$2"
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

echo "[prepare] refreshing P2 tables for backbone auto-selection"
python tools/collect_results.py \
  --input-root outputs \
  --output-csv "${COLLECT_CSV}" \
  --output-json "${COLLECT_JSON}" >/dev/null

python tools/make_p2_tables.py \
  --input-root outputs \
  --output-dir "${TABLE_DIR}" >/dev/null

if [[ -z "$WITHIN_BACKBONE" ]]; then
  WITHIN_BACKBONE="$(python - "$SELECTION_JSON" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
print(data.get("best_backbones", {}).get("within", ""))
PY
)"
fi

if [[ -z "$CROSS_BACKBONE" ]]; then
  CROSS_BACKBONE="$(python - "$SELECTION_JSON" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
print(data.get("best_backbones", {}).get("cross", ""))
PY
)"
fi

if [[ "$MODE" == "within" || "$MODE" == "both" ]]; then
  if [[ -z "$WITHIN_BACKBONE" ]]; then
    echo "Failed to resolve within backbone from ${SELECTION_JSON}. Run Stage A first or pass --within-backbone." >&2
    exit 1
  fi
fi

if [[ "$MODE" == "cross" || "$MODE" == "both" ]]; then
  if [[ -z "$CROSS_BACKBONE" ]]; then
    echo "Failed to resolve cross backbone from ${SELECTION_JSON}. Run Stage A first or pass --cross-backbone." >&2
    exit 1
  fi
fi

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
  for target in "${TARGETS[@]}"; do
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
  for target in "${TARGETS[@]}"; do
    if [[ "$MODE" == "within" || "$MODE" == "both" ]]; then
      CURRENT_INDEX=$((CURRENT_INDEX + 1))
      run_one "p2B_within_${WITHIN_BACKBONE}_${target}_seed${seed}" \
        --config configs/p2_targetspace_within.yaml \
        --set train.gpu="${GPU}" \
        --set train.epochs="${EPOCHS}" \
        --set train.seed="${seed}" \
        --set clip.variant="${WITHIN_BACKBONE}" \
        --set model.encoder_type=NCT_S \
        --set model.subject_conditioning=false \
        --set data.avg=true \
        --set ubp.target_space="${target}" \
        --set ubp.mode="${target}" || FAILED=1
    fi

    if [[ "$MODE" == "cross" || "$MODE" == "both" ]]; then
      CURRENT_INDEX=$((CURRENT_INDEX + 1))
      run_one "p2B_cross_${CROSS_BACKBONE}_${target}_seed${seed}" \
        --config configs/p2_targetspace_cross.yaml \
        --set train.gpu="${GPU}" \
        --set train.epochs="${EPOCHS}" \
        --set train.seed="${seed}" \
        --set clip.variant="${CROSS_BACKBONE}" \
        --set model.encoder_type=NCT_S \
        --set model.subject_conditioning=false \
        --set data.avg=true \
        --set ubp.target_space="${target}" \
        --set ubp.mode="${target}" || FAILED=1
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
