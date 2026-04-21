#!/usr/bin/env bash
set -u

GPU="cuda:0"
EPOCHS="100"
DRY_RUN="false"
MODE="both"
SEEDS=(3407)
MANUAL_CONFIG=""
EXTRA_SETS=()
COLLECT_CSV="outputs/collected/p2_best_rerun_results.csv"
COLLECT_JSON="outputs/collected/p2_best_rerun_results.json"
TABLE_DIR="outputs/collected/p2"
SELECTION_JSON="${TABLE_DIR}/selection.json"

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_p2_best_rerun.sh [--mode within|cross|both] [--gpu cuda:0] [--epochs 100] [--seeds "3407"] [--config configs/p2_targetspace_cross.yaml --set clip.variant=ViT-H-14 --set ubp.target_space=adaptive --set ubp.mode=adaptive] [--dry-run]

Description:
  Stage C of P2. Re-runs the best within-setting and/or best cross-setting with one or more extra seeds.

Behavior:
  - Auto mode: if --config is not provided, the script reads outputs/collected/p2/selection.json
    and reruns the current best within/cross settings.
  - Manual mode: if --config is provided, the script reruns that config with the supplied --set overrides.
  - All runs keep:
    - encoder_type=NCT_S
    - subject_conditioning=false
    - avg=true
  - Ends by refreshing collected results and P2 tables.

Examples:
  bash scripts/run_p2_best_rerun.sh --dry-run
  bash scripts/run_p2_best_rerun.sh --mode both --seeds "3407"
  bash scripts/run_p2_best_rerun.sh --config configs/p2_targetspace_cross.yaml --set clip.variant=ViT-H-14 --set ubp.target_space=adaptive --set ubp.mode=adaptive --seeds "3407 42"
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
    --config)
      MANUAL_CONFIG="$2"
      shift 2
      ;;
    --set)
      EXTRA_SETS+=("$2")
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

prepare_tables() {
  python tools/collect_results.py \
    --input-root outputs \
    --output-csv "${COLLECT_CSV}" \
    --output-json "${COLLECT_JSON}" >/dev/null

  python tools/make_p2_tables.py \
    --input-root outputs \
    --output-dir "${TABLE_DIR}" >/dev/null
}

load_selection() {
  local mode_key="$1"
  python - "$SELECTION_JSON" "$mode_key" <<'PY'
import json
import sys

path = sys.argv[1]
mode_key = sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
setting = data.get("best_settings", {}).get(mode_key, {})
if not setting.get("backbone"):
    sys.exit(1)
config_path = "configs/p2_targetspace_within.yaml" if mode_key == "within" else "configs/p2_targetspace_cross.yaml"
target = setting.get("target_space", "curriculum")
overrides = [
    f"clip.variant={setting.get('backbone', '')}",
    "model.encoder_type=NCT_S",
    "model.subject_conditioning=false",
    "data.avg=true",
    f"ubp.target_space={target}",
    f"ubp.mode={target}",
]
print(config_path)
for item in overrides:
    print(item)
PY
}

FAILED=0
CURRENT_INDEX=0
TOTAL_RUNS=0

if [[ -n "$MANUAL_CONFIG" ]]; then
  TOTAL_RUNS=${#SEEDS[@]}
  for seed in "${SEEDS[@]}"; do
    CURRENT_INDEX=$((CURRENT_INDEX + 1))
    cmd_args=(
      --config "${MANUAL_CONFIG}"
      --set train.gpu="${GPU}"
      --set train.epochs="${EPOCHS}"
      --set train.seed="${seed}"
      --set model.encoder_type=NCT_S
      --set model.subject_conditioning=false
      --set data.avg=true
    )
    for item in "${EXTRA_SETS[@]}"; do
      cmd_args+=(--set "${item}")
    done
    run_one "p2C_manual_seed${seed}" "${cmd_args[@]}" || FAILED=1
  done
else
  prepare_tables

  SELECTED_MODES=()
  if [[ "$MODE" == "within" || "$MODE" == "both" ]]; then
    SELECTED_MODES+=(within)
  fi
  if [[ "$MODE" == "cross" || "$MODE" == "both" ]]; then
    SELECTED_MODES+=(cross)
  fi

  TOTAL_RUNS=$(( ${#SELECTED_MODES[@]} * ${#SEEDS[@]} ))

  for mode_key in "${SELECTED_MODES[@]}"; do
    mapfile -t selection_lines < <(load_selection "${mode_key}")
    if [[ "${#selection_lines[@]}" -lt 2 ]]; then
      echo "Failed to resolve best ${mode_key} setting from ${SELECTION_JSON}. Run Stage A/B first." >&2
      exit 1
    fi

    config_path="${selection_lines[0]}"
    overrides=()
    for ((i=1; i<${#selection_lines[@]}; i++)); do
      if [[ -n "${selection_lines[$i]}" ]]; then
        overrides+=("${selection_lines[$i]}")
      fi
    done

    for seed in "${SEEDS[@]}"; do
      CURRENT_INDEX=$((CURRENT_INDEX + 1))
      cmd_args=(
        --config "${config_path}"
        --set train.gpu="${GPU}"
        --set train.epochs="${EPOCHS}"
        --set train.seed="${seed}"
      )
      for item in "${overrides[@]}"; do
        cmd_args+=(--set "${item}")
      done
      run_one "p2C_${mode_key}_seed${seed}" "${cmd_args[@]}" || FAILED=1
    done
  done
fi

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
