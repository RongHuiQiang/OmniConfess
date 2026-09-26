#!/bin/bash
# run_qwen3_grid_cell.sh -- run ONE Qwen3(=Qwen2.5-Omni "_qwen3" runner) baseline
# grid cell on ONE card, then merge->score->stamp. Single-card single-task.
# Mirrors the canonical live launch template + run_base_cell.sh scoring tail.
#
# Usage: run_qwen3_grid_cell.sh <gpu> <dataset> <method> <data_json> <expected_rows>
#   <method> in: cad|dola|gd|pd|sc|tot|vcd   (avcd/opera have NO _qwen3 runner)
set -u
cd "$(dirname "$0")"
PY=/slow_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS
export HF_HOME=/slow_share/tsq/tsq_data/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GPU=$1; DS=$2; METHOD=$3; DATA=$4; EXP=$5
CELL="${DS}_qwen3_${METHOD}"
OUT="outputs/${CELL}"
FILE="${CELL}_full"
RAW="$OUT/${FILE}.jsonl"
SCORED="$OUT/${FILE}_single_scored.jsonl"
LOG="$OUT/run.log"
mkdir -p "$OUT" "$OUT/time"

RUNNER="OmniConfess/methods/base_${METHOD}_qwen3.py"
if [ ! -f "$RUNNER" ]; then echo "[cell] NO RUNNER $RUNNER" | tee -a "$LOG"; exit 5; fi

echo "[cell $(date +%T)] $CELL gpu=$GPU exp=$EXP data=$DATA" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PY "$RUNNER" \
    --datasets "$DS" \
    --data_path "$DATA" \
    --data_root "$PWD/OmniHalluBench/${DS}/" \
    --output_dir "$OUT" \
    --time_path "$OUT/time" \
    --file_name "$FILE" \
    --prompt_version v1 \
    --no-record_process \
    >> "$LOG" 2>&1
RC=$?
echo "[cell $(date +%T)] runner exit=$RC" | tee -a "$LOG"
if [ $RC -ne 0 ]; then echo "[cell] FAILED $CELL rc=$RC" | tee -a "$LOG"; exit $RC; fi
if [ ! -f "$RAW" ]; then echo "[cell] no raw $RAW" | tee -a "$LOG"; exit 3; fi
NROWS=$(wc -l < "$RAW")
if [ "$NROWS" -lt "$EXP" ]; then
  echo "[cell] SHORT $CELL rows=$NROWS/$EXP -- not scoring, will retry next round" | tee -a "$LOG"; exit 6
fi

# score (from cell dir so shared CSVs aren't polluted)
REPO="$(pwd)"
unset ZHIPU_API_KEY
( cd "$OUT" && $PY "$REPO/OmniConfess/eval/single_eval.py" --target_files "$REPO/$RAW" ) >> "$LOG" 2>&1
SC=$?
if [ $SC -ne 0 ] || [ ! -f "$SCORED" ]; then echo "[cell] SCORING FAILED $CELL sc=$SC" | tee -a "$LOG"; exit 4; fi
SN=$(wc -l < "$SCORED")
echo "DONE ${METHOD} $(date '+%Y-%m-%d %H:%M:%S') rows=$SN" > "$OUT/cell_done.txt"
echo "[cell $(date +%T)] STAMPED cell_done.txt for $CELL rows=$SN" | tee -a "$LOG"
