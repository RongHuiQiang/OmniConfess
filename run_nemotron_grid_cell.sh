#!/bin/bash
# run_nemotron_grid_cell.sh -- run ONE Nemotron baseline grid cell on ONE card,
# then score + stamp. Mirrors run_qwen3_grid_cell.sh; differences: runner venv
# is /slow_share/tsq/venvs/nemotron-hf (transformers 5.6.2 + trust_remote_code
# + compiled Mamba2 kernels), env exports mirror run_nemotron_sc_desc.sh, runner
# class is base_<method>_nemotron_hf.py.
#
# Usage: run_nemotron_grid_cell.sh <gpu> <dataset> <method> <data_json> <expected_rows>
#   <method> in: cad|dola|gd|pd|tot|vcd|avcd|opera
set -u
cd "$(dirname "$0")"

PY=/slow_share/tsq/venvs/nemotron-hf/bin/python
EVALPY=/slow_share/tsq/venvs/omniconfess/bin/python
export TMPDIR=/slow_share/tsq/tsq/tmp
export PYTHONPATH=/home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/methods:${PYTHONPATH:-}
export HF_HOME=/slow_share/tsq/tsq_data/.cache/huggingface
# 2026-08-30 emptyrun root cause: the shell that launches this script (daemon
# or login) still carries TRITON_CACHE_DIR/TORCH_HOME=/ultrafast_share/tsq/...,
# which was wiped on 2026-08-29. NemotronH's Mamba2 mixers launch triton
# kernels on EVERY forward; the first launch os.makedirs()es the cache root and
# dies with PermissionError(13, '/ultrafast_share/tsq') -> every sample fails,
# runners write empty rows, the cell gets scored + stamped = empty cell.
# Override to the slow_share cache (same place HF_HOME moved to). The kernels
# recompile there once per (triton, torch) build and are reused afterwards.
export TRITON_CACHE_DIR=/slow_share/tsq/tsq_data/.cache/triton
export TORCH_HOME=/slow_share/tsq/tsq_data/.cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export NEMO_REPO=/home/tsq/rhq/omniconfess/OmniConfess
export NEMO_KEEP_CACHE=1
export MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS
mkdir -p "$TMPDIR"

GPU=$1; DS=$2; METHOD=$3; DATA=$4; EXP=$5
CELL="${DS}_nemotron_${METHOD}"
OUT="outputs/${CELL}"
FILE="${CELL}_full"
RAW="$OUT/${FILE}.jsonl"
SCORED="$OUT/${FILE}_single_scored.jsonl"
LOG="$OUT/run.log"
mkdir -p "$OUT" "$OUT/time"

RUNNER="OmniConfess/methods/base_${METHOD}_nemotron_hf.py"
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

# score (from cell dir so shared CSVs aren't polluted; zhipu unset -> GAVIE frozen)
REPO="$(pwd)"
unset ZHIPU_API_KEY
( cd "$OUT" && $EVALPY "$REPO/OmniConfess/eval/single_eval.py" --target_files "$REPO/$RAW" ) >> "$LOG" 2>&1
SC=$?
if [ $SC -ne 0 ] || [ ! -f "$SCORED" ]; then echo "[cell] SCORING FAILED $CELL sc=$SC" | tee -a "$LOG"; exit 4; fi
SN=$(wc -l < "$SCORED")
echo "DONE ${METHOD} $(date '+%Y-%m-%d %H:%M:%S') rows=$SN" > "$OUT/cell_done.txt"
echo "[cell $(date +%T)] STAMPED cell_done.txt for $CELL rows=$SN" | tee -a "$LOG"
