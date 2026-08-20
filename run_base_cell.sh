#!/bin/bash
# run_base_cell.sh -- run ONE Qwen2.5-Omni base grid cell on ONE card, whole
# (or --max_samples-limited) dataset in a single process, then merge->score->stamp.
# Single-card single-task. Everything under setsid by the caller.
#
# Usage:
#   run_base_cell.sh <gpu> <dataset> <method> <data_json> <prompt_version> [max_samples]
#     <dataset>  : pubmedqa|ragtruth|haloquest|avhbench  (also the --datasets key)
#     <method>   : vcd|avcd|opera|tot|gd|pd
#     <data_json>: path under OmniHalluBench (full file)
#     <prompt_version>: v0|v1 ...
#     [max_samples]: optional; if set, only first N samples (for 1/3 degenerate cells)
set -u
cd "$(dirname "$0")"
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS

GPU=$1; DS=$2; METHOD=$3; DATA=$4; PV=$5; MAXS=${6:-0}

CELL="${DS}_base_${METHOD}"
OUT="outputs/${CELL}"
mkdir -p "$OUT" "$OUT/time"
FILE="${CELL}"                 # runner writes ${FILE}.jsonl
LOG="$OUT/run.log"

EXTRA=""
if [ "$MAXS" -gt 0 ]; then EXTRA="--max_samples $MAXS"; fi

echo "[cell $(date +%T)] $CELL gpu=$GPU pv=$PV maxs=$MAXS data=$DATA" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU $PY OmniConfess/methods/base_${METHOD}.py \
    --datasets "$DS" \
    --data_path "$DATA" \
    --data_root ./OmniHalluBench/ \
    --output_dir "$OUT" \
    --file_name "$FILE" \
    --time_path "$OUT/time" \
    --prompt_version "$PV" \
    --max_new_tokens 256 \
    $EXTRA \
    >> "$LOG" 2>&1
RC=$?
echo "[cell $(date +%T)] runner exit=$RC" | tee -a "$LOG"
if [ $RC -ne 0 ]; then echo "[cell] FAILED $CELL rc=$RC" | tee -a "$LOG"; exit $RC; fi

# merge -> _full.jsonl (single process, so just copy/rename the produced jsonl)
RAW="$OUT/${FILE}.jsonl"
FULL="$OUT/${CELL}_full.jsonl"
if [ ! -f "$RAW" ]; then echo "[cell] no raw output $RAW" | tee -a "$LOG"; exit 3; fi
cp -f "$RAW" "$FULL"
NROWS=$(wc -l < "$FULL")
echo "[cell $(date +%T)] merged $NROWS rows -> $FULL" | tee -a "$LOG"

# score -> _full_single_scored.jsonl
# single_eval appends an aggregate CSV to its CWD; run from the cell dir so the
# shared repo-root CSVs (judgement_*.csv / *.csv) are not polluted.
REPO="$(pwd)"
( cd "$OUT" && $PY "$REPO/OmniConfess/eval/single_eval.py" --target_files "$REPO/$FULL" ) >> "$LOG" 2>&1
SC=$?
SCORED="$OUT/${CELL}_full_single_scored.jsonl"
if [ $SC -ne 0 ] || [ ! -f "$SCORED" ]; then echo "[cell] SCORING FAILED $CELL sc=$SC" | tee -a "$LOG"; exit 4; fi
SN=$(wc -l < "$SCORED")
echo "[cell $(date +%T)] scored $SN rows -> $SCORED" | tee -a "$LOG"

echo "DONE ${METHOD} $(date '+%Y-%m-%d %H:%M:%S')" > "$OUT/cell_done.txt"
echo "[cell $(date +%T)] STAMPED cell_done.txt for $CELL" | tee -a "$LOG"
