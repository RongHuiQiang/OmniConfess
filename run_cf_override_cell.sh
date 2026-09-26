#!/bin/bash
# run_cf_override_cell.sh -- CF_OVERRIDE (PI 修改案 2026-08-31): run ONE
# withheld CF-Confess grid cell on ONE card: smoke -> full -> score -> stamp.
# Mirrors run_qwen3_grid_cell.sh / run_nemotron_grid_cell.sh (same env exports,
# same scoring tail, same cell_done stamp). Only cards 0-3.
#
# Usage: run_cf_override_cell.sh <gpu> <arm>
#   arm in: qwen3_phd   (PHD x Qwen3-Omni  x CF lam2.0, 906)
#           nemo_phd    (PHD x Nemotron    x CF lam2.0, 906)
#           nemo_cmm    (CMM x Nemotron    x CF lam1.0, 721)
# Env: SMOKE=1 -> 8-sample smoke (+ lam50 mechanics check) only, no full run.
set -u
cd "$(dirname "$0")"

GPU=$1
ARM=$2
SMOKE=${SMOKE:-0}

case "$ARM" in
  qwen3_phd) DS=phd; BACKBONE=qwen3;  LAM=2.0; EXP=906; CELL=phd_qwen3_cf ;;
  nemo_phd)  DS=phd; BACKBONE=nemotron; LAM=2.0; EXP=906; CELL=phd_nemotron_cf ;;
  nemo_cmm)  DS=cmm; BACKBONE=nemotron; LAM=1.0; EXP=721; CELL=cmm_nemotron_cf ;;
  *) echo "[cell] unknown arm $ARM"; exit 2 ;;
esac

OUT="outputs/${CELL}"
FILE="${CELL}_full"
RAW="$OUT/${FILE}.jsonl"
SCORED="$OUT/${FILE}_single_scored.jsonl"
LOG="$OUT/run.log"
DATA="$PWD/OmniHalluBench/${DS}/${DS}.json"
mkdir -p "$OUT" "$OUT/time"

if [ "$BACKBONE" = "qwen3" ]; then
  PY=/slow_share/tsq/venvs/omniconfess/bin/python
  export MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS
  export HF_HOME=/slow_share/tsq/tsq_data/.cache/huggingface
  RUNNER="cf_confess_qwen3.py"
else
  PY=/slow_share/tsq/venvs/nemotron-hf/bin/python
  EVALPY=/slow_share/tsq/venvs/omniconfess/bin/python
  export TMPDIR=/slow_share/tsq/tsq/tmp
  export PYTHONPATH=/home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/methods:${PYTHONPATH:-}
  export HF_HOME=/slow_share/tsq/tsq_data/.cache/huggingface
  # TRITON_CACHE_DIR fix (2026-08-30 emptyrun root cause; see
  # run_nemotron_grid_cell.sh): Mamba2 kernels launch triton on EVERY forward.
  export TRITON_CACHE_DIR=/slow_share/tsq/tsq_data/.cache/triton
  export TORCH_HOME=/slow_share/tsq/tsq_data/.cache/torch
  export NEMO_REPO=/home/tsq/rhq/omniconfess/OmniConfess
  export NEMO_KEEP_CACHE=1
  export MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS
  mkdir -p "$TMPDIR"
  RUNNER="cf_confess_nemotron_hf.py"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

run_batch () {  # $1 outdir, $2 filename, rest extra runner args
  local OD=$1; local FN=$2; shift 2
  mkdir -p "$OD" "$OD/time"
  if [ "$BACKBONE" = "qwen3" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY "$RUNNER" \
        --datasets "$DS" --data_path "$DATA" \
        --output_dir "$OD" --time_path "$OD/time" --file_name "$FN" \
        --prompt_version v1 "$@"
  else
    CUDA_VISIBLE_DEVICES=$GPU $PY "$RUNNER" \
        --datasets "$DS" --data_path "$DATA" \
        --output_dir "$OD" --time_path "$OD/time" --file_name "$FN" \
        --prompt_version v1 "$@"
  fi
}

if [ ! -f "$RUNNER" ]; then echo "[cell] NO RUNNER $RUNNER" | tee -a "$LOG"; exit 5; fi

if [ "$SMOKE" = "1" ]; then
  mkdir -p "$OUT/smoke" "$OUT/smoke/time"
  echo "[smoke $(date +%T)] $CELL gpu=$GPU lam=$LAM" | tee -a "$LOG"
  run_batch "$OUT/smoke" "smoke_lam${LAM}" --limit 8 --lambda_cf "$LAM" \
      >> "$OUT/smoke/smoke.log" 2>&1
  RC=$?
  echo "[smoke $(date +%T)] real-lam exit=$RC" | tee -a "$LOG"
  # mechanics check: huge lam must trigger the forced-token regen path on any
  # sample with delta != 0 (trigger rate nonzero proof; smoke dir only)
  run_batch "$OUT/smoke" "smoke_lam50_mechanics" --limit 8 --lambda_cf 50 \
      >> "$OUT/smoke/smoke.log" 2>&1
  RC2=$?
  echo "[smoke $(date +%T)] lam50-mechanics exit=$RC2" | tee -a "$LOG"
  exit $(( RC + RC2 ))
fi

echo "[cell $(date +%T)] $CELL gpu=$GPU lam=$LAM exp=$EXP" | tee -a "$LOG"
run_batch "$OUT" "$FILE" --lambda_cf "$LAM" >> "$LOG" 2>&1
RC=$?
echo "[cell $(date +%T)] runner exit=$RC" | tee -a "$LOG"
if [ $RC -ne 0 ]; then echo "[cell] FAILED $CELL rc=$RC" | tee -a "$LOG"; exit $RC; fi
if [ ! -f "$RAW" ]; then echo "[cell] no raw $RAW" | tee -a "$LOG"; exit 3; fi
NROWS=$(wc -l < "$RAW")
if [ "$NROWS" -lt "$EXP" ]; then
  echo "[cell] SHORT $CELL rows=$NROWS/$EXP -- not scoring, will retry" | tee -a "$LOG"; exit 6
fi

# score (from cell dir so shared CSVs aren't polluted; zhipu unset -> GAVIE frozen)
REPO="$(pwd)"
EVAL_PY=${EVALPY:-$PY}
unset ZHIPU_API_KEY
( cd "$OUT" && $EVAL_PY "$REPO/OmniConfess/eval/single_eval.py" --target_files "$REPO/$RAW" ) >> "$LOG" 2>&1
SC=$?
if [ $SC -ne 0 ] || [ ! -f "$SCORED" ]; then echo "[cell] SCORING FAILED $CELL sc=$SC" | tee -a "$LOG"; exit 4; fi
SN=$(wc -l < "$SCORED")
echo "DONE cf_lam${LAM} $(date '+%Y-%m-%d %H:%M:%S') rows=$SN" > "$OUT/cell_done.txt"
echo "[cell $(date +%T)] STAMPED cell_done.txt for $CELL rows=$SN" | tee -a "$LOG"
