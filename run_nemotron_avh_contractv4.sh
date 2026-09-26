#!/bin/bash
# run_nemotron_avh_contractv4.sh -- [描述族迁移臂最后一格]
#   AVHBench x Nemotron-3-Nano-Omni x base_model + 输出契约 v4, 全量 645.
#   Direct sibling of ragtruth_nemotron_contractv4 (same runner
#   base_model_nemotron_hf_pv.py, same --prompt_version v4 flag, same greedy /
#   enable_thinking=False / record_process default-on); env exports mirror
#   run_nemotron_grid_cell.sh (slow_share caches after the 0829 ultrafast wipe).
#   ONE delta vs the recent broken avhbench nemotron cells: PATH prepends
#   /slow_share/tsq/bin so the runner's ffmpeg audio fallback resolves a LIVE
#   static ffmpeg (repo bin/ffmpeg symlink is dead -> ultrafast wiped) and the
#   mp4 audio track (16 kHz) really enters, matching the 0816 base arm's
#   645/645 non-empty audio evidence (RESULTS.md line 82).
#
# Usage: run_nemotron_avh_contractv4.sh <gpu> smoke|full|score
set -u
cd "$(dirname "$0")"

PY=/slow_share/tsq/venvs/nemotron-hf/bin/python
EVALPY=/slow_share/tsq/venvs/omniconfess/bin/python
export TMPDIR=/slow_share/tsq/tsq/tmp
export PYTHONPATH=/home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/methods:${PYTHONPATH:-}
export HF_HOME=/slow_share/tsq/tsq_data/.cache/huggingface
export TRITON_CACHE_DIR=/slow_share/tsq/tsq_data/.cache/triton
export TORCH_HOME=/slow_share/tsq/tsq_data/.cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export NEMO_REPO=/home/tsq/rhq/omniconfess/OmniConfess
export NEMO_KEEP_CACHE=1
export MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS
export PATH=/slow_share/tsq/bin:$PATH     # live ffmpeg 7.0.2 for the mp4->wav audio fallback
mkdir -p "$TMPDIR"

GPU=$1; MODE=$2
DS=avhbench; TAG=contractv4; PV=v4; EXP=645

if [ "$MODE" = "smoke" ]; then
  OUT=outputs/avhbench_nemotron_contractv4_smoke
  FILE=avhbench_nemotron_contractv4_smoke
  mkdir -p "$OUT" "$OUT/time"
  rm -f "$OUT/${FILE}.jsonl"
  echo "[smoke $(date +%T)] gpu=$GPU pv=$PV (md5 6a2234810ed7e51965713c1a0220ccfb)"
  CUDA_VISIBLE_DEVICES=$GPU $PY OmniConfess/methods/base_model_nemotron_hf_pv.py \
      --datasets "$DS" \
      --data_path "$PWD/OmniHalluBench/${DS}/${DS}.json" \
      --data_root "$PWD/OmniHalluBench/${DS}/" \
      --output_dir "$OUT" \
      --time_path "$OUT/time" \
      --file_name "$FILE" \
      --prompt_version "$PV" \
      --dry_run 2>&1 | tee "$OUT/run.log"
  echo "[smoke $(date +%T)] rows=$(wc -l < "$OUT/${FILE}.jsonl")"
  exit $?
fi

OUT=outputs/${DS}_nemotron_${TAG}
FILE="${DS}_nemotron_${TAG}"
RAW="$OUT/${FILE}.jsonl"
SCORED="$OUT/${FILE}_single_scored.jsonl"
LOG="$OUT/run.log"
mkdir -p "$OUT" "$OUT/time"

if [ "$MODE" = "full" ]; then
  N0=$( [ -f "$RAW" ] && wc -l < "$RAW" || echo 0 )
  echo "[cell $(date +%T)] ${DS}_nemotron_${TAG} gpu=$GPU exp=$EXP have=$N0" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PY OmniConfess/methods/base_model_nemotron_hf_pv.py \
      --datasets "$DS" \
      --data_path "$PWD/OmniHalluBench/${DS}/${DS}.json" \
      --data_root "$PWD/OmniHalluBench/${DS}/" \
      --output_dir "$OUT" \
      --time_path "$OUT/time" \
      --file_name "$FILE" \
      --prompt_version "$PV" \
      >> "$LOG" 2>&1
  RC=$?
  N1=$( [ -f "$RAW" ] && wc -l < "$RAW" || echo 0 )
  echo "[cell $(date +%T)] runner exit=$RC rows=$N1/$EXP" | tee -a "$LOG"
  [ "$N1" -ge "$EXP" ] || { echo "[cell] SHORT rows=$N1/$EXP" | tee -a "$LOG"; exit 6; }
  echo "DONE contractv4 $(date '+%Y-%m-%d %H:%M:%S') rows=$N1" > "$OUT/cell_done.txt"
  exit 0
fi

if [ "$MODE" = "score" ]; then
  REPO="$(pwd)"
  unset ZHIPU_API_KEY
  ( cd "$OUT" && $EVALPY "$REPO/OmniConfess/eval/single_eval.py" \
      --target_files "$REPO/$RAW" ) 2>&1 | tee "$OUT/score.log"
  [ -f "$SCORED" ] && echo "[score $(date +%T)] scored rows=$(wc -l < "$SCORED")"
  exit 0
fi

echo "unknown mode $MODE"; exit 2
