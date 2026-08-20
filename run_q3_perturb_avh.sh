#!/bin/bash
# Vote-SC cross-generation pool extension (EXPERIMENT_DESIGN.md avhbench §6):
# Qwen3-Omni-native input-perturbation arms on avhbench 645, v1 prompt, greedy.
#
#   q3_res384 : OMNI_FRAME_SIZE=384  OMNI_NUM_FRAMES=8   (task's "OMNI_FRAME_SIZE=384")
#   q3_f16    : OMNI_FRAME_SIZE=224  OMNI_NUM_FRAMES=16  (task's "OMINI_NUM_FRAMES=16"
#               is the typo'd spelling; data_loader.py reads OMINI_NUM_FRAMES ->
#               the real knob is OMNI_NUM_FRAMES)
#
# Mirrors the shipped q3 bare-arm protocol: same 8-way GPFS shards, eos-token-id
# fix, thinker-only 30B on one card per process (8 cards), TMPDIR on GPFS
# (root fs still 94% full). Differences from run_qwen3omni_8gpu.sh are only the
# runner (variant runner records model/input_variant/frame_size/num_frames),
# the output dir (outputs/avhbench_vote_sc/<variant>/, alongside the other vote
# pool arm dirs) and the two env knobs.
#
# Usage: bash run_q3_perturb_avh.sh q3_res384|q3_f16 [--dry_run]
set -u
VARIANT=${1:?q3_res384 | q3_f16}
EXTRA="${*:2}"

case "$VARIANT" in
  q3_res384) export OMNI_FRAME_SIZE=384; export OMNI_NUM_FRAMES=8;  TAG=res384 ;;
  q3_f16)    export OMNI_FRAME_SIZE=224; export OMNI_NUM_FRAMES=16; TAG=frames16 ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac

cd /home/tsq/rhq/omniconfess/OmniConfess
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/ultrafast_share/tsq/qwen3omni_outputs/tmp
mkdir -p "$TMPDIR"
source /home/tsq/rhq/omniconfess/env_ffmpeg.sh

NGPU=8
SHARD_DIR=/ultrafast_share/tsq/qwen3omni_outputs/shards/avhbench   # existing 8-way split
DATA_ROOT=/home/tsq/rhq/omniconfess/OmniConfess/OmniHalluBench/avhbench/
RUNNER=/home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/methods/shared/base_model_qwen3_omni_var.py
OUT=outputs/avhbench_vote_sc/$VARIANT
FNAME=avhbench_${VARIANT}
mkdir -p "$OUT" "$OUT/time"

for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i $PY "$RUNNER" \
    --datasets avhbench \
    --data_path "$SHARD_DIR/avhbench_shard${i}of${NGPU}.json" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUT" \
    --time_path "$OUT/time" \
    --file_name "${FNAME}_shard${i}" \
    --model_tag Qwen3-Omni-30B-A3B-Instruct \
    --variant "$TAG" \
    $EXTRA \
    > "$OUT/shard${i}.log" 2>&1 &
done
wait
cat "$OUT"/${FNAME}_shard*.jsonl > "$OUT/${FNAME}_full.jsonl" 2>/dev/null
echo "[q3 perturb x $VARIANT] done -> $OUT/${FNAME}_full.jsonl ($(wc -l < "$OUT/${FNAME}_full.jsonl") lines)"
