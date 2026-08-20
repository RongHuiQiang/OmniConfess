#!/bin/bash
# CF-Confess on PhD full set (906) on Qwen3-Omni-30B thinker.
# One process per card (thinker ~60.5GB). Uses GPUs 0-6 (7 shards), leaves
# card 7 alone. Arms: lam* (dev-selected) then random-sign control.
# Usage: bash run_cf_confess_qwen3_7gpu.sh          # both arms
#        bash run_cf_confess_qwen3_7gpu.sh lam 4.0  # main arm only
set -u
cd "$(dirname "$0")"

PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
NGPU=7
OUT=outputs/phd_cf_confess_qwen3
mkdir -p "$OUT" "$OUT/time"

$PY ../shard_data.py OmniHalluBench/phd/phd.json $NGPU >/dev/null   # phd_shardKof7.json

check_free () {  # refuse to start on a card with foreign allocation
  for i in $(seq 0 $((NGPU-1))); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
    if [ "$used" -gt 15000 ]; then echo "[abort] GPU $i has ${used}MiB foreign alloc"; exit 1; fi
  done
}

run_arm () {  # $1 prefix, $2 lam, rest extra args
  local PREFIX=$1; local LAM=$2; shift 2
  echo "[arm] $PREFIX lam=$LAM $*"
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i $PY cf_confess_qwen3.py \
        --datasets phd \
        --data_path "OmniHalluBench/phd/phd_shard${i}of${NGPU}.json" \
        --output_dir "$OUT" \
        --time_path "$OUT/time" \
        --file_name "${PREFIX}_shard${i}" \
        --lambda_cf "$LAM" \
        "$@" \
        > "$OUT/${PREFIX}_shard${i}.log" 2>&1 &
  done
  wait
  echo "[arm] $PREFIX done"
}

MODE=${1:-all}
LAM=4.0
if [ "$MODE" = "all" ] || [ "$MODE" = "lam" ]; then
  check_free
  run_arm "cf_qwen3_lam${LAM}" "$LAM"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "randctl" ]; then
  check_free
  run_arm "cf_qwen3_lam${LAM}_randctl" "$LAM" --control random_sign
fi
echo "all arms done -> $OUT"
