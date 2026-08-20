#!/bin/bash
# CF-Confess on PhD full set (906), 8 shards x 1 GPU each.
# Usage: bash run_cf_confess_8gpu.sh            # runs main lam arm + random-sign control + lam1.0 arm
#        bash run_cf_confess_8gpu.sh lam         # only the main arm
set -u
cd "$(dirname "$0")"

DATASET=phd
NGPU=8
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
DATA=OmniHalluBench/phd/phd.json
OUT=outputs/phd_cf_confess
mkdir -p "$OUT" "$OUT/time"

# 1) shards (round-robin, deterministic; reuses existing phd_shardKof8.json if present)
$PY ../shard_data.py "$DATA" $NGPU >/dev/null

run_arm () {  # $1 = file prefix, $2... = extra args
  local PREFIX=$1; shift
  echo "[arm] $PREFIX $*"
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i $PY cf_confess_phd.py \
        --datasets "$DATASET" \
        --data_path "OmniHalluBench/phd/phd_shard${i}of${NGPU}.json" \
        --output_dir "$OUT" \
        --time_path "$OUT/time" \
        --file_name "${PREFIX}_shard${i}" \
        "$@" \
        > "$OUT/${PREFIX}_shard${i}.log" 2>&1 &
  done
  wait
  echo "[arm] $PREFIX done"
}

MODE=${1:-all}
if [ "$MODE" = "all" ] || [ "$MODE" = "lam" ]; then
  run_arm cf_lam2.0   --lambda_cf 2.0
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "randctl" ]; then
  run_arm cf_lam2.0_randctl --lambda_cf 2.0 --control random_sign
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "lam1" ]; then
  run_arm cf_lam1.0   --lambda_cf 1.0
fi

echo "all arms done -> $OUT"
