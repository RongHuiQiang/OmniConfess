#!/bin/bash
# CF-Confess on PubMedQA full set (468), 8 shards x 1 GPU each.
# Usage: bash run_cf_confess_pubmedqa.sh            # main arm + random-sign control
#        bash run_cf_confess_pubmedqa.sh lam        # only the main arm
#        bash run_cf_confess_pubmedqa.sh randctl    # only the random-sign control
set -u
cd "$(dirname "$0")"

DATASET=pubmedqa
NGPU=8
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
DATA=OmniHalluBench/pubmedqa/pubmedqa.json
OUT=outputs/pubmedqa_cf_confess
LAM=${LAM:-2.0}
mkdir -p "$OUT" "$OUT/time"

$PY ../shard_data.py "$DATA" $NGPU >/dev/null

run_arm () {  # $1 = file prefix, $2... = extra args
  local PREFIX=$1; shift
  echo "[arm] $PREFIX $*"
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i $PY cf_confess_pubmedqa.py \
        --datasets "$DATASET" \
        --data_path "OmniHalluBench/pubmedqa/pubmedqa_shard${i}of${NGPU}.json" \
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
  run_arm cf_lam${LAM}   --lambda_cf "$LAM"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "randctl" ]; then
  run_arm cf_lam${LAM}_randctl --lambda_cf "$LAM" --control random_sign
fi

echo "all arms done -> $OUT"
