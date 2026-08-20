#!/bin/bash
# PubMedQA CF-Confess + matched-prompt CAD arms, GPUs 0-3 (2 shards per GPU,
# sharing with the baseline queue: ~22GB x3 procs on an 80GB card).
# lam*=0.25 chosen on the dev probe by the pre-registered phd rule
# (max dev Youden under yes-rate move <= +-5pp).
set -u
cd "$(dirname "$0")"
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
NGPU=4
LAM=0.25
OUT=outputs/pubmedqa_cf_confess
CADOUT=outputs/pubmedqa_base_cad_v1
mkdir -p "$OUT" "$OUT/time" "$CADOUT" "$CADOUT/time"

run_sharded () {  # $1=script, $2=prefix, $3=outdir, rest=args
  local SCRIPT=$1 PREFIX=$2 OUTDIR=$3; shift 3
  echo "[arm] $PREFIX $*"
  for i in $(seq 0 7); do
    g=$((i % NGPU))
    CUDA_VISIBLE_DEVICES=$g $PY "$SCRIPT" \
        --datasets pubmedqa \
        --data_path "OmniHalluBench/pubmedqa/pubmedqa_shard${i}of8.json" \
        --output_dir "$OUTDIR" \
        --time_path "$OUTDIR/time" \
        --file_name "${PREFIX}_shard${i}" \
        "$@" \
        > "$OUTDIR/${PREFIX}_shard${i}.log" 2>&1 &
    if [ $(( (i+1) % NGPU )) -eq 0 ]; then wait; fi
  done
  wait
  echo "[arm] $PREFIX done"
}

run_sharded cf_confess_pubmedqa.py cf_lam${LAM} "$OUT" --lambda_cf "$LAM"
run_sharded cf_confess_pubmedqa.py cf_lam${LAM}_randctl "$OUT" --lambda_cf "$LAM" --control random_sign
# matched-prompt (v1) CAD control arm -- unmodified baseline implementation
run_sharded OmniConfess/methods/base_cad.py cad_v1 "$CADOUT" --prompt_version v1

echo "pubmedqa arms done"
