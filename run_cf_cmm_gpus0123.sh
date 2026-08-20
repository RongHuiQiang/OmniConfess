#!/bin/bash
# CMM CF-Confess arms (lambda 1.0 main + 0.5 conservative + random-sign control)
# + matched-prompt CAD v1 arm. GPUs 0-3, 2 shards per GPU.
# lambda*=1.0: max dev Youden (0.471) among the pre-registered grid; the +-5pp
# yes-rate constraint is infeasible on cmm (base yes-rate is +30.5pp away from
# the 41.7% truth rate) and is waived + documented in EXPERIMENT_DESIGN.md.
# Full-population offline probe predicts F1 76.36 / Youden 0.600 at lam=1.0.
set -u
cd "$(dirname "$0")"
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
NGPU=4
OUT=outputs/cmm_cf_confess
CADOUT=outputs/cmm_base_cad_v1
mkdir -p "$OUT" "$OUT/time" "$CADOUT" "$CADOUT/time"

run_sharded () {  # $1=script, $2=prefix, $3=outdir, rest=args
  local SCRIPT=$1 PREFIX=$2 OUTDIR=$3; shift 3
  echo "[arm] $PREFIX $*"
  for i in $(seq 0 7); do
    g=$((i % NGPU))
    CUDA_VISIBLE_DEVICES=$g $PY "$SCRIPT" \
        --datasets cmm \
        --data_path "OmniHalluBench/cmm/cmm_shard${i}of8.json" \
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

run_sharded cf_confess_cmm.py cf_lam1.0 "$OUT" --lambda_cf 1.0
run_sharded cf_confess_cmm.py cf_lam1.0_randctl "$OUT" --lambda_cf 1.0 --control random_sign
run_sharded cf_confess_cmm.py cf_lam0.5 "$OUT" --lambda_cf 0.5
# matched-prompt (v1) CAD control arm -- unmodified baseline implementation
run_sharded OmniConfess/methods/base_cad.py cad_v1 "$CADOUT" --prompt_version v1

echo "cmm arms done"
