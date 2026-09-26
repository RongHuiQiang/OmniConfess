#!/bin/bash
# run_cf_gate_op1.sh -- OP1 VAS gate driver, dynamic-card edition.
# Polls cards 0-3; the moment any card has memory.used < 2000 MiB (its grid
# cell just ended) it writes that card to /tmp/grid_card_reserve (so the grid
# daemon does not grab it for the next cell) and runs smoke -> full pipeline
# (phd then cmm: calib/thresh/gated/report) on that card. Reserve is removed
# when the run ends. Cards 4-7 never touched; grid processes never killed.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
LOG=outputs/cf_gate
mkdir -p "$LOG"

GPU="${GPU:-}"
while [ -z "$GPU" ]; do
  for c in 0 1 2 3; do
    # card 1 is OUR pre-existing reservation (made while waiting); other
    # reserved cards belong to other agents and stay untouchable.
    if grep -qw "$c" /tmp/grid_card_reserve 2>/dev/null && [ "$c" != "1" ]; then continue; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$c")
    if [ "$used" -lt 2000 ]; then GPU=$c; break; fi
  done
  [ -z "$GPU" ] && sleep 60
done
echo "$GPU" >> /tmp/grid_card_reserve
trap 'grep -vx "$GPU" /tmp/grid_card_reserve > /tmp/gcr.tmp 2>/dev/null && mv /tmp/gcr.tmp /tmp/grid_card_reserve' EXIT
echo "[driver] claiming GPU $GPU at $(date), reserve=$(cat /tmp/grid_card_reserve)"

for ds in phd cmm; do
  echo "[driver] === smoke $ds ($(date))"
  CUDA_VISIBLE_DEVICES=$GPU $PY cf_confess_gated.py --datasets $ds --stage calib \
      --n_sample 2 --out_dir $LOG/smoke > $LOG/smoke_${ds}_calib.log 2>&1 \
      || { echo "[driver] smoke calib $ds FAILED"; continue; }
  $PY cf_confess_gated.py --datasets $ds --stage thresh --n_sample 2 \
      --out_dir $LOG/smoke > $LOG/smoke_${ds}_thresh.log 2>&1 || true
  CUDA_VISIBLE_DEVICES=$GPU $PY cf_confess_gated.py --datasets $ds --stage gated \
      --n_sample 2 --out_dir $LOG/smoke > $LOG/smoke_${ds}_gated.log 2>&1 \
      || { echo "[driver] smoke gated $ds FAILED"; continue; }
  echo "[driver] === smoke $ds OK, full run ($(date))"
  CUDA_VISIBLE_DEVICES=$GPU $PY cf_confess_gated.py --datasets $ds --stage calib \
      > $LOG/${ds}_calib.log 2>&1 || { echo "[driver] calib $ds FAILED"; continue; }
  $PY cf_confess_gated.py --datasets $ds --stage thresh \
      > $LOG/${ds}_thresh.log 2>&1 || echo "[driver] thresh $ds coverage assert (see log)"
  CUDA_VISIBLE_DEVICES=$GPU $PY cf_confess_gated.py --datasets $ds --stage gated \
      > $LOG/${ds}_gated.log 2>&1 || { echo "[driver] gated $ds FAILED"; continue; }
  $PY cf_confess_gated.py --datasets $ds --stage report \
      > $LOG/${ds}_report.log 2>&1
done
echo "[driver] all done $(date)"
