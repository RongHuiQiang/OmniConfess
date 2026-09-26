#!/bin/bash
# run_dc_shard.sh CARD START N SHARD NSHARD OUT CRASHFILE
# Restarts on exit-42 (poison CUDA context), accumulating skip ids from CRASHFILE.
set -u
CARD=$1; START=$2; N=$3; SHARD=$4; NSHARD=$5; OUT=$6; CRASH=$7
PY=/home/tsq/.local/bin/python3.13
cd /home/tsq/rhq/omniconfess/OmniConfess
: > "$CRASH.done" 2>/dev/null || true
for attempt in $(seq 1 15); do
  SKIP=""
  [ -f "$CRASH" ] && SKIP=$(paste -sd, "$CRASH")
  echo "[runner card$CARD] attempt=$attempt skip=[$SKIP]"
  CUDA_VISIBLE_DEVICES=$CARD MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/dev/shm/site-packages LD_LIBRARY_PATH=/dev/shm/so_mirror \
    $PY scripts/avh_confess_dualchannel.py \
      --start $START --n $N --shard $SHARD --nshard $NSHARD \
      --skip-ids "$SKIP" --crash-file "$CRASH" --out "$OUT"
  rc=$?
  echo "[runner card$CARD] rc=$rc"
  if [ $rc -eq 42 ]; then
    echo "[runner card$CARD] poison restart"; sleep 5; continue
  fi
  break
done
echo "[runner card$CARD] FINISHED rc=$rc" | tee "$CRASH.done"
