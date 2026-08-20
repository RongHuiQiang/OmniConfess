#!/bin/bash
# run_base_queue.sh <gpu> -- serially drain a queue of base cells on ONE card.
# Reads cell specs (one per line) from the file given as $2. Each line:
#   <dataset> <method> <data_json> <prompt_version> <max_samples(0=full)>
# Skips a cell if its cell_done.txt already exists. Single-card single-task:
# only one cell runs at a time on this card.
set -u
cd "$(dirname "$0")"
GPU=$1; QFILE=$2
QLOG="outputs/base_queue_gpu${GPU}.log"
echo "[queue $(date +%T)] start gpu=$GPU qfile=$QFILE" | tee -a "$QLOG"
while IFS=' ' read -r DS METHOD DATA PV MAXS; do
  [ -z "${DS:-}" ] && continue
  case "$DS" in \#*) continue;; esac
  CELL="${DS}_base_${METHOD}"
  if [ -f "outputs/$CELL/cell_done.txt" ]; then
    echo "[queue $(date +%T)] skip done $CELL" | tee -a "$QLOG"; continue
  fi
  echo "[queue $(date +%T)] >>> $CELL on gpu=$GPU (maxs=$MAXS)" | tee -a "$QLOG"
  bash run_base_cell.sh "$GPU" "$DS" "$METHOD" "$DATA" "$PV" "$MAXS" >> "$QLOG" 2>&1
  rc=$?
  echo "[queue $(date +%T)] <<< $CELL rc=$rc" | tee -a "$QLOG"
done < "$QFILE"
echo "[queue $(date +%T)] DONE gpu=$GPU" | tee -a "$QLOG"
