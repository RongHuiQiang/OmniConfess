#!/bin/bash
# Sequential orchestrator for the [机制迁移@Qwen3] descriptive queue:
#   1. ragtruth  x contract v4 (435)
#   2. haloquest x contract v2 (365)   [single layer; gate/BoN 待后续]
#   3. avhbench  x contract v4 (645)   [first contract iteration ever]
# Waits for free cards (co-agent GPU contention), retries on collision.
set -u
cd "$(dirname "$0")"

want () {  # $1 dataset, $2 expected_n -> echo 1 if run complete
  local f="outputs/${1}_qwen3omni_contract*/${1}_qwen3omni_*_full.jsonl"
  local best=0 f1 n
  for fp in $f; do
    [ -f "$fp" ] || continue
    n=$(wc -l < "$fp")
    if [ "$n" -ge "$2" ]; then best=1; fi
  done
  echo $best
}

run_one () {  # $1 dataset $2 tag $3 pv $4 expected_n
  local DS=$1 TAG=$2 PV=$3 N=$4 tries=0
  while [ "$(want "$DS" "$N")" = "0" ] && [ $tries -lt 2000 ]; do
    tries=$((tries+1))
    echo "[queue $(date +%H:%M:%S)] $DS/$TAG attempt $tries"
    bash run_qwen3omni_contract.sh "$DS" "$TAG" "$PV" auto
    sleep 60   # patient retry: loop resumes, runner skips done ids
  done
  echo "[queue $(date +%H:%M:%S)] $DS/$TAG complete=$(want "$DS" "$N")"
}

run_one ragtruth  contractv4 v4 435
run_one haloquest contractv2 v2 365
run_one avhbench  contractv4 v4 645
echo "[queue] ALL DONE $(date +%H:%M:%S)"
