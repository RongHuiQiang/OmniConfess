#!/bin/bash
# run_p5b_qwen3_gpu7.sh -- P5b blindspot cross-backbone serial queue, GPU7 ONLY.
# One dataset at a time, one thinker process per card (~62GB, exclusive).
set -u
cd /home/tsq/rhq/omniconfess/OmniConfess
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/ultrafast_share/tsq/qwen3omni_outputs/tmp
mkdir -p "$TMPDIR"
source /home/tsq/rhq/omniconfess/env_ffmpeg.sh 2>/dev/null || true
LOG=outputs/cf_probe/p5b_qwen3
mkdir -p "$LOG"

guard () {  # abort if GPU7 has a foreign allocation
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)
  if [ "$used" -gt 2000 ]; then
    echo "[guard $(date +%T)] ABORT: GPU7 has ${used}MiB foreign alloc" | tee -a "$LOG/queue.log"
    exit 9
  fi
}

run_ds () {  # $1 dataset  $2 expected_n
  local DS=$1 N=$2 OUT="$LOG/${1}_qwen3_cf.jsonl"
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$N" ]; then
    echo "[queue $(date +%T)] $DS already complete"; return 0
  fi
  guard
  echo "[queue $(date +%T)] START $DS (expect $N)" | tee -a "$LOG/queue.log"
  CUDA_VISIBLE_DEVICES=7 $PY probe_cf_qwen3_p5b.py --dataset "$DS" \
      >> "$LOG/${DS}.log" 2>&1
  RC=$?
  echo "[queue $(date +%T)] $DS exit=$RC rows=$(wc -l < "$OUT" 2>/dev/null || echo 0)" | tee -a "$LOG/queue.log"
}

run_ds ragtruth  435
run_ds pubmedqa  468
run_ds haloquest 365
run_ds avhbench  250
echo "[queue $(date +%T)] ALL DONE" | tee -a "$LOG/queue.log"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 7 | tee -a "$LOG/queue.log"
