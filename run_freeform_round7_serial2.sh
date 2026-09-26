#!/bin/bash
# run_freeform_round7_serial2.sh — Round 7 serial rerun #2
#
# Root cause of previous failures: PYTHONPATH had torch 2.14.0+cu130 which
# requires CUDA driver ≥ 13.x, but the system only has 12.4 (12040).
# CUDA init silently fails → model loads to CPU RAM → generate() stalls.
#
# Fix: prepend /tmp/torch-cu121/lib/python (torch 2.5.1+cu121) to PYTHONPATH
# and point LD_LIBRARY_PATH to its torch/lib directory.
#
# Arms to run: TAU10, TAU20, MAX1-TAU10, TOPK (COV20/40/60/MAX1-COV60 done)
#
# Usage: nohup bash run_freeform_round7_serial2.sh > outputs/freeform_grid/round7/serial2.log 2>&1 &

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python3
MODEL_DIR=/dev/shm
OUTDIR="$REPO/outputs/freeform_grid/round7"
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

LIMIT=50
for arg in "$@"; do
  case $arg in
    --limit=*) LIMIT="${arg#*=}" ;;
    --limit)   shift; LIMIT="$1"; shift ;;
  esac
done

# torch 2.5.1+cu121 must come first; the rest of pip-target provides everything else
TORCH_CU121=/tmp/torch-cu121/lib/python
PIP_TARGET=/tmp/pip-target-b9a06nc9/lib/python

echo "[round7-serial2] limit=$LIMIT  model=$MODEL_DIR/Qwen2.5-Omni-7B"
echo "[round7-serial2] torch path: $TORCH_CU121"
echo "[round7-serial2] pip-target: $PIP_TARGET"

# Verify CUDA is usable before spending time on model loads
echo "[round7-serial2] verifying CUDA..."
CUDA_TEST=$(
  PYTHONPATH="$TORCH_CU121:$PIP_TARGET" \
  LD_LIBRARY_PATH=/dev/shm/so_mirror:"$TORCH_CU121/torch/lib" \
    python3 -c "
import torch
ok = torch.cuda.is_available()
print(f'cuda={ok} torch={torch.__version__}')
if ok:
    print('device:', torch.cuda.get_device_name(0))
" 2>&1)
echo "[round7-serial2] $CUDA_TEST"
if ! echo "$CUDA_TEST" | grep -q "cuda=True"; then
  echo "[round7-serial2] CUDA not available — aborting"
  exit 1
fi

# Format: "GPU  TAG  TAU  MAX_SPANS  COV_CAP  SELECT_MODE"
declare -a EXPS=(
  "2  R7-TAU10       1.0  2  0     rho"
  "3  R7-TAU20       2.0  2  0     rho"
  "0  R7-MAX1-TAU10  1.0  1  0     rho"
  "1  R7-TOPK        0    2  0     topk"
)

FAIL=0
for spec in "${EXPS[@]}"; do
  read -r GPU TAG TAU MAX_SPANS COV_CAP SELECT_MODE <<< "$spec"
  OUTFILE="$OUTDIR/haloquest_${TAG}.jsonl"
  LOGFILE="$LOGDIR/haloquest_${TAG}.log"

  # Skip arms that already have complete output
  if [ -f "$OUTFILE" ]; then
    n=$(wc -l < "$OUTFILE")
    if [ "$n" -ge "$LIMIT" ]; then
      echo "[skip]   $TAG already has $n rows"
      continue
    fi
    echo "[resume] $TAG has $n rows, re-running"
    rm -f "$OUTFILE"
  fi

  echo "[launch] GPU=$GPU  $TAG  tau=$TAU  max_spans=$MAX_SPANS  cov_cap=$COV_CAP  select=$SELECT_MODE"

  CUDA_VISIBLE_DEVICES=$GPU \
  MMHALU_MODEL_ROOT=$MODEL_DIR \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$TORCH_CU121:$PIP_TARGET" \
  LD_LIBRARY_PATH=/dev/shm/so_mirror:"$TORCH_CU121/torch/lib" \
    "$PYTHON" "$REPO/run_halo_cu121.py" \
      --tau "$TAU" \
      --limit "$LIMIT" \
      --min-span-len 1 \
      --max-spans "$MAX_SPANS" \
      --select-mode "$SELECT_MODE" \
      --diag-mode none \
      --coverage-cap "$COV_CAP" \
      --out "$OUTFILE" \
    2>&1 | tee "$LOGFILE"

  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -eq 0 ]; then
    echo "[done]  $TAG OK"
  else
    echo "[FAIL]  $TAG exit=$EXIT_CODE"
    FAIL=$((FAIL+1))
  fi
  echo ""
done

echo "===== ROUND 7 SERIAL2 DONE ====="
echo "failed=$FAIL / total=${#EXPS[@]}"
