#!/bin/bash
# Wrapper to run run_avh_lv2.py using local (non-NFS) python + /dev/shm site-packages.
# Prevents rpc_wait_bit_killable NFS mmap stall that occurs when using
# /slow_share/tsq/venvs/omniconfess/bin/python3 directly.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#     ./run_avh_lv2_local.sh --arm combined --limit 50 --out outputs/lv2_avh_v2/combined_n50.jsonl

PYTHON=/home/tsq/.local/bin/python3.13
DEVSHM_SITE=/dev/shm/py313_site
REPO=$(cd "$(dirname "$0")" && pwd)

# Verify the local python exists and is not on NFS
if [ ! -x "$PYTHON" ]; then
  echo "[ERROR] $PYTHON not found"
  exit 1
fi

# Build PYTHONPATH: devshm site first, then repo root, then OmniConfess subdir
# PYTHONNOUSERSITE=1 prevents loading ~/.local/lib/python3.13/site-packages (on NFS)
export PYTHONNOUSERSITE=1
export PYTHONPATH="$DEVSHM_SITE:$REPO:$REPO/OmniConfess${PYTHONPATH:+:$PYTHONPATH}"

echo "[wrapper] PYTHON=$PYTHON"
echo "[wrapper] DEVSHM_SITE=$DEVSHM_SITE"
echo "[wrapper] PYTHONPATH=$PYTHONPATH"
echo "[wrapper] args: $@"

exec "$PYTHON" "$REPO/run_avh_lv2.py" "$@"
