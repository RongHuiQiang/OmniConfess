#!/bin/bash
# _dev_probe.sh -- run the structure probe with the warm-claim handshake so the
# card is defended during the multi-minute load (farm is contended).
set -u
REPO=/home/tsq/rhq/omniconfess/OmniConfess
cd "$REPO"
source ./run_claim_helper.sh
PY=/ultrafast_share/tsq/venvs/nemotron-hf/bin/python
export TMPDIR=/ultrafast_share/tsq/tsq/tmp
export PYTHONPATH=/home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/methods:${PYTHONPATH:-}
export HF_HOME=/ultrafast_share/tsq/tsq/.cache/huggingface
export NEMO_REPO=/home/tsq/rhq/omniconfess/OmniConfess
export NEMO_KEEP_CACHE=1
export PYTHONDONTWRITEBYTECODE=1

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 2000 {print $1; exit}'
}
gpu=""
while [ -z "$gpu" ]; do gpu=$(pick_gpu); [ -z "$gpu" ] && sleep 8; done
echo "[dev] claiming GPU $gpu for probe"
claim_run "$gpu" "$PY" 62 "$REPO/outputs/nemo_dev/probe_struct2.log" "$REPO" -- \
  "OmniConfess/methods/_probe_nemo_struct.py"
echo "[dev] probe done gpu=$gpu"
