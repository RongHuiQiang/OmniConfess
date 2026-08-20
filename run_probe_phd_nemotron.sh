#!/bin/bash
# run_probe_phd_nemotron.sh -- [跨骨干矩阵] A.1: wait for a clean card, then
# run the phd x Nemotron CF probe with the warm-claim handshake (the first
# attempt lost its card mid-.to()-ramp to a concurrent 58GB tenant; same
# failure mode chain_nemotron v1 -> v2 solved).
set -u
REPO=/home/tsq/rhq/omniconfess/OmniConfess
cd "$REPO"
source ./run_claim_helper.sh
PY=/ultrafast_share/tsq/venvs/nemotron-hf/bin/python
export TMPDIR=/ultrafast_share/tsq/tsq/tmp
export HF_HOME=/ultrafast_share/tsq/tsq/.cache/huggingface
export NEMO_KEEP_CACHE=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$REPO/outputs/cf_probe"

OUT=$REPO/outputs/cf_probe/phd_probe_nemotron.jsonl
LOG=$REPO/outputs/cf_probe/phd_probe_nemotron.log

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 12000 {print $1; exit}'
}

for attempt in $(seq 1 15); do
  n=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 )
  if [ "$n" -ge 95 ]; then echo "[watch $(date +%T)] probe already complete ($n)"; break; fi
  gpu=""
  while [ -z "$gpu" ]; do
    gpu=$(pick_gpu)
    [ -z "$gpu" ] && sleep 10
  done
  echo "[watch $(date +%T)] claiming GPU $gpu (attempt $attempt)"
  claim_run "$gpu" "$PY" 66 "$LOG" "$REPO" -- \
    probe_cf_signal_phd_nemotron.py --n_per_cell 25
  n=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 )
  echo "[watch $(date +%T)] probe run ended rows=$n/100"
  [ "$n" -ge 95 ] && break
  sleep 20
done
if [ "$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0)" -ge 95 ]; then
  ( cd "$REPO" && $PY probe_judgment_phd_nemotron.py "$OUT" ) \
      > "$REPO/outputs/cf_probe/phd_probe_nemotron_judgment.txt" 2>&1
  echo "[watch $(date +%T)] judgment written"
else
  echo "[watch $(date +%T)] exhausted attempts"
fi
