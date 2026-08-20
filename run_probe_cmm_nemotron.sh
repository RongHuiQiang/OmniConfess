#!/bin/bash
# run_probe_cmm_nemotron.sh -- [跨骨干矩阵] A.3: wait for the cmm@Nemotron
# base (the [Nemotron 补格] chain's last dataset) to be scored, then wait for
# an exclusive card, then run the cmm CF probe + pre-registered judgment.
set -u
REPO=/home/tsq/rhq/omniconfess/OmniConfess
cd "$REPO"
source ./run_claim_helper.sh
PY=/ultrafast_share/tsq/venvs/nemotron-hf/bin/python
export TMPDIR=/ultrafast_share/tsq/tsq/tmp
export HF_HOME=/ultrafast_share/tsq/tsq/.cache/huggingface
export NEMO_KEEP_CACHE=1
export PYTHONDONTWRITEBYTECODE=1

BASE=/ultrafast_share/tsq/tsq/nemotron/outputs/cmm_nemotron/cmm_nemotron_single_scored.jsonl
OUT=$REPO/outputs/cf_probe/cmm_probe_nemotron.jsonl
LOG=$REPO/outputs/cf_probe/cmm_probe_nemotron.log

# 1. wait for the base to exist, be complete (721) and be scored
while true; do
  if [ -f "$BASE" ]; then
    n=$(wc -l < "$BASE")
    # scored == every row carries eval_metrics
    ok=$($PY -c "
import json,sys
rows=[json.loads(l) for l in open('$BASE')]
print(sum(1 for r in rows if r.get('eval_metrics') is not None)==len(rows) and len(rows)>=721)" 2>/dev/null || echo False)
    [ "$ok" = "True" ] && break
  fi
  sleep 120
done
echo "[watch $(date +%T)] cmm base ready -> probing"


claim_gpu() {
  for g in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
             | awk -F', ' '$2 < 12000 {print $1}'); do
    if mkdir "$CLAIMDIR/gpu${g}.busy" 2>/dev/null; then
      echo "$g"
      return 0
    fi
  done
  return 0
}


for attempt in $(seq 1 12); do
  n=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 )
  [ "$n" -ge 95 ] && break
  gpu=""
  while [ -z "$gpu" ]; do gpu=$(claim_gpu); [ -z "$gpu" ] && sleep 10; done
  echo "[watch $(date +%T)] claiming GPU $gpu (attempt $attempt)"
  claim_run "$gpu" "$PY" 66 "$LOG" "$REPO" -- \
    probe_cf_signal_cmm_nemotron.py --n_per_cell 25
  echo "[watch $(date +%T)] run ended rows=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0)/100"
  rm -rf "$CLAIMDIR/gpu${gpu}.busy"
  sleep 20
done
( cd "$REPO" && $PY probe_judgment_cmm_nemotron.py "$OUT" ) \
    > "$REPO/outputs/cf_probe/cmm_probe_nemotron_judgment.txt" 2>&1
echo "[watch $(date +%T)] judgment written"
