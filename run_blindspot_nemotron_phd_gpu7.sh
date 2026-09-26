#!/bin/bash
# run_blindspot_nemotron_phd_gpu7.sh -- FINAL phd-only card-7 batch:
# smoke already ran separately; wait for the smoke process to exit and the
# card to drain, then run the full phd n=250 blindspot probe, then analyze.
set -u
REPO=/home/tsq/rhq/omniconfess/OmniConfess
cd "$REPO"
source ./run_claim_helper.sh
PY=/ultrafast_share/tsq/venvs/nemotron-hf/bin/python
export TMPDIR=/ultrafast_share/tsq/tsq/tmp
export HF_HOME=/ultrafast_share/tsq/tsq/.cache/huggingface
export NEMO_KEEP_CACHE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=7
OUTD=$REPO/outputs/cf_probe/blindspot_nemotron

free7() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7; }

# wait for any current probe/claim process on card 7 to finish
while pgrep -f "probe_cf_nemotron_blindspot" > /dev/null; do sleep 30; done
while [ "$(free7)" -ge 2000 ]; do sleep 30; done
echo "[watch $(date +%T)] GPU7 free ($(free7)MiB), starting full phd n=250"

claim_run 7 "$PY" 66 "$OUTD/phd_run.log" "$REPO" -- \
  probe_cf_nemotron_blindspot.py --family phd --n 250
echo "[final $(date +%T)] phd rows=$(wc -l < "$OUTD/phd_blindspot_nemotron.jsonl" 2>/dev/null || echo 0)"

# zero-GPU analysis immediately (cmm falls back to the n=100 signal probe)
cd "$REPO/outputs/problem_structure_fig"
/usr/bin/python3.10 unified_cf_robustness_nemotron.py > "$OUTD/analysis.log" 2>&1
echo "[watch $(date +%T)] all done; GPU7 now: $(free7)MiB"
