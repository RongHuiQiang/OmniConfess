#!/bin/bash
# run_blindspot_nemotron_gpu7.sh -- third-backbone blindspot cells:
# wait for GPU7 (daemon-reserved, current cmm cell draining), smoke 3, then
# full n=250 phd + cmm via the warm-claim launcher.
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
mkdir -p "$REPO/outputs/cf_probe/blindspot_nemotron"

OUTD=$REPO/outputs/cf_probe/blindspot_nemotron

free7() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7
}

# ---- wait for the current cell on card 7 to drain (<2000MiB) ----
while [ "$(free7)" -ge 2000 ]; do sleep 60; done
echo "[watch $(date +%T)] GPU7 free ($(free7)MiB), starting smoke"

# ---- smoke: 3 samples each family ----
for fam in phd cmm; do
  claim_run 7 "$PY" 66 "$OUTD/${fam}_smoke.log" "$REPO" -- \
    probe_cf_nemotron_blindspot.py --family $fam --n 3 --limit 3 \
    --out "$OUTD/_smoke_${fam}.jsonl"
done
for fam in phd cmm; do
  n=$(wc -l < "$OUTD/_smoke_${fam}.jsonl")
  echo "[smoke] $fam rows=$n"
  if [ "$n" -lt 3 ]; then echo "[smoke] $fam FAILED, aborting"; exit 1; fi
done
echo "[smoke] OK"

# ---- full runs (resume-aware) ----
claim_run 7 "$PY" 66 "$OUTD/phd_run.log" "$REPO" -- \
  probe_cf_nemotron_blindspot.py --family phd --n 250
claim_run 7 "$PY" 66 "$OUTD/cmm_run.log" "$REPO" -- \
  probe_cf_nemotron_blindspot.py --family cmm --n 250

for fam in phd cmm; do
  echo "[final] $fam rows=$(wc -l < "$OUTD/${fam}_blindspot_nemotron.jsonl")"
done
echo "[watch $(date +%T)] all done; GPU7 now: $(free7)MiB"
