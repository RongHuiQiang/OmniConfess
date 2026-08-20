#!/bin/bash
# run_xbackbone_sequencer.sh -- [跨骨干矩阵] staggered launcher:
#   A phd-Nemotron probe (already launched separately; waits for its card)
#     -> B Qwen3 prompt cells (2 slots) once the probe's model is loaded
#     -> C Nemotron prompt cells (serial 1 card) once B has claimed cards.
# Etiquette vs other agents: everything single-card, waits for an
# essentially-free card, retries on farm contention / GPFS hiccups.
set -u
R=/home/tsq/rhq/omniconfess/OmniConfess

# 1. wait for the probe to be up (model loaded = card claimed and held)
while ! grep -q '\[model\] loaded' "$R/outputs/cf_probe/phd_probe_nemotron.log" 2>/dev/null; do
  sleep 30
done
echo "[seq $(date +%T)] probe model loaded -> starting B (Qwen3 prompt queue)"
nohup bash "$R/run_qwen3_prompt_queue.sh" > "$R/outputs/q3_prompt_queue.log" 2>&1 &
BPID=$!

# 2. wait until B has at least one runner up (a run_gpu*.log with a response,
#    i.e. its card is claimed) or 90 min pass
deadline=$(( $(date +%s) + 5400 ))
while [ $(date +%s) -lt $deadline ]; do
  for ds in pubmedqa cmm; do
    for lg in "$R"/outputs/${ds}_qwen3omni_promptv*/run_gpu*.log; do
      if [ -f "$lg" ] && grep -q 'response\[:120\]' "$lg" 2>/dev/null; then
        break 2
      fi
    done
  done
  sleep 60
done
echo "[seq $(date +%T)] starting C (Nemotron prompt queue, serial)"
nohup bash "$R/run_nemotron_prompt_queue.sh" > "$R/outputs/nemo_prompt_queue.log" 2>&1 &
CPID=$!
wait $BPID $CPID
echo "[seq $(date +%T)] B and C queues finished"
