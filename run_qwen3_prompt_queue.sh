#!/bin/bash
# run_qwen3_prompt_queue.sh -- [跨骨干矩阵] B: Qwen3-Omni prompt-column cells.
#   B1 pubmedqa@Qwen3 x v5 (v3 criteria system + user w/o "general knowledge")
#   B2 cmm@Qwen3      x v6 (question-type conditional routing, SEE/HEAR)
# Two parallel slots. GPU coordination between the slots = atomic mkdir busy
# markers (pick and mark are one critical section; the marker lives as long
# as the run). External tenants are kept off the multi-minute load ramp by
# the warm-claim handshake (holder 55GB -> 66GB reservation, chain-v2 lesson).
set -u
cd /home/tsq/rhq/omniconfess/OmniConfess
source ./run_claim_helper.sh
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/ultrafast_share/tsq/qwen3omni_outputs/tmp
export PYTHONPATH=/home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/methods:${PYTHONPATH:-}
export NEMO_KEEP_CACHE=1
mkdir -p "$TMPDIR"

QUEUE=/tmp/xbackbone_q3_tasks.$$
cat > "$QUEUE" <<'EOF'
pubmedqa|v5|base_model_qwen3_omni.py
cmm|v6|base_model_qwen3_omni_pv6.py
EOF

take_task() {
  flock "$QUEUE.lock" bash -c "head -1 '$QUEUE'; tail -n +2 '$QUEUE' > '$QUEUE.new' && mv '$QUEUE.new' '$QUEUE'" 2>/dev/null
}

claim_gpu() {   # echoes the gpu id; empty if none free
  for g in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
             | awk -F', ' '$2 < 2500 {print $1}'); do
    if mkdir "$CLAIMDIR/gpu${g}.busy" 2>/dev/null; then
      echo "$g"
      return 0
    fi
  done
  return 0
}

worker() {
  local slot=$1
  while true; do
    local task; task=$(take_task)
    [ -z "$task" ] && break
    IFS='|' read -r DS PV RUNNER <<< "$task"
    local out=outputs/${DS}_qwen3omni_prompt${PV}
    mkdir -p "$out" "$out/time"
    local f=$out/${DS}_qwen3omni_prompt${PV}.jsonl
    local exp=$([ "$DS" = pubmedqa ] && echo 468 || echo 721)
    for attempt in $(seq 1 12); do
      local n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
      [ "$n" -ge "$exp" ] && break
      local gpu=""
      while [ -z "$gpu" ]; do gpu=$(claim_gpu); [ -z "$gpu" ] && sleep 15; done
      echo "[slot$slot $(date +%T)] $DS pv=$PV gpu=$gpu attempt=$attempt have=$n/$exp"
      claim_run "$gpu" "$PY" 66 "$out/run_gpu${gpu}.log" "$PWD" -- \
        "OmniConfess/methods/$RUNNER" \
        --datasets "$DS" \
        --data_path "OmniHalluBench/${DS}/${DS}.json" \
        --data_root "$PWD/OmniHalluBench/${DS}/" \
        --output_dir "$out" \
        --time_path "$out/time" \
        --file_name "${DS}_qwen3omni_prompt${PV}" \
        --prompt_version "$PV" \
        --record_process
      echo "[slot$slot $(date +%T)] $DS run ended rows=$( [ -f "$f" ] && wc -l < "$f" || echo 0)/$exp"
      rm -rf "$CLAIMDIR/gpu${gpu}.busy"
      sleep 10
    done
    unset ZHIPU_API_KEY
    ( cd "$out" && $PY /home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/eval/single_eval.py \
        --target_files "$PWD/${DS}_qwen3omni_prompt${PV}.jsonl" ) \
        > "$out/score.log" 2>&1
    echo "[slot$slot $(date +%T)] $DS scored -> $out"
  done
}

worker 1 & P1=$!
worker 2 & P2=$!
wait $P1 $P2
rm -f "$QUEUE" "$QUEUE.lock" 2>/dev/null
echo "[queue $(date +%T)] ALL QWEN3 PROMPT CELLS DONE"
