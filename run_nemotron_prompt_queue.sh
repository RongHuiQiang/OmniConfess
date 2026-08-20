#!/bin/bash
# run_nemotron_prompt_queue.sh -- [跨骨干矩阵] C: Nemotron mechanism+prompt cells
#   C1 ragtruth  @ Nemotron x contract v4 (extraction-window format contract)
#   C2 haloquest @ Nemotron x contract v2 (avoidance template + min sentence)
#   C3 pubmedqa  @ Nemotron x v5 (v3 criteria system + user w/o "general knowledge")
# SERIAL, one card at a time (farm contention etiquette; the [Nemotron 补格]
# chain owns its own card for base cells). Runner = repo copy of the GPFS
# base_model_nemotron_hf.py with prompt_version choices extended (v4/v5/v6),
# byte-identical otherwise; warm-claim handshake per the chain v2 lesson.
set -u
cd /home/tsq/rhq/omniconfess/OmniConfess
source ./run_claim_helper.sh
PY=/ultrafast_share/tsq/venvs/nemotron-hf/bin/python
export TMPDIR=/ultrafast_share/tsq/tsq/tmp
export PYTHONPATH=/home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/methods:${PYTHONPATH:-}
export HF_HOME=/ultrafast_share/tsq/tsq/.cache/huggingface
export PYTHONDONTWRITEBYTECODE=1
export NEMO_REPO=/home/tsq/rhq/omniconfess/OmniConfess
export NEMO_KEEP_CACHE=1
mkdir -p "$TMPDIR"

QUEUE=("ragtruth|contractv4|v4|435" "haloquest|contractv2|v2|365" "pubmedqa|promptv5|v5|468")

claim_gpu() {   # echoes the gpu id; empty if none free (marker-coordinated)
  for g in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
             | awk -F', ' '$2 < 2500 {print $1}'); do
    if mkdir "$CLAIMDIR/gpu${g}.busy" 2>/dev/null; then
      echo "$g"
      return 0
    fi
  done
  return 0
}

for spec in "${QUEUE[@]}"; do
  IFS='|' read -r DS TAG PV EXP <<< "$spec"
  out=outputs/${DS}_nemotron_${TAG}
  mkdir -p "$out" "$out/time"
  f=$out/${DS}_nemotron_${TAG}.jsonl
  for attempt in $(seq 1 12); do
    n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
    [ "$n" -ge "$EXP" ] && break
    gpu=""
    while [ -z "$gpu" ]; do gpu=$(claim_gpu); [ -z "$gpu" ] && sleep 15; done
    echo "[nemo $(date +%T)] $DS tag=$TAG pv=$PV gpu=$gpu attempt=$attempt have=$n/$EXP"
    claim_run "$gpu" "$PY" 66 "$out/run_gpu${gpu}.log" "$PWD" -- \
      "OmniConfess/methods/base_model_nemotron_hf_pv.py" \
      --datasets "$DS" \
      --data_path "OmniHalluBench/${DS}/${DS}.json" \
      --data_root "$PWD/OmniHalluBench/${DS}/" \
      --output_dir "$out" \
      --time_path "$out/time" \
      --file_name "${DS}_nemotron_${TAG}" \
      --prompt_version "$PV"
    echo "[nemo $(date +%T)] $DS run ended rows=$( [ -f "$f" ] && wc -l < "$f" || echo 0)/$EXP"
    rm -rf "$CLAIMDIR/gpu${gpu}.busy"
    sleep 10
  done
  unset ZHIPU_API_KEY
  ( cd "$out" && $PY /home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/eval/single_eval.py \
      --target_files "$PWD/${DS}_nemotron_${TAG}.jsonl" ) > "$out/score.log" 2>&1
  echo "[nemo $(date +%T)] $DS scored -> $out"
done
echo "[nemo $(date +%T)] ALL NEMOTRON PROMPT CELLS DONE"
