#!/bin/bash
# run_nemotron_sc_judgment.sh -- SC (self-consistency) on the 3 judgment-family
# datasets for the Nemotron arm. Card 3 ONLY (safety). Serial, resume-safe.
set -u
cd /home/tsq/rhq/omniconfess/OmniConfess
PY=/ultrafast_share/tsq/venvs/nemotron-hf/bin/python
EVALPY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export TMPDIR=/ultrafast_share/tsq/tsq/tmp
export PYTHONPATH=/home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/methods:${PYTHONPATH:-}
export HF_HOME=/ultrafast_share/tsq/tsq/.cache/huggingface
export PYTHONDONTWRITEBYTECODE=1
export NEMO_REPO=/home/tsq/rhq/omniconfess/OmniConfess
export NEMO_KEEP_CACHE=1
mkdir -p "$TMPDIR"

# dataset|expected_rows
QUEUE=("pubmedqa|468" "cmm|721" "phd|566")

for spec in "${QUEUE[@]}"; do
  IFS='|' read -r DS EXP <<< "$spec"
  out=outputs/${DS}_nemotron_sc
  mkdir -p "$out" "$out/time"
  f=$out/${DS}_nemotron_sc.jsonl
  echo "[sc $(date +%T)] START $DS have=$( [ -f "$f" ] && wc -l < "$f" || echo 0)/$EXP"
  CUDA_VISIBLE_DEVICES=3 $PY OmniConfess/methods/base_sc_nemotron_hf.py \
    --datasets "$DS" \
    --data_path "OmniHalluBench/${DS}/${DS}.json" \
    --data_root "$PWD/OmniHalluBench/${DS}/" \
    --output_dir "$out" \
    --time_path "$out/time" \
    --file_name "${DS}_nemotron_sc" \
    --num_rollout 5 --max_new_tokens 512 --prompt_version v1
  echo "[sc $(date +%T)] $DS run ended rows=$( [ -f "$f" ] && wc -l < "$f" || echo 0)/$EXP"
  unset ZHIPU_API_KEY
  ( cd "$out" && $EVALPY /home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/eval/single_eval.py \
      --target_files "$PWD/${DS}_nemotron_sc.jsonl" ) > "$out/score.log" 2>&1
  echo "[sc $(date +%T)] $DS scored -> $out/score.log"
done
echo "[sc $(date +%T)] ALL SC JUDGMENT CELLS DONE"
