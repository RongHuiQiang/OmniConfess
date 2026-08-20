#!/bin/bash
# Output-contract mechanism migration to Qwen3-Omni-30B-A3B-Instruct
# (descriptive cells; [机制迁移@Qwen3] queue, 2026-08-16).
#
# One process per card (thinker ~62GB resident). Uses the eos-fixed repo
# runner OmniConfess/methods/base_model_qwen3_omni.py (fix back-written from
# the GPFS copy 2026-08-16; do not run the pre-fix GPFS-era behavior without
# eos_token_id -- see the runner docstring).
#
# Usage:
#   bash run_qwen3omni_contract.sh <dataset> <outtag> <prompt_version> <gpu_list|auto>
#     dataset : ragtruth | haloquest | avhbench
#     outtag  : output subdir suffix, e.g. contractv4 -> outputs/<ds>_qwen3omni_<outtag>/
#     gpu_list: comma-separated, e.g. "0,1,2" (one shard process per card,
#               thinker ~62GB needs a fully free 80GB card); or "auto" =
#               use every card with <1.5GB foreign allocation
# Example:
#   bash run_qwen3omni_contract.sh ragtruth contractv4 v4 auto
set -u
cd "$(dirname "$0")"

DATASET=$1
OUTTAG=$2
PV=$3
GPUS=${4:-auto}

if [ "$GPUS" = "auto" ]; then
  GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$2 < 1500 {print $1}' | paste -sd, -)
fi
IFS=',' read -ra GPUA <<< "$GPUS"
NGPU=${#GPUA[@]}
if [ "$NGPU" -eq 0 ] || [ -z "$GPUS" ]; then echo "[abort] no free GPU"; exit 1; fi
echo "[cfg] dataset=$DATASET tag=$OUTTAG prompt=$PV gpus=$GPUS (n=$NGPU)"

PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
export PYTHONDONTWRITEBYTECODE=1
# Python tempfile (numba/librosa JIT, tokenizer fast) off the 94%-full root fs
export TMPDIR=/ultrafast_share/tsq/qwen3omni_outputs/tmp
mkdir -p "$TMPDIR"
# avhbench needs the bundled static ffmpeg for librosa/audioread (mp4 audio)
source /home/tsq/rhq/omniconfess/env_ffmpeg.sh

OUT=outputs/${DATASET}_qwen3omni_${OUTTAG}
mkdir -p "$OUT" "$OUT/time"

# ── foreign-allocation guard: thinker needs ~62GB of an 80GB card ──────────
for i in "${GPUA[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i")
  if [ "$used" -gt 15000 ]; then echo "[abort] GPU $i has ${used}MiB foreign alloc"; exit 1; fi
done

# shard next to the source annotation (shard_data.py writes _shardKofN.json)
$PY ../shard_data.py "OmniHalluBench/${DATASET}/${DATASET}.json" "$NGPU" >/dev/null

si=0
for i in "${GPUA[@]}"; do
  CUDA_VISIBLE_DEVICES=$i $PY OmniConfess/methods/base_model_qwen3_omni.py \
      --datasets "$DATASET" \
      --data_path "OmniHalluBench/${DATASET}/${DATASET}_shard${si}of${NGPU}.json" \
      --data_root "$PWD/OmniHalluBench/${DATASET}/" \
      --output_dir "$OUT" \
      --time_path "$OUT/time" \
      --file_name "${DATASET}_shard${si}_gpu${i}" \
      --prompt_version "$PV" \
      --record_process \
      > "$OUT/shard${si}_gpu${i}.log" 2>&1 &
  si=$((si+1))
done
wait

# merge shard jsonls, deduping by id (retries may shard differently; keep first)
$PY - "$OUT" "$DATASET" "$OUTTAG" <<'PYEOF'
import json, sys, glob, os
out, ds, tag = sys.argv[1], sys.argv[2], sys.argv[3]
dst = os.path.join(out, f"{ds}_qwen3omni_{tag}_full.jsonl")
seen, rows = set(), []
for fp in sorted(glob.glob(os.path.join(out, f"{ds}_shard*.jsonl"))):
    for line in open(fp):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r['id'] in seen: continue
        seen.add(r['id']); rows.append(r)
with open(dst, 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"[merge] {dst}: {len(rows)} rows, {len(seen)} unique ids")
PYEOF
N=$(wc -l < "$OUT/${DATASET}_qwen3omni_${OUTTAG}_full.jsonl")
echo "[check] lines=$N"
echo "[qwen3omni x $DATASET pv=$PV] done -> $OUT/${DATASET}_qwen3omni_${OUTTAG}_full.jsonl ($(wc -l < "$OUT/${DATASET}_qwen3omni_${OUTTAG}_full.jsonl") lines)"

# local Token-F1 scoring (GAVIE frozen: no ZHIPU_API_KEY -> judge skipped,
# Token-F1 computed locally by eval/single_eval.py; writes descriptive CSV +
# *_single_scored.jsonl in CWD)
unset ZHIPU_API_KEY
$PY OmniConfess/eval/single_eval.py --target_files "$OUT/${DATASET}_qwen3omni_${OUTTAG}_full.jsonl" 2>&1 | tail -5
