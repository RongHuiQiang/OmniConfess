#!/bin/bash
# run_pope_gpu3.sh -- POPE public-benchmark arm on Qwen2.5-Omni-7B, GPU 3 only.
# Waits for the current grid cell on card 3 to finish (3 is reserved in
# /tmp/grid_card_reserve), then runs: VRAM peaks -> base x3 -> rule lambda ->
# CF(rule lambda) x3 -> VCD x3 -> CAD x3 -> scoring. Resume-safe per runner.
# After everything, releases card 3 back to the pool.
set -u
cd /home/tsq/rhq/omniconfess/OmniConfess
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
export CUDA_VISIBLE_DEVICES=3
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
export HF_HUB_OFFLINE=1
POPE=/slow_share/tsq/datasets/pope
OUT=outputs/pope
mkdir -p $OUT $OUT/time outputs/vram
LOG=$OUT/driver.log
exec >> $LOG 2>&1
echo "=== driver start $(date) ==="

# ---- wait for GPU 3 to be free (grid cell 2384204 / any big proc done) ----
while true; do
  used=$(nvidia-smi --id=3 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$used" -lt 2000 ]; then break; fi
  echo "[wait] gpu3 used=${used}MiB $(date +%H:%M:%S)"
  sleep 120
done
echo "[wait] gpu3 free at $(date)"

SETTINGS="random popular adversarial"

# ---- 1. VRAM peaks (3 samples each, minutes) ----
if [ ! -f $OUT/.vram_done ]; then
  for m in base cf; do
    extra=""
    [ "$m" = "cf" ] && extra="--lambda_cf 2.0"
    $PY scripts/log_vram_peak.py --label $m --out outputs/vram/vram_pope.jsonl \
      -- $PY OmniConfess/methods/base_pope_qwen25.py --arm $m $extra \
         --data_path $POPE/pope_random.json --output_dir $OUT --time_path $OUT/time --limit 3 \
      && echo "[vram] $m ok"
  done
  for m in avcd opera; do
    $PY scripts/log_vram_peak.py --label $m --out outputs/vram/vram_pope.jsonl \
      -- $PY OmniConfess/methods/base_${m}.py --datasets pope \
         --data_path $POPE/pope_random.json --data_root $POPE/ \
         --output_dir $OUT/vram_$m --time_path $OUT/time --file_name vram_${m}_probe --max_samples 3 \
         --max_new_tokens 32 --no-record_process \
      && echo "[vram] $m ok"
  done
  $PY scripts/log_vram_peak.py --summarize outputs/vram/vram_pope.jsonl > $OUT/vram_table.txt
  cat $OUT/vram_table.txt
  touch $OUT/.vram_done
fi

# ---- 2. base arm, 3 settings x 500 ----
for s in $SETTINGS; do
  $PY OmniConfess/methods/base_pope_qwen25.py --arm base \
    --data_path $POPE/pope_$s.json --output_dir $OUT --time_path $OUT/time
done

# ---- 3. rule lambda from base yes-rate ----
LAM=$($PY scripts/pope_score.py --rule-lambda $OUT/pope_base_random.jsonl \
        $OUT/pope_base_popular.jsonl $OUT/pope_base_adversarial.jsonl \
        | grep '^\[rule\]' | sed 's/.*lambda_hat = //')
echo "[lambda] rule lambda_hat = $LAM"

# ---- 4. CF arm at rule lambda ----
for s in $SETTINGS; do
  $PY OmniConfess/methods/base_pope_qwen25.py --arm cf --lambda_cf $LAM \
    --data_path $POPE/pope_$s.json --output_dir $OUT --time_path $OUT/time
done

# ---- 5. VCD / CAD baselines ----
for s in $SETTINGS; do
  $PY OmniConfess/methods/base_vcd.py --datasets pope \
    --data_path $POPE/pope_$s.json --data_root $POPE/ \
    --output_dir $OUT --time_path $OUT/time --file_name pope_vcd_$s \
    --max_new_tokens 32 --no-record_process
  $PY OmniConfess/methods/base_cad.py --datasets pope \
    --data_path $POPE/pope_$s.json --data_root $POPE/ \
    --output_dir $OUT --time_path $OUT/time --file_name pope_cad_$s \
    --max_new_tokens 32 --no-record_process
done

# ---- 6. scoring ----
$PY scripts/pope_score.py --dir $OUT --pattern 'pope_*.jsonl' \
  --oracle $OUT/pope_cf_lam${LAM}_random.jsonl $OUT/pope_cf_lam${LAM}_popular.jsonl \
           $OUT/pope_cf_lam${LAM}_adversarial.jsonl > $OUT/score_report.txt 2>&1
cat $OUT/score_report.txt

# ---- 7. release card 3 back to the grid pool ----
sed -i '/^3$/d' /tmp/grid_card_reserve
echo "=== driver done $(date), card 3 released ==="
