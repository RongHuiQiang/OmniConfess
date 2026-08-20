#!/bin/bash
# Vote-SC phrasing arms: avhbench 645 x Qwen2.5-Omni-7B x v1 prompt x T=0.7, 2 arms.
# 8 dataset shards over GPUs 0-6 (GPU 6 takes two shards); GPU 7 busy with nemotron.
set -u
cd /home/tsq/rhq/omniconfess/OmniConfess
source /home/tsq/rhq/omniconfess/env_ffmpeg.sh
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
mkdir -p outputs/avhbench_vote_sc/temp07_arm1 outputs/avhbench_vote_sc/temp07_arm2
for i in 0 1 2 3 4 5 6; do
  CUDA_VISIBLE_DEVICES=$i $PY avhbench_temp07_gen.py --shard $i --shards 8 \
    > outputs/avhbench_vote_sc/temp07_arm1/shard${i}.log 2>&1 &
done
# shard 7 shares GPU 6 (80 GB, two ~20 GB processes fit)
CUDA_VISIBLE_DEVICES=6 $PY avhbench_temp07_gen.py --shard 7 --shards 8 \
  > outputs/avhbench_vote_sc/temp07_arm1/shard7.log 2>&1 &
wait
echo "ALL TEMP07 SHARDS DONE"
for a in 1 2; do
  wc -l outputs/avhbench_vote_sc/temp07_arm$a/avhbench_shard*.jsonl | tail -1
done
