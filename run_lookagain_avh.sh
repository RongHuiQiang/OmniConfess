#!/bin/bash
# Look-Again resolution/frame ladder for avhbench base_model (Qwen2.5-Omni).
# Usage: bash run_lookagain_avh.sh <FRAME_SIZE> <NUM_FRAMES> <OUTDIR_NAME>
#   e.g. bash run_lookagain_avh.sh 384 8 avhbench_res384
# Shards: the existing 4-way split (avhbench_shard{0..3}of4.json) on GPUs 4-7.
set -u
SIZE=${1:?frame size}
NFR=${2:?num frames}
OUT=${3:?output dir name}
cd /home/tsq/rhq/omniconfess/OmniConfess
source /home/tsq/rhq/omniconfess/env_ffmpeg.sh
export MMHALU_MODEL_ROOT=/ultrafast_share/tsq/MODELS
export OMNI_FRAME_SIZE=$SIZE
export OMNI_NUM_FRAMES=$NFR
PY=/ultrafast_share/tsq/venvs/omniconfess/bin/python
mkdir -p outputs/$OUT/time
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((4+i)) $PY OmniConfess/methods/base_model.py \
    --datasets avhbench \
    --data_path OmniHalluBench/avhbench/avhbench_shard${i}of4.json \
    --data_root ./OmniHalluBench/ \
    --output_dir outputs/$OUT \
    --file_name avhbench_shard${i} \
    --time_path outputs/$OUT/time \
    > outputs/$OUT/shard${i}.log 2>&1 &
done
wait
echo "ALL SHARDS DONE ($OUT size=$SIZE frames=$NFR)"
