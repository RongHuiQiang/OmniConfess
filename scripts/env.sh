# scripts/env.sh -- single source of truth for this box's runtime.
#
# WHAT THIS IS (and is not):
#   * This file documents the *local-machine* runtime contract that the batch
#     runner (scripts/run_dc_shard.sh) already uses. Paths under /dev/shm are
#     this box's tmpfs staging of the deps; they do NOT ship with the repo.
#   * The portable, publishable dependency spec is requirements.txt. That is
#     what an open-source user installs. This file is just the local wiring so
#     nobody has to rediscover the PYTHONPATH / LD_LIBRARY_PATH recipe.
#
# USAGE:  source scripts/env.sh   (from the repo root), then e.g.
#   $PY confession/cf_confess_phd.py --dry_run --data_path ... --lambda_cf 0
#   $PY scripts/avh_confess_dualchannel.py --start 0 --n 10 ...
#
# Deps live on tmpfs (/dev/shm, wiped on reboot). If import torch fails after a
# reboot, the site-packages/so_mirror trees need re-staging (they mirror the
# versions pinned in requirements.txt: torch 2.6.0+cu124, transformers 4.57.6).

# repo root = parent of this scripts/ dir (works regardless of invocation cwd)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export REPO

# interpreter: system CPython 3.13 (matches the /dev/shm/site-packages ABI)
export PY=/home/tsq/.local/bin/python3.13

# deps on tmpfs + the shared/ method package (double-nested: OmniConfess/OmniConfess/methods)
export PYTHONPATH="/dev/shm/site-packages:$REPO/OmniConfess/methods"
export LD_LIBRARY_PATH="/dev/shm/so_mirror"

# offline: never touch the network; ~/.cache is a broken symlink so redirect it
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/dev/shm/.cache
export UV_CACHE_DIR=/dev/shm/.cache/uv

# CUDA: all cards 0-7 are usable on this box (updated 2026-09-27)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# model weights (pipeline's hard-coded /ultrafast_share path is dead; use this)
export MMHALU_MODEL_ROOT=/tmp

echo "[env] PY=$PY  torch=$("$PY" -c 'import torch;print(torch.__version__,torch.cuda.is_available())' 2>/dev/null)  REPO=$REPO"
