#!/usr/bin/env bash
# run_omniconfess.sh — one-command entry for the OmniConfess unified runner.
#
# Thin wrapper around the top-level run_confess.py entry: sets up the runtime
# environment (scripts/env.sh) and the offline model root, then delegates.
#
# Usage:
#   bash confession/run_omniconfess.sh <dataset> [extra args passed through]
#
#   <dataset> = phd | cmm | pubmedqa | ragtruth | haloquest | avhbench
#
# Examples:
#   bash confession/run_omniconfess.sh phd
#   bash confession/run_omniconfess.sh haloquest --halo_shard 0 --halo_nshards 8
#   bash confession/run_omniconfess.sh pubmedqa --dry_run
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"

# Qwen2.5-Omni-7B weights live on tmpfs; point the model resolver at them so the
# offline load_model() default path finds a local checkpoint.
export MMHALU_MODEL_ROOT="${MMHALU_MODEL_ROOT_OMNI:-/dev/shm}"

exec "$PY" "$REPO/run_confess.py" "$@"
