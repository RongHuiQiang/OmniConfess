#!/bin/bash
# run_freeform_round7_serial.sh — Round 7 失败 arm 的串行重跑
#
# 原始并行跑时 GPU 4-7 属于他人 sglang 进程，进程被外部杀死。
# 本脚本在 GPU 0-3 上串行跑剩余 5 个 arm（每次只用一张卡），避免争抢。
#
# 使用前确认 R7-COV60 已完成（GPU 0 空闲）:
#   ps aux | grep run_halo_cu121
#
# 用法: bash run_freeform_round7_serial.sh [--limit N]

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python3
MODEL_DIR=/dev/shm
OUTDIR="$REPO/outputs/freeform_grid/round7"
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

LIMIT=50
for arg in "$@"; do
  case $arg in
    --limit=*) LIMIT="${arg#*=}" ;;
    --limit)   shift; LIMIT="$1"; shift ;;
  esac
done

echo "[round7-serial] limit=$LIMIT  model=$MODEL_DIR/Qwen2.5-Omni-7B"

# 格式: "GPU  TAG  TAU  MAX_SPANS  COV_CAP  SELECT_MODE"
# 只包含上次失败的 arm，按 GPU 顺序在 GPU 0-3 上串行运行
declare -a EXPS=(
  "1  R7-MAX1-COV60  0    1  0.60  rho"
  "2  R7-TAU10       1.0  2  0     rho"
  "3  R7-TAU20       2.0  2  0     rho"
  "0  R7-MAX1-TAU10  1.0  1  0     rho"
  "1  R7-TOPK        0    2  0     topk"
)

FAIL=0
for spec in "${EXPS[@]}"; do
  read -r GPU TAG TAU MAX_SPANS COV_CAP SELECT_MODE <<< "$spec"
  OUTFILE="$OUTDIR/haloquest_${TAG}.jsonl"
  LOGFILE="$LOGDIR/haloquest_${TAG}.log"

  # 跳过已有完整输出（50 行）的 arm
  if [ -f "$OUTFILE" ]; then
    n=$(wc -l < "$OUTFILE")
    if [ "$n" -ge "$LIMIT" ]; then
      echo "[skip]   $TAG already has $n rows"
      continue
    fi
    echo "[resume] $TAG has $n rows, will re-run"
    rm -f "$OUTFILE"
  fi

  echo "[launch] GPU=$GPU  $TAG  tau=$TAU  max_spans=$MAX_SPANS  cov_cap=$COV_CAP  select=$SELECT_MODE"

  CUDA_VISIBLE_DEVICES=$GPU \
  MMHALU_MODEL_ROOT=$MODEL_DIR \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/tmp/pip-target-b9a06nc9/lib/python \
  LD_LIBRARY_PATH=/dev/shm/so_mirror:/tmp/pip-target-b9a06nc9/lib/python/torch/lib \
    "$PYTHON" "$REPO/run_halo_cu121.py" \
      --tau "$TAU" \
      --limit "$LIMIT" \
      --min-span-len 1 \
      --max-spans "$MAX_SPANS" \
      --select-mode "$SELECT_MODE" \
      --diag-mode none \
      --coverage-cap "$COV_CAP" \
      --out "$OUTFILE" \
    2>&1 | tee "$LOGFILE"

  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -eq 0 ]; then
    echo "[done]  $TAG OK"
  else
    echo "[FAIL]  $TAG exit=$EXIT_CODE"
    FAIL=$((FAIL+1))
  fi
  echo ""
done

echo "===== ROUND 7 SERIAL SUMMARY ====="
echo "failed=$FAIL / total=${#EXPS[@]}"
echo ""
printf "%-16s %-5s %-5s %-6s %-8s %-10s %-10s %-8s %-8s\n" \
  "tag" "tau" "cap" "delta" "d_edited" "base_f1" "rep_f1" "cov_skip%" "n_edited"
printf '%0.s-' {1..90}; echo ""

for spec in "${EXPS[@]}"; do
  read -r GPU TAG TAU MAX_SPANS COV_CAP SELECT_MODE <<< "$spec"
  F="$OUTDIR/haloquest_${TAG}.jsonl"
  if [ -f "$F" ]; then
    python3 - "$F" "$TAG" "$TAU" "$COV_CAP" "$SELECT_MODE" <<'PYEOF'
import sys, json
fpath, tag, tau, cov_cap, select_mode = sys.argv[1:]
rows = [json.loads(l) for l in open(fpath) if l.strip()]
ok = [r for r in rows if r.get("status") == "success"]
if ok:
    base  = sum(r["base_f1"]     for r in ok) / len(ok)
    rep   = sum(r["repaired_f1"] for r in ok) / len(ok)
    delta = rep - base
    edited = [r for r in ok if r.get("n_spans", 0) > 0]
    n_skip = sum(1 for r in edited if r.get("coverage_cap_skip", False))
    frac_skip = n_skip / len(edited) if edited else 0.0
    actually_edited = [r for r in edited if not r.get("coverage_cap_skip", False)]
    d_edited = (
        sum(r["repaired_f1"] - r["base_f1"] for r in actually_edited) / len(actually_edited)
        if actually_edited else 0.0
    )
    print(f"{tag:<16} {tau:<5} {cov_cap:<5} {delta:+6.2f}  {d_edited:+7.2f}  "
          f"{base:<10.2f} {rep:<10.2f} {frac_skip*100:6.1f}%   {len(actually_edited):<8}")
else:
    print(f"{tag:<16} NO_OK_ROWS")
PYEOF
  else
    printf "%-16s %s\n" "$TAG" "OUTPUT MISSING"
  fi
done
