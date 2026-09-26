#!/bin/bash
# run_freeform_round2.sh — Round 2 span 选择策略网格
#
# Round 1 诊断: tau=0 勉强 +0.12，其余全 -5%~-7%。
# 根因: select_spans 选入了太多正确 token（单 token 噪声 span、每样本 span 数过多）。
# Round 2 在三个维度上收紧:
#   策略A — 提高 tau 阈值(减少标记量)
#   策略B — min_span_len 过滤短噪声 span (单 token span 质量差)
#   策略C — max_spans 限制每样本最多修订数量，按 rho 均值保留最高危的
#
# 实验矩阵 (7 卡并行，卡 0-6):
#   GPU0  R2-A2: tau=3.0  min_span=1  max_spans=0   (高阈值对照)
#   GPU1  R2-A3: tau=5.0  min_span=1  max_spans=0   (更高阈值)
#   GPU2  R2-B1: tau=0    min_span=2  max_spans=0   (过滤单 token span)
#   GPU3  R2-B2: tau=0    min_span=3  max_spans=0   (更激进过滤)
#   GPU4  R2-C1: tau=0    min_span=1  max_spans=2   (每样本最多 2 个 span)
#   GPU5  R2-C2: tau=1.0  min_span=2  max_spans=2   (组合策略)
#   GPU6  R2-C3: tau=0    min_span=2  max_spans=2   (组合策略)
#
# 用法: bash run_freeform_round2.sh [--limit N]  (默认 N=50)

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python3
MODEL_DIR=/dev/shm
OUTDIR="$REPO/outputs/freeform_grid"
LOGDIR="$OUTDIR/logs_r2"
mkdir -p "$OUTDIR" "$LOGDIR"

LIMIT=50
for arg in "$@"; do
  case $arg in
    --limit=*) LIMIT="${arg#*=}" ;;
    --limit)   shift; LIMIT="$1"; shift ;;
  esac
done

echo "[round2] limit=$LIMIT  model=$MODEL_DIR/Qwen2.5-Omni-7B"

# 实验配置: (GPU, tau, min_span_len, max_spans, tag)
declare -a EXPS=(
  "0  3.0  1  0  R2-A2"
  "1  5.0  1  0  R2-A3"
  "2  0    2  0  R2-B1"
  "3  0    3  0  R2-B2"
  "4  0    1  2  R2-C1"
  "5  1.0  2  2  R2-C2"
  "6  0    2  2  R2-C3"
)

PIDS=()
TAGS=()

for spec in "${EXPS[@]}"; do
  read -r GPU TAU MIN_SPAN MAX_SPANS TAG <<< "$spec"
  OUTFILE="$OUTDIR/haloquest_${TAG}.jsonl"
  LOGFILE="$LOGDIR/haloquest_${TAG}.log"

  CUDA_VISIBLE_DEVICES=$GPU \
  MMHALU_MODEL_ROOT=$MODEL_DIR \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/tmp/pip-target-b9a06nc9/lib/python \
  LD_LIBRARY_PATH=/dev/shm/so_mirror:/tmp/pip-target-b9a06nc9/lib/python/torch/lib \
    "$PYTHON" "$REPO/run_halo_cu121.py" \
      --tau "$TAU" \
      --limit "$LIMIT" \
      --min-span-len "$MIN_SPAN" \
      --max-spans "$MAX_SPANS" \
      --out "$OUTFILE" \
    > "$LOGFILE" 2>&1 &

  PIDS+=($!)
  TAGS+=("$TAG")
  echo "[launch] GPU=$GPU  $TAG  tau=$TAU  min_span=$MIN_SPAN  max_spans=$MAX_SPANS  pid=${PIDS[-1]}"
done

echo ""
echo "[round2] 全部 ${#PIDS[@]} 个进程已启动，等待完成..."
FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[done]  ${TAGS[$i]}  OK"
  else
    echo "[FAIL]  ${TAGS[$i]}  exit=$?"
    FAIL=$((FAIL+1))
  fi
done

echo ""
echo "===== ROUND 2 SUMMARY ====="
echo "failed=$FAIL / total=${#PIDS[@]}"
echo ""
printf "%-10s %-8s %-10s %-10s %-10s %-10s %-10s\n" \
  "tag" "tau" "min_span" "max_spans" "base_f1" "repair_f1" "delta_f1"
printf '%0.s-' {1..75}; echo ""

for spec in "${EXPS[@]}"; do
  read -r GPU TAU MIN_SPAN MAX_SPANS TAG <<< "$spec"
  F="$OUTDIR/haloquest_${TAG}.jsonl"
  if [ -f "$F" ]; then
    python3 - "$F" "$TAG" "$TAU" "$MIN_SPAN" "$MAX_SPANS" <<'PYEOF'
import sys, json
fpath, tag, tau, msl, ms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
rows = [json.loads(l) for l in open(fpath) if l.strip()]
ok = [r for r in rows if r.get("status") == "success"]
if ok:
    base = sum(r["base_f1"] for r in ok) / len(ok)
    rep  = sum(r["repaired_f1"] for r in ok) / len(ok)
    delta = rep - base
    print(f"{tag:<10} {tau:<8} {msl:<10} {ms:<10} {base:<10.2f} {rep:<10.2f} {delta:+.2f}")
else:
    print(f"{tag:<10} {tau:<8} {msl:<10} {ms:<10} {'NO_OK_ROWS'}")
PYEOF
  else
    printf "%-10s %-8s %-10s %-10s %s\n" "$TAG" "$TAU" "$MIN_SPAN" "$MAX_SPANS" "OUTPUT MISSING"
  fi
done
